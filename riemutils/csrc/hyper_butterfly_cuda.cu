// riemutils/csrc/hyper_butterfly_cuda.cu

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include "hyper_butterfly.h"

// -----------------------------------------------------------------------------
// 오류 체크 매크로
// -----------------------------------------------------------------------------
#define CHECK_CUDA_CONTIGUOUS(x)                                    \
    TORCH_CHECK((x).device().is_cuda(), #x " must be CUDA tensor"); \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

#define CUDA_CHECK(err)                               \
    do                                                \
    {                                                 \
        auto e = (err);                               \
        TORCH_CHECK(e == cudaSuccess, "CUDA error: ", \
                    cudaGetErrorString(e));           \
    } while (0)

static constexpr float EPS = 1e-7f;

// -----------------------------------------------------------------------------
// atanh 헬퍼 (인자 클램핑 적용)
// -----------------------------------------------------------------------------
__device__ __forceinline__ float atanh_device(float x)
{
    x = fminf(fmaxf(x, -1.0f + 1e-6f), 1.0f - 1e-6f);
    return 0.5f * logf((1.0f + x) / (1.0f - x));
}

// -----------------------------------------------------------------------------
// 다음 2의 거듭제곱 계산 (MSVC/Clang 호환)
// -----------------------------------------------------------------------------
static inline int next_pow2(int v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

// -----------------------------------------------------------------------------
// 1) 로그 맵 커널 (클램핑 + 안전 분모)
// -----------------------------------------------------------------------------
template <typename scalar_t>
__global__ void log_map_origin_kernel(
    const scalar_t *__restrict__ x,
    scalar_t *__restrict__ out,
    float c,
    int batch,
    int dim)
{
    extern __shared__ float shared_norm[];
    int tid = threadIdx.x, bid = blockIdx.x;

    // 1) shared 초기화
    if (tid == 0)
        shared_norm[0] = 0.f;
    __syncthreads();

    // 2) partial sum of squares
    float sum = 0.f;
    const scalar_t *xb = x + bid * dim;
    for (int i = tid; i < dim; i += blockDim.x)
        sum += xb[i] * xb[i];
    // warp‐reduce
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if ((tid & (warpSize - 1)) == 0)
        atomicAdd(&shared_norm[0], sum);
    __syncthreads();

    // 3) 안정화
    if (tid == 0)
        shared_norm[0] = fmaxf(shared_norm[0], EPS);
    __syncthreads();

    // 4) clamp √c·‖x‖, 안전 분모
    float norm = sqrtf(shared_norm[0]);
    float scn = sqrtf(c) * norm;
    scn = fminf(fmaxf(scn, 1e-6f), 0.999999f); // 하한/상한 클램핑
    float denom = scn + 1e-6f;                 // 분모 안전화
    float numer = atanh_device(scn);
    float factor = numer / denom;

    // 5) output
    scalar_t *yb = out + bid * dim;
    for (int i = tid; i < dim; i += blockDim.x)
        yb[i] = factor * xb[i];
}

// -----------------------------------------------------------------------------
// 2) 지수 맵 커널 (클램핑 + 안전 분모)
// -----------------------------------------------------------------------------
template <typename scalar_t>
__global__ void exp_map_origin_kernel(
    const scalar_t *__restrict__ v,
    scalar_t *__restrict__ out,
    float c,
    int batch,
    int dim)
{
    extern __shared__ float shared_norm[];
    int tid = threadIdx.x, bid = blockIdx.x;

    if (tid == 0)
        shared_norm[0] = 0.f;
    __syncthreads();

    float sum = 0.f;
    const scalar_t *vb = v + bid * dim;
    for (int i = tid; i < dim; i += blockDim.x)
        sum += vb[i] * vb[i];
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if ((tid & (warpSize - 1)) == 0)
        atomicAdd(&shared_norm[0], sum);
    __syncthreads();

    if (tid == 0)
        shared_norm[0] = fmaxf(shared_norm[0], EPS);
    __syncthreads();

    float norm = sqrtf(shared_norm[0]);
    float scn = sqrtf(c) * norm;
    scn = fminf(fmaxf(scn, 1e-6f), 10.0f); // 상한 클램핑 (max=10)
    float denom = scn + 1e-3f;             // 분모 여유
    float numer = tanhf(scn);
    float factor = numer / denom;

    scalar_t *yb = out + bid * dim;
    for (int i = tid; i < dim; i += blockDim.x)
        yb[i] = factor * vb[i];
}

// -----------------------------------------------------------------------------
// 3) Butterfly 레이어 커널 (변경 없음)
// -----------------------------------------------------------------------------
template <typename scalar_t>
__global__ void butterfly_layer_kernel(
    const scalar_t *__restrict__ input,
    scalar_t *__restrict__ output,
    const scalar_t *__restrict__ params,
    int batch,
    int dim,
    int layer_idx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int bs = 1 << layer_idx;
    int nb = dim / (2 * bs);

    if (idx >= batch * dim)
        return;
    while (idx < batch * dim)
    {
        int b = idx / dim, f = idx % dim;
        int blk = (f / (2 * bs)) % nb;
        int loc = f % (2 * bs), off = loc % bs;
        int pidx = blk * 2;

        float a = params[pidx], bb = params[pidx + 1];
        int base = b * dim + blk * 2 * bs;
        float x1 = input[base + off], x2 = input[base + off + bs];
        bool hi = loc >= bs;
        output[idx] = hi
                          ? (-bb * x1 + a * x2)
                          : (a * x1 + bb * x2);
        idx += stride;
    }
}

// -----------------------------------------------------------------------------
// 4) CUDA 래퍼 함수들
// -----------------------------------------------------------------------------
torch::Tensor log_map_origin_cuda(torch::Tensor x, float c)
{
    CHECK_CUDA_CONTIGUOUS(x);
    int batch = x.size(0), dim = x.size(1);
    auto out = torch::empty_like(x);
    int th = std::min(dim, 1024), sh = sizeof(float);
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "log_map_origin_cuda", ([&]
                                                                        { log_map_origin_kernel<scalar_t><<<batch, th, sh>>>(
                                                                              x.data_ptr<scalar_t>(),
                                                                              out.data_ptr<scalar_t>(),
                                                                              c, batch, dim); }));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return out;
}

torch::Tensor exp_map_origin_cuda(torch::Tensor v, float c)
{
    CHECK_CUDA_CONTIGUOUS(v);
    int batch = v.size(0), dim = v.size(1);
    auto out = torch::empty_like(v);
    int th = std::min(dim, 1024), sh = sizeof(float);
    AT_DISPATCH_FLOATING_TYPES(v.scalar_type(), "exp_map_origin_cuda", ([&]
                                                                        { exp_map_origin_kernel<scalar_t><<<batch, th, sh>>>(
                                                                              v.data_ptr<scalar_t>(),
                                                                              out.data_ptr<scalar_t>(),
                                                                              c, batch, dim); }));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return out;
}

std::vector<torch::Tensor> hyper_butterfly_cuda(
    torch::Tensor x,
    torch::Tensor params,
    torch::Tensor /*args*/,
    float c,
    int L)
{
    CHECK_CUDA_CONTIGUOUS(x);
    CHECK_CUDA_CONTIGUOUS(params);

    int batch = x.size(0), dim = x.size(1), orig = dim;
    // 1) 2^n 패딩
    int pd = next_pow2(dim);
    if (pd != dim)
    {
        auto xp = torch::zeros({batch, pd}, x.options());
        xp.narrow(1, 0, dim).copy_(x);
        x = xp;
        dim = pd;
    }
    // 2) params 길이 확인
    int log2d = int(log2f((float)dim)), need = 0;
    for (int i = 0; i < L; ++i)
    {
        int li = i % log2d;
        need += (dim / (2 * (1 << li))) * 2;
    }
    TORCH_CHECK(params.size(0) >= need, "not enough params");

    // 3) 버퍼 준비
    auto u = torch::empty_like(x), v = torch::empty_like(x), y = torch::empty_like(x);

    // 4) 로그 맵
    {
        int th = std::min(dim, 1024), sh = sizeof(float);
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "logmap", ([&]
                                                               { log_map_origin_kernel<scalar_t><<<batch, th, sh>>>(
                                                                     x.data_ptr<scalar_t>(), u.data_ptr<scalar_t>(),
                                                                     c, batch, dim); }));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    // 5) Butterfly 레이어
    {
        int th = 256, bl = std::min(1024, (batch * dim + th - 1) / th);
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "butterfly", ([&]
                                                                  {
            int ofs=0;
            for(int i=0; i<L; ++i) {
                int li = i % log2d;
                butterfly_layer_kernel<scalar_t><<<bl,th>>>(
                    (i%2==0 ? u.data_ptr<scalar_t>() : v.data_ptr<scalar_t>()),
                    (i%2==0 ? v.data_ptr<scalar_t>() : u.data_ptr<scalar_t>()),
                    params.data_ptr<scalar_t>() + ofs,
                    batch, dim, li
                );
                CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
                ofs += (dim/(2*(1<<li))) * 2;
            }
            if (L%2==1) u.copy_(v); }));
    }
    // 6) 지수 맵
    {
        int th = std::min(dim, 1024), sh = sizeof(float);
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "expmap", ([&]
                                                               { exp_map_origin_kernel<scalar_t><<<batch, th, sh>>>(
                                                                     u.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),
                                                                     c, batch, dim); }));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    // 7) 원본 차원으로 슬라이스
    if (dim != orig)
    {
        auto yo = y.narrow(1, 0, orig).contiguous();
        return {yo, u, v};
    }
    return {y, u, v};
}
