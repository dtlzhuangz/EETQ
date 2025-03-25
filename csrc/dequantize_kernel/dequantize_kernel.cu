#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include "cutlass/array.h"
#include "dequantize_kernel.h"

template<typename result_type = cutlass::Array<half, 4>, typename source_type = cutlass::Array<uint8_t, 4>>
__device__ static result_type convert(source_type const& source)
{
    result_type result;

    uint32_t*      h   = reinterpret_cast<uint32_t*>(&result);
    uint32_t const i8s = reinterpret_cast<uint32_t const&>(source);

    static constexpr uint32_t mask_for_elt_01     = 0x5250;
    static constexpr uint32_t mask_for_elt_23     = 0x5351;
    static constexpr uint32_t start_byte_for_fp16 = 0x64646464;
    asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h[0]) : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
    asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h[1]) : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));

    static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[1]) : "r"(h[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
    return result;
}

__global__ void dequantize_kernel(half* dequantized_weight, const uint8_t* weight, const half* scale, int m, int n){
    half* out = &dequantized_weight[blockIdx.x * n];
    const uint8_t* in = &weight[blockIdx.x * n];

    for(int i = threadIdx.x; i < n; i += 4 * blockDim.x){
        cutlass::Array<half, 4> output;

        output = convert(reinterpret_cast<uint32_t*>(in[i])[0]);
        uint64_t* t = reinterpret_cast<uint64_t*>(&output);
        reinterpret_cast<uint64_t*>(&out[i])[0] = t[0];
    }
}

void invoke_dequantize(half* dequantized_weight,
                       const uint8_t*     weight,
                       const half*        scale,
                       int                    m,
                       int                    n)
{
    dim3 block(256);
    dim3 grid(m);
    dequantize_kernel<<<block, grid>>>(dequantized_weight, weight, scale, m, n);
}

// input b, n, c
void dequantize_weight_cuda(torch::Tensor _dequantized_weight,
                            torch::Tensor             _weight,
                            torch::Tensor              _scale)
{
    int m = _dequantized_weight.size(0);
    int n = _dequantized_weight.size(1);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_weight));
    
    auto dequantized_weight = reinterpret_cast<half*>(_dequantized_weight.data_ptr<at::Half>());
    auto weight = reinterpret_cast<uint8_t*>(_weight.data_ptr<int8_t>());
    auto scale = reinterpret_cast<half*>(_scale.data_ptr<at::Half>());

    invoke_dequantize(dequantized_weight, weight, scale, m, n);
}
