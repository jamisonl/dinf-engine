#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

// Native BF16 operands and F32 accumulation. No FP16 conversion or quantization.
kernel void diffusion_dense_bf16(
    device bfloat *input [[buffer(0)]],
    device bfloat *weight [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &inner [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]]) {
    constexpr int bm = 128, bn = 64;
    auto a = tensor(input, dextents<int32_t, 2>(int(inner), int(rows)),
        array<int, 2>({1, int(inner)}));
    auto b = tensor(weight, dextents<int32_t, 2>(int(inner), int(columns)),
        array<int, 2>({1, int(inner)}));
    mpp::tensor_ops::matmul2d<
        // Last flag disables relaxed precision, even on newer GPU hardware.
        mpp::tensor_ops::matmul2d_descriptor(bm, bn, int(dynamic_extent), false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> op;
    auto accum = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
    auto ax = a.slice(0, int(group.x) * bm);
    auto bx = b.slice(0, int(group.y) * bn);
    op.run(ax, bx, accum);
    for (uint i = 0; i < accum.get_capacity(); ++i) {
        if (accum.is_valid_element(i)) {
            const auto index = accum.get_multidimensional_index(i);
            const uint column = group.y * bn + index[0];
            const uint row = group.x * bm + index[1];
            if (row < rows && column < columns) output[row * columns + column] = bfloat(accum[i]);
        }
    }
}
