#include <metal_stdlib>

// Correct for rounding in 1 + exp(x), including a sum that rounds to one.
// The volatile sum prevents fast-math simplification of sum - 1 back to exp(x).
inline float dinf_softplus(float x) {
    if (x > 20.0f) { return x; }
    const float exponent = metal::precise::exp(x);
    volatile float sum = 1.0f + exponent;
    if (sum == 1.0f) { return exponent; }
    return metal::precise::log(sum) * (exponent / (sum - 1.0f));
}
