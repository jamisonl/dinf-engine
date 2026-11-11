// Synthetic CPU-only proof. No models, weights, saved arrays, or GPU backends.
// Graph construction follows the pinned ggml C API (see LICENSE-MIT).
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void need(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static uint32_t bits(float x) {
    uint32_t result;
    std::memcpy(&result, &x, sizeof(result));
    return result;
}

static std::vector<float> graph(int width, int columns, bool precise,
        const std::vector<ggml_fp16_t> & left, const std::vector<float> & right) {
    auto * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    need(ctx, "context");
    auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, width, columns, 8);
    auto * y = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, 8, 32);
    auto * z = ggml_mul_mat(ctx, x, y);
    if (precise) need(ggml_prec_set_acc(z, GGML_PREC_F32), "precision request");
    ggml_set_input(x); ggml_set_input(y); ggml_set_output(z);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, z);
    auto backend = ggml_backend_cpu_init();
    need(backend, "CPU backend");
    ggml_backend_cpu_set_n_threads(backend, 2);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    need(buffer, "CPU buffer");
    ggml_backend_tensor_set(x, left.data(), 0, left.size() * sizeof(left[0]));
    ggml_backend_tensor_set(y, right.data(), 0, right.size() * sizeof(right[0]));
    need(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute");
    std::vector<float> result(size_t(columns) * 8 * 32);
    ggml_backend_tensor_get(z, result.data(), 0, result.size() * sizeof(result[0]));
    ggml_backend_buffer_free(buffer); ggml_backend_free(backend); ggml_free(ctx);
    return result;
}

int main(int argc, char ** argv) {
    try {
        need(argc == 3 || argc == 4, "control PREFIX | repaired PREFIX CONTROL_DEFAULTS");
        const bool repaired = std::string(argv[1]) == "repaired";
        need(repaired ? argc == 4 : std::string(argv[1]) == "control" && argc == 3, "mode");
        const std::string prefix = argv[2];
        need(!std::filesystem::exists(prefix + ".json") &&
             !std::filesystem::exists(prefix + ".defaults.f32"), "fresh outputs required");
        std::ifstream control;
        if (repaired) { control.open(argv[3], std::ios::binary); need(bool(control), "control"); }
        std::ofstream output(prefix + ".json");
        std::ofstream defaults(prefix + ".defaults.f32", std::ios::binary);
        need(bool(output) && bool(defaults), "outputs");
        output << std::setprecision(17) << "{\"cases\":[";
        size_t default_differences = 0;
        bool all_precise = true;
        for (int mode = 0; mode < 6; ++mode) {
            const int width = mode >= 4 ? 32 : mode == 3 ? 512 : 128;
            const int columns = mode >= 4 ? 128 : 25;
            std::vector<ggml_fp16_t> x(size_t(width) * columns * 8);
            std::vector<float> y(size_t(width) * 8 * 32);
            for (size_t i = 0; i < x.size(); ++i) {
                const float value = mode == 0 ? .25f : mode == 1 ? 64.f :
                    mode == 4 ? 4096.f : float(int((i * 17 + 23) % 257) - 128) / 64.f;
                x[i] = ggml_fp32_to_fp16(value);
            }
            for (size_t i = 0; i < y.size(); ++i) {
                y[i] = mode >= 4 ? (i % size_t(width) < 25 ? 1.f / 25.f : 0.f) :
                    mode == 0 ? .5f : mode == 1 ? 64.f :
                    float(int((i * 29 + 11) % 251) - 125) / 64.f;
            }
            const auto legacy = graph(width, columns, false, x, y);
            const auto precise = graph(width, columns, true, x, y);
            defaults.write(reinterpret_cast<const char *>(legacy.data()),
                           std::streamsize(legacy.size() * sizeof(float)));
            if (repaired) {
                std::vector<float> old(legacy.size());
                control.read(reinterpret_cast<char *>(old.data()), std::streamsize(old.size() * sizeof(float)));
                need(size_t(control.gcount()) == old.size() * sizeof(float), "control size");
                for (size_t i = 0; i < old.size(); ++i) default_differences += bits(old[i]) != bits(legacy[i]);
            }
            const double u = std::ldexp(1.0, -24);
            const double gamma = width * u / (1 - width * u);
            double max_error = 0, max_ratio = 0;
            size_t nonfinite = 0;
            for (int h = 0; h < 32; ++h) for (int q = 0; q < 8; ++q) for (int k = 0; k < columns; ++k) {
                double exact = 0, sum_abs = 0;
                for (int d = 0; d < width; ++d) {
                    const float a = ggml_fp16_to_fp32(x[(size_t(h / 4) * columns + k) * width + d]);
                    const float b = y[(size_t(h) * 8 + q) * width + d];
                    const double product = double(a) * double(b);
                    exact += product; sum_abs += std::abs(product);
                }
                const float actual = precise[(size_t(h) * 8 + q) * columns + k];
                if (!std::isfinite(actual)) { ++nonfinite; continue; }
                const double error = std::abs(double(actual) - exact);
                max_error = std::max(max_error, error);
                max_ratio = std::max(max_ratio, error / (gamma * sum_abs + std::ldexp(1., -149)));
            }
            const bool pass = nonfinite == 0 && max_ratio <= 1;
            all_precise &= pass;
            if (mode) output << ',';
            output << "{\"mode\":" << mode << ",\"width\":" << width << ",\"columns\":" << columns
                   << ",\"outputs\":" << precise.size() << ",\"f32_nonfinite\":" << nonfinite
                   << ",\"max_abs_f64_error\":" << max_error << ",\"max_forward_bound_ratio\":" << max_ratio
                   << ",\"precision_passed\":" << (pass ? "true" : "false") << '}';
        }
        if (repaired) need(control.peek() == EOF, "extra control bytes");
        const bool passed = !repaired || (all_precise && default_differences == 0);
        output << "],\"mode\":\"" << argv[1] << "\",\"passed\":" << (passed ? "true" : "false")
               << ",\"all_requested_f32_cases_passed\":" << (all_precise ? "true" : "false")
               << ",\"default_control_bit_differences\":" << default_differences
               << ",\"cases_checked\":6,\"model_calls\":0,\"gpu_calls\":0,"
               << "\"bound\":\"gamma_n*sumabs(F16-left times original-F32-right)+2^-149; u=2^-24\"}\n";
        defaults.flush(); output.flush(); need(bool(defaults) && bool(output), "flush");
        return passed ? 0 : 2;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
