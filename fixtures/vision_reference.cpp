// Projector-only reference using llama.cpp b8920 / commit
// 15fa3c493bfcd040b5f4dcb29e1c998a0846de16. Compile against that exact
// revision's tools/mtmd internal headers AND matching libmtmd binary.
// No text weights load: b8920's public mtmd API rejects vocab_only text
// models (their embedding width is zero), so use the exported clip API.
//
// Build (LLAMA_SRC and LLAMA_PREFIX must refer to the same pinned revision):
// c++ -std=c++17 -O2 -I"$LLAMA_SRC/tools/mtmd" -I"$LLAMA_SRC/ggml/include" \
//   -I"$LLAMA_PREFIX/include" vision_reference.cpp -L"$LLAMA_PREFIX/lib" \
//   -Wl,-rpath,"$LLAMA_PREFIX/lib" -lmtmd -lllama -lggml-base -lggml -o vision_reference
//
// Metadata / input only (no backend init, model weights or inference):
//   vision_reference --inspect TEXT.gguf MMPROJ.gguf
//   vision_reference --fixture OUTPUT_PREFIX [64|256|768]
// Encode (caller must verify both supplied full SHA256 values first):
//   vision_reference TEXT.gguf MMPROJ.gguf TEXT_SHA256 MMPROJ_SHA256 cpu|metal OUTPUT_PREFIX [64|256|768]
// Exports the exact 64x64 RGB image used by qualify_vision, official
// preprocessed HWC float pixels, and token-major projected F32 rows per tile.
// The JSON records layout and resize policy. Equal source RGB does NOT imply
// equal preprocessed pixels, token count, tile order, or numerical parity.
// The optional 256/768 image scales the 64 pattern by 4/12 without blending.
// At 768, a patch-size-16 projector with a 48x48 learned position table needs
// no position interpolation. Check pixel equality and pinned table geometry;
// choosing an image size alone does not establish equivalence.
// This is a component reference; it does not qualify final image descriptions.
// A separately patched reference build must define VISION_ORACLE_VARIANT as
// "hf-contract-v1" and VISION_ORACLE_PATCH_SHA256 as the full combined patch
// SHA256. The runner must also pin the actual modified source and libraries;
// these compile-time labels alone do not prove which library was loaded.
#include "clip.h"
#include "clip-impl.h"
#include "mtmd-image.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef VISION_ORACLE_VARIANT
#define VISION_ORACLE_VARIANT "upstream"
#endif
#ifndef VISION_ORACLE_PATCH_SHA256
#define VISION_ORACLE_PATCH_SHA256 ""
#endif

static constexpr const char * revision = "15fa3c493bfcd040b5f4dcb29e1c998a0846de16";

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static std::string quote(const std::string & value) {
    std::string out = "\"";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out += '\\';
        require(c >= 32, "control character in JSON string");
        out += char(c);
    }
    return out + '"';
}

static bool valid_hash(const std::string & hash) {
    return hash.size() == 64 && hash.find_first_not_of("0123456789abcdef") == std::string::npos;
}

using header_ptr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
static header_ptr read_header(const char * path) {
    header_ptr header(gguf_init_from_file(path, {true, nullptr}), gguf_free);
    require(bool(header), "cannot read GGUF header");
    return header;
}

static std::string string_key(const gguf_context * header, const std::string & key) {
    const auto index = gguf_find_key(header, key.c_str());
    require(index >= 0 && gguf_get_kv_type(header, index) == GGUF_TYPE_STRING, "missing string metadata");
    return gguf_get_val_str(header, index);
}

static uint64_t positive_key(const gguf_context * header, const std::string & key) {
    const auto index = gguf_find_key(header, key.c_str());
    require(index >= 0, "missing integer metadata");
    uint64_t value = 0;
    switch (gguf_get_kv_type(header, index)) {
        case GGUF_TYPE_UINT32: value = gguf_get_val_u32(header, index); break;
        case GGUF_TYPE_UINT64: value = gguf_get_val_u64(header, index); break;
        case GGUF_TYPE_INT32: {
            const auto signed_value = gguf_get_val_i32(header, index);
            require(signed_value > 0, "nonpositive integer metadata");
            value = uint64_t(signed_value);
            break;
        }
        default: throw std::runtime_error("unsupported integer metadata type");
    }
    require(value > 0 && value <= uint64_t(std::numeric_limits<int32_t>::max()), "integer metadata outside bounds");
    return value;
}

struct metadata {
    std::string architecture;
    std::string projector;
    int32_t width;
};

static metadata inspect(const char * text, const char * projector) {
    const auto text_header = read_header(text);
    const auto vision_header = read_header(projector);
    metadata result;
    result.architecture = string_key(text_header.get(), "general.architecture");
    result.projector = string_key(vision_header.get(),
        gguf_find_key(vision_header.get(), "clip.projector_type") >= 0
            ? "clip.projector_type" : "clip.vision.projector_type");
    result.width = int32_t(positive_key(text_header.get(), result.architecture + ".embedding_length"));
    require((result.architecture == "qwen35" && result.projector == "qwen3vl_merger") ||
            (result.architecture == "gemma4" && result.projector == "gemma4v") ||
            (result.architecture == "llama4" && result.projector == "llama4"), "unsupported text/projector family pair");
    return result;
}

static int fixture_size(const char * value) {
    const std::string text(value);
    require(text == "64" || text == "256" || text == "768", "fixture side must be 64, 256 or 768");
    return text == "64" ? 64 : text == "256" ? 256 : 768;
}

static clip_image_u8 fixture(int side = 64) {
    clip_image_u8 image{side, side, {}};
    image.buf.reserve(size_t(side) * size_t(side) * 3);
    const int scale = side / 64;
    for (int row = 0; row < side; ++row) {
        for (int col = 0; col < side; ++col) {
            const int x = col / scale, y = row / scale;
            std::array<uint8_t, 3> color;
            if (x < 2 || y < 2 || x >= 62 || y >= 62) color = {0, 0, 0};
            else if ((x >= 30 && x < 34) || (y >= 30 && y < 34)) color = {255, 255, 255};
            else if (y < 32) color = x < 32 ? std::array<uint8_t, 3>{255, 0, 0} : std::array<uint8_t, 3>{0, 255, 0};
            else color = x < 32 ? std::array<uint8_t, 3>{0, 0, 255} : std::array<uint8_t, 3>{255, 255, 0};
            image.buf.insert(image.buf.end(), color.begin(), color.end());
        }
    }
    return image;
}

static void write_bytes(const std::string & path, const void * data, size_t bytes) {
    std::ofstream output(path, std::ios::binary);
    require(output.good(), "cannot open output file");
    output.write(static_cast<const char *>(data), std::streamsize(bytes));
    output.close();
    require(output.good(), "cannot write output file");
}

static void write_floats(const std::string & path, const std::vector<float> & values) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    const uint32_t endian = 1;
    require(*reinterpret_cast<const uint8_t *>(&endian) == 1, "reference requires little-endian host");
    require(!values.empty() && std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); }),
            "empty/nonfinite float output");
    write_bytes(path, values.data(), values.size() * sizeof(float));
}

static void write_fixture(const std::string & prefix, const clip_image_u8 & image) {
    const auto parent = std::filesystem::path(prefix).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    write_bytes(prefix + ".rgb", image.buf.data(), image.buf.size());
    const std::string header = "P6\n" + std::to_string(image.nx) + " " + std::to_string(image.ny) + "\n255\n";
    std::vector<uint8_t> ppm(header.begin(), header.end());
    ppm.insert(ppm.end(), image.buf.begin(), image.buf.end());
    write_bytes(prefix + ".ppm", ppm.data(), ppm.size());
}

static int run(int argc, char ** argv) {
    const std::string variant = VISION_ORACLE_VARIANT, patch = VISION_ORACLE_PATCH_SHA256;
    require((variant == "upstream" && patch.empty()) ||
            (variant == "hf-contract-v1" && valid_hash(patch)), "invalid oracle variant/patch identity");
    if ((argc == 3 || argc == 4) && std::string(argv[1]) == "--fixture") {
        write_fixture(argv[2], fixture(argc == 4 ? fixture_size(argv[3]) : 64));
        return 0;
    }
    if (argc == 4 && std::string(argv[1]) == "--inspect") {
        const auto info = inspect(argv[2], argv[3]);
        std::cout << "{\"text_architecture\":" << quote(info.architecture)
                  << ",\"projector_type\":" << quote(info.projector)
                  << ",\"embedding_width\":" << info.width << ",\"text_weights_loaded\":false}\n";
        return 0;
    }
    require(argc == 7 || argc == 8, "usage: TEXT MMPROJ TEXT_SHA256 MMPROJ_SHA256 cpu|metal OUTPUT_PREFIX [64|256|768]; or --fixture PREFIX [64|256|768]; or --inspect TEXT MMPROJ");
    require(valid_hash(argv[3]) && valid_hash(argv[4]), "invalid caller-verified SHA256");
    const std::string backend = argv[5], prefix = argv[6];
    require(backend == "cpu" || backend == "metal", "backend must be cpu or metal");
    const auto info = inspect(argv[1], argv[2]);
    const auto image = fixture(argc == 8 ? fixture_size(argv[7]) : 64);
    write_fixture(prefix, image);
    ggml_backend_load_all();
    clip_context_params params{backend == "metal", CLIP_FLASH_ATTN_TYPE_DISABLED, -1, -1, false, nullptr, nullptr};
    const auto loaded = clip_init(argv[2], params);
    const std::unique_ptr<clip_ctx, decltype(&clip_free)> vision(loaded.ctx_v, clip_free);
    const std::unique_ptr<clip_ctx, decltype(&clip_free)> audio(loaded.ctx_a, clip_free);
    require(bool(vision), "projector has no vision encoder");
    require(clip_n_mmproj_embd(vision.get()) == info.width, "projector/text embedding width differs");
    std::unique_ptr<mtmd_image_preprocessor> processor;
    if (info.projector == "llama4") processor = std::make_unique<mtmd_image_preprocessor_llava_uhd>(vision.get());
    else processor = std::make_unique<mtmd_image_preprocessor_dyn_size>(vision.get());
    clip_image_f32_batch images;
    require(processor->preprocess(image, images) && !images.entries.empty(), "image preprocessing failed");
    const auto & hp = *clip_get_hparams(vision.get());
    std::ostringstream json;
    json << std::setprecision(9)
         << "{\n\"schema_version\":1,\"producer\":\"llama.cpp clip projector\",\"revision\":" << quote(revision)
         << ",\"oracle_variant\":" << quote(variant)
         << ",\"oracle_patch_sha256\":" << (patch.empty() ? "null" : quote(patch))
         << ",\"backend_requested\":" << quote(backend) << ",\"text_weights_loaded\":false,\"hash_verification\":\"caller_required_not_performed_by_producer\""
         << ",\"unused_audio_encoder_loaded\":" << (audio ? "true" : "false")
         << ",\"text_model_sha256\":" << quote(argv[3]) << ",\"projector_sha256\":" << quote(argv[4])
         << ",\"text_architecture\":" << quote(info.architecture) << ",\"projector_type\":" << quote(info.projector)
         << ",\"embedding_width\":" << info.width << ",\"input_rgb_file\":" << quote(prefix + ".rgb")
         << ",\"input_shape_hwc\":[" << image.ny << ',' << image.nx << ",3],\"pixel_format\":\"f32le-HWC-RGB\",\"embedding_format\":\"f32le-token-major\""
         << ",\"preprocessing_equivalence_to_engine\":\"not_assumed\",\"quality_pass\":null"
         << ",\"resize_algorithm_enum\":" << int(hp.image_resize_algo)
         << ",\"overview_resize_algorithm_enum\":" << int(hp.image_resize_algo_ov)
         << ",\"refined_resize_algorithm_enum\":" << int(hp.image_resize_algo_rf)
         << ",\"min_pixels\":" << hp.image_min_pixels << ",\"max_pixels\":" << hp.image_max_pixels
         << ",\"patch_size\":" << hp.patch_size << ",\"merge_size\":" << hp.n_merge
         << ",\"image_mean\":[" << hp.image_mean[0] << ',' << hp.image_mean[1] << ',' << hp.image_mean[2] << ']'
         << ",\"image_std\":[" << hp.image_std[0] << ',' << hp.image_std[1] << ',' << hp.image_std[2] << ']'
         << ",\"tile_grid\":[" << images.grid_x << ',' << images.grid_y << "],\"images\":[";
    for (size_t i = 0; i < images.entries.size(); ++i) {
        auto & pixels = *images.entries[i];
        const int rows = clip_n_output_tokens(vision.get(), &pixels);
        require(pixels.nx > 0 && pixels.ny > 0 && rows > 0, "invalid projected image geometry");
        require(pixels.buf.size() == size_t(pixels.nx) * size_t(pixels.ny) * 3, "invalid HWC pixel buffer");
        const std::string pixels_file = prefix + ".image" + std::to_string(i) + ".pixels.f32";
        const std::string embedding_file = prefix + ".image" + std::to_string(i) + ".embeddings.f32";
        write_floats(pixels_file, pixels.buf);
        const size_t count = size_t(rows) * size_t(info.width);
        require(count <= size_t(1024) * 1024 * 1024 / sizeof(float), "reference output exceeds 1GiB bound");
        std::vector<float> output(count);
        require(clip_image_encode(vision.get(), 4, &pixels, output.data()), "projector encode failed");
        write_floats(embedding_file, output);
        if (i > 0) json << ',';
        json << "{\"index\":" << i << ",\"shape_hwc\":[" << pixels.ny << ',' << pixels.nx << ",3]"
             << ",\"pixels_file\":" << quote(pixels_file) << ",\"embeddings_file\":" << quote(embedding_file)
             << ",\"embedding_shape\":[" << rows << ',' << info.width << ']'
             << ",\"output_grid_xy\":[" << clip_n_output_tokens_x(vision.get(), &pixels) << ','
             << clip_n_output_tokens_y(vision.get(), &pixels) << "]}";
    }
    json << "]}\n";
    const std::string result = json.str();
    write_bytes(prefix + ".json", result.data(), result.size());
    std::cout << result;
    return 0;
}

int main(int argc, char ** argv) {
    try { return run(argc, argv); }
    catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
