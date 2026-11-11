// Produce teacher-forced reference logits for examples/qualify_reference.rs.
// Build against a pinned llama.cpp; see fixtures/text-coverage/README.md and
// fixtures/text-coverage/cpu-reference/README.md (`--legacy`) for the command.
// Optional final SHARDS.txt argument: ordered `sha256 bytes basename` lines.
// Split models require all canonical five-digit shards, including an empty
// tensor entry shard. Hashes remain caller-verified, as for MODEL_SHA256;
// this producer checks names, sizes and split metadata and echoes the full
// manifest. qualify_reference hashes every actual shard before engine load.
// Optional FORCED_TOKENS.txt follows SHARDS.txt (use '-' for an unsplit model).
// It must contain ROWS IDs. These choose the next-row inputs independently of
// recorded per-row argmax IDs; the JSON adds teacher_forcing_tokens.
// Context is bounded at 16384 tokens. Short prompts keep the original 4096
// context/batch; longer prompts use 4096-token prefill calls and a context
// large enough for the prompt plus all requested rows.
#include "llama.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef LLAMA_REFERENCE_REVISION
#define LLAMA_REFERENCE_REVISION "unknown"
#endif

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static int32_t parse_integer(const std::string & text) {
    int32_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    require(result.ec == std::errc{} && result.ptr == text.data() + text.size(), "invalid integer");
    return value;
}

static bool valid_hash(const std::string & hash) {
    return hash.size() == 64 && hash.find_first_not_of("0123456789abcdef") == std::string::npos;
}

static constexpr size_t max_reference_context = 16384;
static constexpr size_t prefill_batch = 4096;

static uint32_t context_size(size_t prompt, size_t rows) {
    require(prompt > 0 && rows > 0 && rows <= 256 && prompt <= max_reference_context - rows,
            "invalid prompt/rows or context exceeds 16384 tokens");
    return uint32_t(std::max(prefill_batch, prompt + rows));
}

static std::vector<llama_token> read_forced_tokens(const char * path, size_t rows) {
    std::ifstream input(path);
    require(input.good(), "cannot open teacher-forcing tokens");
    std::vector<llama_token> tokens;
    std::string word;
    while (input >> word) {
        const auto token = parse_integer(word);
        require(token >= 0, "negative teacher-forcing token ID");
        require(tokens.size() < rows, "too many teacher-forcing tokens");
        tokens.push_back(token);
    }
    require(input.eof() && tokens.size() == rows, "teacher-forcing token count differs from ROWS");
    return tokens;
}

static void validate_token_ids(const std::vector<llama_token> & tokens, int32_t vocab, const char * message) {
    require(vocab > 0, "invalid vocabulary size");
    for (auto token : tokens) require(token >= 0 && token < vocab, message);
}

struct model_file {
    std::string name;
    uint64_t size_bytes;
    std::string sha256;
};

static std::vector<model_file> read_manifest(const char * path) {
    std::ifstream input(path);
    require(input.good(), "cannot open shard manifest");
    std::vector<model_file> files;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream row(line);
        model_file file;
        std::string size, extra;
        require(bool(row >> file.sha256 >> size >> file.name) && !(row >> extra), "invalid shard manifest line");
        const auto parsed = std::from_chars(size.data(), size.data() + size.size(), file.size_bytes);
        require(parsed.ec == std::errc{} && parsed.ptr == size.data() + size.size() && file.size_bytes > 0,
                "invalid shard byte count");
        require(valid_hash(file.sha256), "invalid shard SHA256");
        require(!file.name.empty() && file.name != "." && file.name != ".." &&
                file.name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-") == std::string::npos,
                "shard name must be a plain ASCII basename");
        files.push_back(file);
        require(files.size() <= 128, "too many shards");
    }
    require(input.eof() && !files.empty(), "empty or unreadable shard manifest");
    return files;
}

static uint64_t split_integer(const gguf_context * header, const char * name) {
    const int64_t key = gguf_find_key(header, name);
    require(key >= 0, "missing split metadata");
    switch (gguf_get_kv_type(header, key)) {
        case GGUF_TYPE_UINT8: return gguf_get_val_u8(header, key);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(header, key);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(header, key);
        case GGUF_TYPE_UINT64: return gguf_get_val_u64(header, key);
        case GGUF_TYPE_INT8: { auto v = gguf_get_val_i8(header, key); require(v >= 0, "negative split value"); return v; }
        case GGUF_TYPE_INT16: { auto v = gguf_get_val_i16(header, key); require(v >= 0, "negative split value"); return v; }
        case GGUF_TYPE_INT32: { auto v = gguf_get_val_i32(header, key); require(v >= 0, "negative split value"); return v; }
        case GGUF_TYPE_INT64: { auto v = gguf_get_val_i64(header, key); require(v >= 0, "negative split value"); return v; }
        default: throw std::runtime_error("split value must be an integer");
    }
}

static bool has_split(const gguf_context * header) {
    bool found = false;
    for (int64_t i = 0; i < gguf_get_n_kv(header); ++i) {
        const std::string key(gguf_get_key(header, i));
        if (key.rfind("split.", 0) == 0) {
            require(key == "split.no" || key == "split.count" || key == "split.tensors.count", "unknown split metadata");
            found = true;
        }
    }
    return found;
}

static std::string shard_suffix(size_t index, size_t count) {
    std::ostringstream text;
    text << '-' << std::setfill('0') << std::setw(5) << index << "-of-" << std::setw(5) << count << ".gguf";
    return text.str();
}

static void validate_files(const std::filesystem::path & model, const std::string & hash,
                           const std::vector<model_file> & files) {
    auto entry = std::unique_ptr<gguf_context, decltype(&gguf_free)>(
        gguf_init_from_file(model.string().c_str(), {true, nullptr}), gguf_free);
    require(bool(entry), "cannot read GGUF header");
    const bool split = has_split(entry.get());
    const std::string name = model.filename().string();
    // Match the Rust engine's filename-based discovery before accepting an
    // entry-only fixture: it must not load unlisted, unhashed sibling files.
    const auto marker = name.find("-00001-of-");
    if (marker != std::string::npos) {
        const auto start = marker + std::string("-00001-of-").size();
        const auto dot = name.find('.', start);
        uint64_t count = 0;
        if (dot != std::string::npos) {
            const auto number_start = start < dot && name[start] == '+' ? start + 1 : start;
            const auto parsed = std::from_chars(name.data() + number_start, name.data() + dot, count);
            const bool discovered = parsed.ec == std::errc{} && parsed.ptr == name.data() + dot && count >= 2;
            require(!discovered || split, "split-style filename requires split metadata and a manifest");
        }
    }
    if (files.empty()) {
        require(!split, "split model requires a complete SHARDS.txt manifest");
        return;
    }
    require(files[0].name == name && files[0].sha256 == hash, "manifest entry differs from model/MODEL_SHA256");
    require(split == (files.size() > 1), "manifest count disagrees with split metadata");
    const std::string suffix = shard_suffix(1, files.size());
    std::string prefix;
    if (split) {
        require(name.size() > suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0,
                "split entry requires canonical PREFIX-00001-of-NNNNN.gguf name");
        prefix = name.substr(0, name.size() - suffix.size());
        require(prefix.find("-00001-of-") == std::string::npos, "ambiguous split prefix");
    }
    const auto parent = model.has_parent_path() ? model.parent_path() : std::filesystem::path(".");
    const auto canonical_parent = std::filesystem::canonical(parent);
    uint64_t actual_tensors = 0, declared_tensors = 0;
    for (size_t i = 0; i < files.size(); ++i) {
        const auto & file = files[i];
        if (split) require(file.name == prefix + shard_suffix(i + 1, files.size()), "missing, reordered or noncanonical shard");
        const auto path = parent / file.name;
        require(std::filesystem::is_regular_file(path) && std::filesystem::file_size(path) == file.size_bytes,
                "shard file size differs");
        require(std::filesystem::canonical(path).parent_path() == canonical_parent, "shard resolves outside model directory");
        auto header = std::unique_ptr<gguf_context, decltype(&gguf_free)>(
            gguf_init_from_file(path.string().c_str(), {true, nullptr}), gguf_free);
        require(bool(header), "cannot read shard header");
        require(has_split(header.get()) == split, "shard split metadata differs");
        if (!split) continue;
        require(split_integer(header.get(), "split.no") == i && split_integer(header.get(), "split.count") == files.size(),
                "shard split index/count differs");
        const uint64_t count = split_integer(header.get(), "split.tensors.count");
        require(count > 0 && (i == 0 || count == declared_tensors), "shard split tensor total differs");
        declared_tensors = count;
        const int64_t local = gguf_get_n_tensors(header.get());
        require(local >= 0 && uint64_t(local) <= std::numeric_limits<uint64_t>::max() - actual_tensors, "tensor count overflow");
        actual_tensors += uint64_t(local);
    }
    require(!split || actual_tensors == declared_tensors, "actual split tensor total differs");
}

#ifdef LLAMA_CPU_REFERENCE_JSON
#include "text-coverage/cpu-reference/runtime.hpp"
#endif

int main(int argc, char ** argv) try {
#ifdef LLAMA_CPU_REFERENCE_JSON
    if (argc == 2 && std::string(argv[1]) == "--check-cpu-runtime") {
        auto runtime = dinf_reference::load_cpu();
        runtime["models_loaded"] = 0;
        runtime["backend_initialized"] = true;
        std::cout << runtime.dump(2) << '\n';
        llama_backend_free();
        return 0;
    }
#endif

    require(argc >= 7 && argc <= 9, "usage: llama_reference MODEL cpu|metal TOKENS.txt ROWS OUTPUT_PREFIX MODEL_SHA256 [SHARDS.txt|- [FORCED_TOKENS.txt]]");
    const bool metal = std::string(argv[2]) == "metal";
    require(metal || std::string(argv[2]) == "cpu", "backend must be cpu or metal");
    const int rows = parse_integer(argv[4]);
    require(rows > 0 && rows <= 256, "ROWS must be in 1..256");
    const std::string hash(argv[6]);
    require(valid_hash(hash),
            "MODEL_SHA256 must be a lowercase SHA256 digest");
    std::ifstream input(argv[3]);
    require(input.good(), "cannot open prompt tokens");
    std::vector<llama_token> prompt;
    std::string word;
    while (input >> word) {
        const int32_t id = parse_integer(word);
        require(id >= 0, "negative prompt token ID");
        require(prompt.size() < max_reference_context - size_t(rows), "prompt exceeds fixture context");
        prompt.push_back(id);
    }
    require(input.eof() && !prompt.empty(),
            "invalid, empty, or too long prompt");
    const uint32_t context = context_size(prompt.size(), size_t(rows));
    const auto files = argc >= 8 && std::string(argv[7]) != "-" ? read_manifest(argv[7]) : std::vector<model_file>{};
    const auto forced = argc == 9 ? read_forced_tokens(argv[8], size_t(rows)) : std::vector<llama_token>{};
    validate_files(argv[1], hash, files);
#ifdef LLAMA_CPU_REFERENCE_JSON
    const auto cpu_runtime_before = metal ? dinf_reference::json(nullptr) : dinf_reference::load_cpu();
    if (metal) {
        ggml_backend_load_all();
        llama_backend_init();
    }
#else
    ggml_backend_load_all();
    llama_backend_init();
#endif
    ggml_backend_dev_t devices[] = {nullptr, nullptr};
    if (metal) {
        const auto reg = ggml_backend_reg_by_name("MTL");
        require(reg != nullptr && ggml_backend_reg_dev_count(reg) > 0, "Metal backend unavailable");
        devices[0] = ggml_backend_reg_dev_get(reg, 0);
    }
    auto mp = llama_model_default_params();
    mp.devices = devices;
    mp.n_gpu_layers = metal ? -1 : 0;
    auto model = std::unique_ptr<llama_model, decltype(&llama_model_free)>(
        llama_model_load_from_file(argv[1], mp), llama_model_free);
    require(bool(model), "model load failed");
    const int32_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    validate_token_ids(prompt, vocab, "prompt token outside vocabulary");
    validate_token_ids(forced, vocab, "teacher-forcing token outside vocabulary");
    auto cp = llama_context_default_params();
    cp.n_ctx = context;
    cp.n_batch = prefill_batch;
    cp.n_ubatch = prefill_batch;
    cp.n_threads = 8;
    cp.n_threads_batch = 8;
    cp.type_k = GGML_TYPE_F16;
    cp.type_v = GGML_TYPE_F16;
    cp.offload_kqv = metal;
    cp.op_offload = metal;
    auto ctx = std::unique_ptr<llama_context, decltype(&llama_free)>(
        llama_init_from_model(model.get(), cp), llama_free);
    require(bool(ctx), "context creation failed");
    const std::filesystem::path prefix(argv[5]);
    const auto filename = prefix.filename().string() + ".f32";
    require(filename.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-") == std::string::npos,
            "output prefix basename must use letters, digits, dots, underscores or hyphens");
    require(!std::filesystem::exists(prefix.string() + ".f32") &&
            !std::filesystem::exists(prefix.string() + ".json"), "fresh output prefix required");
    std::ofstream output(prefix.string() + ".f32", std::ios::binary);
    require(output.good(), "cannot open logits output");
    for (size_t start = 0; start < prompt.size(); start += prefill_batch) {
        const auto count = static_cast<int32_t>(std::min(prefill_batch, prompt.size() - start));
        require(llama_decode(ctx.get(), llama_batch_get_one(prompt.data() + start, count)) == 0,
                "prompt decode failed");
    }
    std::vector<llama_token> continuation;
    for (int row = 0; row < rows; ++row) {
        const float * logits = llama_get_logits_ith(ctx.get(), -1);
        require(logits != nullptr, "missing logits");
        for (int32_t i = 0; i < vocab; ++i) require(std::isfinite(logits[i]), "non-finite reference logits");
        // This fixture targets little-endian hosts, as does its F32 file format.
        const uint32_t one = 1;
        require(*reinterpret_cast<const uint8_t *>(&one) == 1 && sizeof(float) == 4, "requires little-endian F32");
        output.write(reinterpret_cast<const char *>(logits), size_t(vocab) * sizeof(float));
        auto token = static_cast<llama_token>(std::max_element(logits, logits + vocab) - logits);
        continuation.push_back(token);
        if (row + 1 < rows) {
            if (!forced.empty()) token = forced[size_t(row)];
            require(llama_decode(ctx.get(), llama_batch_get_one(&token, 1)) == 0, "token decode failed");
        }
    }
    output.close();
    require(output.good(), "logits output failed");
    std::ofstream meta(prefix.string() + ".json");
    meta << "{\"source\":\"llama.cpp " LLAMA_REFERENCE_REVISION "\",\"backend\":\"" << argv[2]
         << "\",\"model_sha256\":\"" << hash << '"';
    if (!files.empty()) {
        meta << ",\"model_files\":[";
        for (size_t i = 0; i < files.size(); ++i) {
            meta << (i ? "," : "") << "{\"name\":\"" << files[i].name << "\",\"size_bytes\":" << files[i].size_bytes
                 << ",\"sha256\":\"" << files[i].sha256 << "\"}";
        }
        meta << ']';
    }
    meta << ",\"vocab_size\":" << vocab
         << ",\"logits_file\":\"" << filename << "\",\"prompt_tokens\":[";
    for (size_t i = 0; i < prompt.size(); ++i) meta << (i ? "," : "") << prompt[i];
    meta << "],\"continuation_tokens\":[";
    for (size_t i = 0; i < continuation.size(); ++i) meta << (i ? "," : "") << continuation[i];
    meta << ']';
    if (!forced.empty()) {
        meta << ",\"teacher_forcing_tokens\":[";
        for (size_t i = 0; i < forced.size(); ++i) meta << (i ? "," : "") << forced[i];
        meta << ']';
    }
#ifdef LLAMA_CPU_REFERENCE_JSON
    if (!metal) {
        const auto runtime_after = dinf_reference::runtime();
        require(cpu_runtime_before == runtime_after, "CPU runtime changed");
        meta << ",\"cpu_reference_runtime\":" << runtime_after.dump();
    }
#endif
    meta << "}\n";
    meta.close();
    require(meta.good(), "metadata output failed");
    std::cout << prefix.string() + ".json" << '\n';
    return 0;
} catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 1;
}
