// Independent fixed-history text suite. Build with build_reference.py.
// Input text and IDs must agree with llama.cpp tokenization before any decode.
// One model load; a new F16-KV context for every case; exactly 128 rows.
#include "llama.h"
#include "ggml-backend.h"
#include "gguf.h"
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef LLAMA_REFERENCE_REVISION
#define LLAMA_REFERENCE_REVISION "unknown"
#endif
#ifndef LLAMA_MATRIX_SOURCE_SHA256
#define LLAMA_MATRIX_SOURCE_SHA256 "unknown"
#endif

#ifndef LLAMA_CPU_REFERENCE_JSON
#define LLAMA_CPU_REFERENCE_JSON "null"
#endif

using json = nlohmann::json;
namespace fs = std::filesystem;
constexpr size_t rows_per_case = 128;
constexpr uint32_t prefill_chunk = 128;
// Keep physical batches below the pinned Metal matrix-kernel thresholds:
// those paths round F32 inputs to BF16 (BF16 weights) or F16 (F32 weights).
// CPU BF16 dot products still round their inputs; this is not an all-F32 claim.
constexpr uint32_t micro_batch_tokens = 8;
static_assert(micro_batch_tokens <= prefill_chunk);
constexpr size_t max_input_bytes = 64 * 1024 * 1024;

static void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}

static void check_micro_batch(uint32_t actual) {
    require(actual == micro_batch_tokens, "actual micro batch differs from fixed eight-token reference");
}

// SHA-256, streamed over files and the exact little-endian F32 output bytes.
class sha256 {
    std::array<uint32_t, 8> state_{{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                  0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19}};
    std::array<uint8_t, 64> block_{};
    uint64_t bytes_ = 0;
    size_t used_ = 0;
    static uint32_t rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }
    void compress() {
        static constexpr uint32_t k[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64];
        for (size_t i = 0; i < 16; ++i) {
            w[i] = uint32_t(block_[4*i]) << 24 | uint32_t(block_[4*i+1]) << 16 |
                   uint32_t(block_[4*i+2]) << 8 | uint32_t(block_[4*i+3]);
        }
        for (size_t i = 16; i < 64; ++i) {
            const auto a = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >> 3);
            const auto b = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + a + w[i-7] + b;
        }
        auto a=state_[0],b=state_[1],c=state_[2],d=state_[3];
        auto e=state_[4],f=state_[5],g=state_[6],h=state_[7];
        for (size_t i=0; i<64; ++i) {
            const auto s1=rotr(e,6)^rotr(e,11)^rotr(e,25);
            const auto t1=h+s1+((e&f)^(~e&g))+k[i]+w[i];
            const auto s0=rotr(a,2)^rotr(a,13)^rotr(a,22);
            const auto t2=s0+((a&b)^(a&c)^(b&c));
            h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        state_[0]+=a;state_[1]+=b;state_[2]+=c;state_[3]+=d;
        state_[4]+=e;state_[5]+=f;state_[6]+=g;state_[7]+=h;
    }
public:
    void update(const void * source, size_t count) {
        const auto * data = static_cast<const uint8_t *>(source);
        require(count <= std::numeric_limits<uint64_t>::max() - bytes_, "SHA256 length overflow");
        bytes_ += count;
        while (count) {
            const size_t n = std::min(count, block_.size() - used_);
            std::copy(data, data+n, block_.begin()+used_);
            used_ += n; data += n; count -= n;
            if (used_ == block_.size()) { compress(); used_=0; }
        }
    }
    std::string finish() {
        require(bytes_ <= std::numeric_limits<uint64_t>::max()/8, "SHA256 bit length overflow");
        const uint64_t bits = bytes_*8;
        block_[used_++] = 0x80;
        if (used_ > 56) { std::fill(block_.begin()+used_,block_.end(),0);compress();used_=0; }
        std::fill(block_.begin()+used_,block_.begin()+56,0);
        for (size_t i=0;i<8;++i) block_[63-i]=uint8_t(bits>>(8*i));
        compress();
        std::ostringstream out;out<<std::hex<<std::setfill('0');
        for (auto value:state_) out<<std::setw(8)<<value;
        return out.str();
    }
};

static bool valid_hash(const std::string & s) {
    return s.size()==64 && s.find_first_not_of("0123456789abcdef")==std::string::npos;
}
static std::string hash_text(const std::string & s) { sha256 h;h.update(s.data(),s.size());return h.finish(); }
static std::string hash_file(const fs::path & path) {
    std::ifstream in(path,std::ios::binary);require(bool(in),"cannot hash file: "+path.string());
    sha256 h;std::array<char,65536> data{};
    while (in) { in.read(data.data(),data.size());h.update(data.data(),size_t(in.gcount())); }
    require(in.eof()&&!in.bad(),"file hash read failed: "+path.string());return h.finish();
}
static std::string read_text(const fs::path & path) {
    const auto n=fs::file_size(path);require(n<=max_input_bytes,"input exceeds64MiB");
    std::ifstream in(path,std::ios::binary);require(bool(in),"cannot open input");
    std::string text(size_t(n),'\0');in.read(text.data(),std::streamsize(n));
    require(size_t(in.gcount())==n&&!in.bad(),"short input read");return text;
}
static uint64_t integer(const json & value, uint64_t maximum, const std::string & label) {
    require(value.is_number_integer(),label+" must be an integer");
    uint64_t n;
    if (value.is_number_unsigned()) n=value.get<uint64_t>();
    else { const auto signed_n=value.get<int64_t>();require(signed_n>=0,label+" is negative");n=uint64_t(signed_n); }
    require(n<=maximum,label+" exceeds bound");return n;
}
static std::string identifier(const json & value, const std::string & label) {
    require(value.is_string(),label+" must be a string");auto s=value.get<std::string>();
    require(!s.empty()&&s.size()<=128&&s!="."&&s!=".."&&
            s.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")==std::string::npos,
            label+" must be a safe ASCII identifier");return s;
}
static std::vector<llama_token> token_ids(const json & value, int32_t vocab, const std::string & label) {
    require(value.is_array()&&value.size()<=65536,label+" must be a bounded token array");
    std::vector<llama_token> result;result.reserve(value.size());
    for (const auto & id:value) result.push_back(llama_token(integer(id,uint64_t(vocab-1),label)));
    return result;
}
struct suite_case {
    std::string id,domain,prompt_text,continuation_text;
    uint32_t context;
    std::vector<llama_token> prompt,continuation,teacher;
};
struct model_file { std::string name,sha;uint64_t bytes; };
struct prepared_suite {
    std::string id,corpus_sha,model_sha,architecture;
    int32_t vocab;uint32_t model_context;
    std::vector<model_file> files;std::vector<suite_case> cases;
};
static prepared_suite parse_suite(const json & value) {
    require(value.is_object(),"prepared suite must be an object");
    require(integer(value.at("schema_version"),1,"schema_version")==1,"unsupported schema_version");
    require(integer(value.at("rows"),rows_per_case,"rows")==rows_per_case,"requires128 rows per case");
    prepared_suite s;s.id=identifier(value.at("suite_id"),"suite_id");
    s.corpus_sha=value.at("corpus_sha256").get<std::string>();s.model_sha=value.at("model_sha256").get<std::string>();
    require(valid_hash(s.corpus_sha)&&valid_hash(s.model_sha),"invalid corpus/model SHA256");
    s.architecture=identifier(value.at("architecture"),"architecture");
    s.vocab=int32_t(integer(value.at("vocab_size"),std::numeric_limits<int32_t>::max(),"vocab_size"));require(s.vocab>0,"empty vocabulary");
    s.model_context=uint32_t(integer(value.at("model_context_length"),std::numeric_limits<int32_t>::max(),"model context"));
    const auto & files=value.at("model_files");require(files.is_array()&&!files.empty()&&files.size()<=128,"invalid model_files");
    std::set<std::string> names;
    for (const auto & f:files) {
        model_file file{identifier(f.at("name"),"model basename"),f.at("sha256").get<std::string>(),integer(f.at("size_bytes"),uint64_t(1)<<50,"model bytes")};
        require(file.bytes>0&&valid_hash(file.sha)&&names.insert(file.name).second,"invalid/duplicate model file");s.files.push_back(file);
    }
    require(s.files.front().sha==s.model_sha,"entry SHA differs from model_files");
    const auto & cases=value.at("cases");require(cases.is_array()&&!cases.empty()&&cases.size()<=128,"invalid case count");
    std::set<std::string> ids;
    for (const auto & c:cases) {
        suite_case item;item.id=identifier(c.at("id"),"case id");item.domain=identifier(c.at("domain"),"domain");require(ids.insert(item.id).second,"duplicate case id");
        require(c.at("prompt_add_special").is_boolean()&&c.at("prompt_add_special").get<bool>()&&
                c.at("continuation_add_special").is_boolean()&&!c.at("continuation_add_special").get<bool>()&&
                c.at("parse_special").is_boolean()&&c.at("parse_special").get<bool>(),"unsupported tokenization flags");
        item.context=uint32_t(integer(c.at("context_tokens"),s.model_context,"context_tokens"));
        require(item.context==512||item.context==2048||item.context==4096,"unsupported common context capacity");
        item.prompt_text=c.at("rendered_prompt_utf8").get<std::string>();item.continuation_text=c.at("continuation_utf8").get<std::string>();
        require(!item.prompt_text.empty()&&!item.continuation_text.empty()&&item.prompt_text.size()<=1024*1024&&item.continuation_text.size()<=1024*1024,"empty/oversize case text");
        item.prompt=token_ids(c.at("prompt_original_ids"),s.vocab,"prompt IDs");
        require(item.prompt==token_ids(c.at("prompt_tokens"),s.vocab,"selected prompt IDs"),"prompt was sliced or changed");
        item.continuation=token_ids(c.at("continuation_original_ids"),s.vocab,"continuation IDs");
        item.teacher=token_ids(c.at("teacher_forcing_tokens"),s.vocab,"teacher IDs");
        require(!item.prompt.empty()&&item.prompt.size()+rows_per_case<=item.context,"prompt plus128 rows exceeds common context");
        require(item.continuation.size()>=rows_per_case&&item.teacher.size()==rows_per_case&&
                std::equal(item.teacher.begin(),item.teacher.end(),item.continuation.begin()),"teacher must be first128 original continuation IDs");
        s.cases.push_back(std::move(item));
    }
    return s;
}

static uint64_t split_integer(const gguf_context * header,const char * key_name) {
    const auto key=gguf_find_key(header,key_name);require(key>=0,"missing split metadata");
    switch (gguf_get_kv_type(header,key)) {
        case GGUF_TYPE_UINT8:return gguf_get_val_u8(header,key);
        case GGUF_TYPE_UINT16:return gguf_get_val_u16(header,key);
        case GGUF_TYPE_UINT32:return gguf_get_val_u32(header,key);
        case GGUF_TYPE_UINT64:return gguf_get_val_u64(header,key);
        case GGUF_TYPE_INT8:{auto n=gguf_get_val_i8(header,key);require(n>=0,"negative split value");return uint64_t(n);}
        case GGUF_TYPE_INT16:{auto n=gguf_get_val_i16(header,key);require(n>=0,"negative split value");return uint64_t(n);}
        case GGUF_TYPE_INT32:{auto n=gguf_get_val_i32(header,key);require(n>=0,"negative split value");return uint64_t(n);}
        case GGUF_TYPE_INT64:{auto n=gguf_get_val_i64(header,key);require(n>=0,"negative split value");return uint64_t(n);}
        default:throw std::runtime_error("split value must be an integer");
    }
}
static bool split_metadata(const gguf_context * header) {
    bool found=false;
    for (int64_t i=0;i<gguf_get_n_kv(header);++i) {
        const std::string key=gguf_get_key(header,i);
        if (key.rfind("split.",0)==0) {
            require(key=="split.no"||key=="split.count"||key=="split.tensors.count","unknown split metadata");found=true;
        }
    }
    return found;
}
static std::string shard_suffix(size_t index,size_t count) {
    std::ostringstream s;s<<'-'<<std::setfill('0')<<std::setw(5)<<index<<"-of-"<<std::setw(5)<<count<<".gguf";return s.str();
}
static json model_file_stats(const fs::path & entry,const prepared_suite & suite) {
    const auto parent=entry.has_parent_path()?entry.parent_path():fs::path(".");
    json result=json::array();
    for(const auto & file:suite.files){
        const auto path=parent/file.name;
        require(fs::is_regular_file(path),"model file is not regular: "+file.name);
        result.push_back({{"path",fs::canonical(path).string()},{"size_bytes",fs::file_size(path)},
                          {"write_time_ticks",fs::last_write_time(path).time_since_epoch().count()}});
    }
    return result;
}
static json validate_model_files(const fs::path & entry,const prepared_suite & suite) {
    require(entry.filename().string()==suite.files.front().name,"entry basename differs from prepared model_files");
    const auto parent=entry.has_parent_path()?entry.parent_path():fs::path(".");
    const auto canonical_parent=fs::canonical(parent);const bool split=suite.files.size()>1;
    std::string prefix;
    if (split) {
        const auto suffix=shard_suffix(1,suite.files.size());const auto & name=suite.files.front().name;
        require(name.size()>suffix.size()&&name.compare(name.size()-suffix.size(),suffix.size(),suffix)==0,"split entry requires canonical five-digit shard suffix");
        prefix=name.substr(0,name.size()-suffix.size());require(prefix.find("-00001-of-")==std::string::npos,"ambiguous split prefix");
    } else {
        require(entry.filename().string().find("-00001-of-")==std::string::npos,"split-style entry requires a full shard manifest");
    }
    uint64_t total=0,declared=0;
    std::set<std::string> tensor_names;
    std::map<std::string,std::pair<uint64_t,uint64_t>> storage;
    for (size_t i=0;i<suite.files.size();++i) {
        const auto & file=suite.files[i];const auto path=parent/file.name;
        require(!split||file.name==prefix+shard_suffix(i+1,suite.files.size()),"missing or reordered shard");
        require(fs::is_regular_file(path)&&fs::file_size(path)==file.bytes,"model file size differs");
        require(fs::canonical(path).parent_path()==canonical_parent,"model file resolves outside model directory");
        require(hash_file(path)==file.sha,"model file SHA256 differs: "+file.name);
        auto header=std::unique_ptr<gguf_context,decltype(&gguf_free)>(gguf_init_from_file(path.string().c_str(),{true,nullptr}),gguf_free);
        require(bool(header),"cannot read GGUF header");require(split_metadata(header.get())==split,"split metadata differs from manifest");
        for(int64_t t=0;t<gguf_get_n_tensors(header.get());++t){
            require(tensor_names.insert(gguf_get_tensor_name(header.get(),t)).second,"duplicate tensor across model files");
            const auto type=ggml_type_name(gguf_get_tensor_type(header.get(),t));require(type!=nullptr,"unknown tensor storage type");
            auto & counts=storage[type];const auto bytes=gguf_get_tensor_size(header.get(),t);
            require(uint64_t(bytes)<=std::numeric_limits<uint64_t>::max()-counts.second,"tensor byte count overflow");
            ++counts.first;counts.second+=uint64_t(bytes);
        }
        if (!split) continue;
        require(split_integer(header.get(),"split.no")==i&&split_integer(header.get(),"split.count")==suite.files.size(),"split index/count differs");
        const auto count=split_integer(header.get(),"split.tensors.count");require(count>0&&(i==0||count==declared),"split tensor total differs");declared=count;
        const auto local=gguf_get_n_tensors(header.get());require(local>=0&&uint64_t(local)<=std::numeric_limits<uint64_t>::max()-total,"split tensor count overflow");total+=uint64_t(local);
    }
    require(!split||total==declared,"actual split tensor total differs");
    json types=json::array();
    for(const auto & item:storage)types.push_back({{"type",item.first},{"tensor_count",item.second.first},{"encoded_tensor_bytes",item.second.second}});
    return {{"tensor_storage",types},{"total_tensor_count",tensor_names.size()}};
}

class exclusive_file {
    std::FILE * stream_;
public:
    explicit exclusive_file(const fs::path & path):stream_(std::fopen(path.string().c_str(),"wbx")) {
        require(stream_!=nullptr,"cannot create new output: "+path.string());
    }
    exclusive_file(const exclusive_file &)=delete;
    exclusive_file & operator=(const exclusive_file &)=delete;
    ~exclusive_file(){if(stream_)std::fclose(stream_);}
    void write(const void * data,size_t bytes){require(std::fwrite(data,1,bytes,stream_)==bytes,"output write failed");}
    void flush(){require(std::fflush(stream_)==0,"output flush failed");}
    void close(){auto * stream=stream_;stream_=nullptr;require(std::fclose(stream)==0,"output close failed");}
};
static void write_json(const fs::path & path,const json & value) {
    const auto text=value.dump(2)+"\n";exclusive_file file(path);file.write(text.data(),text.size());file.close();
}
static std::vector<llama_token> tokenize(const llama_vocab * vocab,const std::string & text,bool add_special) {
    require(text.size()<=size_t(std::numeric_limits<int32_t>::max()),"text too long for tokenizer");
    const int32_t required=llama_tokenize(vocab,text.data(),int32_t(text.size()),nullptr,0,add_special,true);
    require(required!=std::numeric_limits<int32_t>::min(),"token count overflow");
    require(required<=0,"unexpected tokenizer sizing result");
    std::vector<llama_token> ids(size_t(-int64_t(required)));
    const auto count=llama_tokenize(vocab,text.data(),int32_t(text.size()),ids.data(),int32_t(ids.size()),add_special,true);
    require(count>=0&&size_t(count)==ids.size(),"tokenizer count changed");return ids;
}
static std::string model_architecture(const llama_model * model) {
    std::array<char,256> value{};const auto size=llama_model_meta_val_str(model,"general.architecture",value.data(),value.size());
    require(size>0&&size_t(size)<value.size(),"missing/oversize model architecture");return {value.data(),size_t(size)};
}
static int32_t finite_argmax(const float * values,int32_t count) {
    require(values!=nullptr&&count>0,"missing/empty logits");int32_t best=0;
    for(int32_t i=0;i<count;++i){require(std::isfinite(values[i]),"non-finite reference logit");if(values[i]>values[best])best=i;}
    return best;
}
// The CPU repair is explicit and build-pinned. Metal retains stock backend
// discovery. Vocabulary-only checks initialize no dynamic backend.
static json cpu_reference_policy() {
    const auto policy=json::parse(LLAMA_CPU_REFERENCE_JSON);
    require(policy.is_object()&&(policy.at("id")=="ggml-cpu-flash-f32-v2"||policy.at("id")=="ggml-cpu-f32-activations-v3"||policy.at("id")=="ggml-cpu-flash-f32-query-v4"),
            "CPU oracle requires build_reference.py --cpu-plugin-build");
    require(policy.at("source_revision")==LLAMA_REFERENCE_REVISION,
            "CPU repair source revision differs from the producer");
    require(valid_hash(policy.at("source_patch_sha256").get<std::string>()),"CPU patch hash missing");
    for(const char * key:{"plugin","blas_plugin"}){
        const auto & file=policy.at(key);const fs::path path=file.at("path").get<std::string>();
        require(path.is_absolute()&&fs::is_regular_file(path),"CPU runtime path is not a file");
        require(fs::canonical(path)==path,"CPU runtime path must be canonical");
        require(fs::file_size(path)==file.at("size_bytes").get<uintmax_t>()&&
                hash_file(path)==file.at("sha256").get<std::string>(),"CPU runtime hash/size differs");
    }
    return policy;
}
static json cpu_reference_runtime() {
#if defined(__APPLE__)
    const auto policy=cpu_reference_policy();
    require(ggml_backend_reg_count()==2,"expected only explicit BLAS and CPU registries");
    const fs::path cpu=policy.at("plugin").at("path").get<std::string>();
    const fs::path blas=policy.at("blas_plugin").at("path").get<std::string>();
    json images=json::array();size_t cpu_images=0,blas_images=0;
    for(uint32_t i=0;i<_dyld_image_count();++i){
        const fs::path path=_dyld_get_image_name(i);const auto name=path.filename().string();
        if(name.find("libggml-cpu")==0){++cpu_images;require(fs::canonical(path)==cpu,"unexpected CPU runtime image");}
        if(name=="libggml-blas.so"){++blas_images;require(fs::canonical(path)==blas,"unexpected BLAS runtime image");}
        require(name!="libggml-metal.so","Metal backend must not be loaded by CPU oracle");
        images.push_back(path.string());
    }
    require(cpu_images==1&&blas_images==1,"explicit CPU/BLAS runtime image count differs");
    for(size_t i=0;i<ggml_backend_dev_count();++i){
        const auto type=ggml_backend_dev_type(ggml_backend_dev_get(i));
        require(type!=GGML_BACKEND_DEVICE_TYPE_GPU&&type!=GGML_BACKEND_DEVICE_TYPE_IGPU,"GPU registered in CPU oracle");
    }
    return {{"policy",policy},{"runtime_images",images},{"cpu_plugin_image_count",cpu_images},
        {"blas_plugin_image_count",blas_images},{"registry_count",ggml_backend_reg_count()},{"gpu_backends_loaded",false}};
#else
    throw std::runtime_error("repaired CPU oracle supports only the tested macOS arm64 build");
#endif
}
struct backend_guard {
    explicit backend_guard(bool load_backends=true,const std::string & backend="metal"){
        if(load_backends){
            if(backend=="cpu"){
                const auto policy=cpu_reference_policy();
                require(ggml_backend_reg_count()==0,"unexpected preloaded backend registry");
                const auto blas_path=policy.at("blas_plugin").at("path").get<std::string>();
                const auto cpu_path=policy.at("plugin").at("path").get<std::string>();
                const auto blas=ggml_backend_load(blas_path.c_str());
                require(blas&&std::string(ggml_backend_reg_name(blas))=="BLAS","explicit BLAS load failed");
                const auto cpu=ggml_backend_load(cpu_path.c_str());
                require(cpu&&std::string(ggml_backend_reg_name(cpu))=="CPU"&&ggml_backend_reg_dev_count(cpu)==1,"explicit CPU load failed");
                require(ggml_backend_dev_type(ggml_backend_reg_dev_get(cpu,0))==GGML_BACKEND_DEVICE_TYPE_CPU,"selected plugin is not CPU");
            }else{ggml_backend_load_all();}
        }
        llama_backend_init();
        if(load_backends&&backend=="cpu")(void)cpu_reference_runtime();
    }
    ~backend_guard(){llama_backend_free();}
    backend_guard(const backend_guard &)=delete;
    backend_guard & operator=(const backend_guard &)=delete;
};

static json token_difference(const std::vector<llama_token> & expected,const std::vector<llama_token> & actual) {
    size_t index=0;
    while(index<std::min(expected.size(),actual.size())&&expected[index]==actual[index])++index;
    if(index==expected.size()&&index==actual.size())return nullptr;
    json difference={{"index",index},{"expected",nullptr},{"actual",nullptr}};
    if(index<expected.size())difference["expected"]=expected[index];
    if(index<actual.size())difference["actual"]=actual[index];
    return difference;
}

static int tokenize_probe(const fs::path & model_path,const fs::path & prepared_path,const fs::path & output) {
    require(!fs::exists(output),"token probe output already exists");
    require(valid_hash(LLAMA_MATRIX_SOURCE_SHA256),"build must pin producer source SHA256");
    const auto text=read_text(prepared_path);const auto input=json::parse(text);const auto suite=parse_suite(input);
    const auto stats=model_file_stats(model_path,suite);
    // No backend plug-ins or weights are loaded. llama.cpp reads the GGUF
    // vocabulary/metadata and skips tensor allocation with vocab_only=true.
    backend_guard backend_lifetime(false);
    ggml_backend_dev_t devices[]={nullptr};
    auto mp=llama_model_default_params();mp.devices=devices;mp.n_gpu_layers=0;mp.vocab_only=true;
    auto model=std::unique_ptr<llama_model,decltype(&llama_model_free)>(llama_model_load_from_file(model_path.string().c_str(),mp),llama_model_free);
    require(bool(model),"vocabulary-only model load failed");
    const auto * vocab=llama_model_get_vocab(model.get());
    require(llama_vocab_n_tokens(vocab)==suite.vocab,"probe vocabulary size differs");
    json cases=json::array();size_t mismatches=0;
    for(const auto & c:suite.cases){
        const auto prompt=tokenize(vocab,c.prompt_text,true);
        const auto continuation=tokenize(vocab,c.continuation_text,false);
        const auto prompt_diff=token_difference(c.prompt,prompt);
        const auto continuation_diff=token_difference(c.continuation,continuation);
        const bool same=prompt_diff.is_null()&&continuation_diff.is_null();if(!same)++mismatches;
        cases.push_back({{"id",c.id},{"matched",same},{"prompt_tokens",prompt},{"continuation_tokens",continuation},
            {"expected_prompt_tokens",c.prompt.size()},{"actual_prompt_tokens",prompt.size()},
            {"expected_continuation_tokens",c.continuation.size()},{"actual_continuation_tokens",continuation.size()},
            {"prompt_first_difference",prompt_diff},{"continuation_first_difference",continuation_diff},
            {"prompt_text_sha256",hash_text(c.prompt_text)},{"continuation_text_sha256",hash_text(c.continuation_text)}});
    }
    require(model_file_stats(model_path,suite)==stats,"model stats changed during token probe");
    require(hash_file(prepared_path)==hash_text(text),"prepared input changed during token probe");
    write_json(output,{{"schema_version",1},{"source",std::string("llama.cpp ")+LLAMA_REFERENCE_REVISION},
        {"producer_source_sha256",LLAMA_MATRIX_SOURCE_SHA256},{"prepared_sha256",hash_text(text)},
        {"model_path",fs::absolute(model_path).string()},{"model_files",input.at("model_files")},
        {"model_file_stats",stats},{"model_payload_sha256_verified",false},{"vocabulary_only",true},
        {"weights_loaded",false},{"backend_plugins_loaded",false},{"prompt_add_special",true},
        {"continuation_add_special",false},{"parse_special",true},{"case_count",suite.cases.size()},
        {"mismatching_cases",mismatches},{"all_matched",mismatches==0},{"cases",cases}});
    std::cout<<output.string()<<'\n';return 0;
}

static json run_case(llama_model * model,const prepared_suite & suite,const suite_case & c,
                     const json & input,const fs::path & out,const std::string & backend,const std::string & prepared_sha) {
    const auto * vocab=llama_model_get_vocab(model);
    require(tokenize(vocab,c.prompt_text,true)==c.prompt,c.id+": independent prompt tokenization differs");
    require(tokenize(vocab,c.continuation_text,false)==c.continuation,c.id+": independent continuation tokenization differs");
    auto cp=llama_context_default_params();cp.n_ctx=c.context;cp.n_batch=prefill_chunk;cp.n_ubatch=micro_batch_tokens;
    cp.n_threads=8;cp.n_threads_batch=8;cp.type_k=GGML_TYPE_F16;cp.type_v=GGML_TYPE_F16;
    cp.offload_kqv=backend=="metal";cp.op_offload=backend=="metal";
    auto ctx=std::unique_ptr<llama_context,decltype(&llama_free)>(llama_init_from_model(model,cp),llama_free);
    require(bool(ctx),c.id+": context allocation failed");
    check_micro_batch(llama_n_ubatch(ctx.get()));
    require(llama_n_ctx(ctx.get())>=c.context,c.id+": actual context smaller than requested");
    for(size_t start=0;start<c.prompt.size();start+=prefill_chunk){
        const auto n=int32_t(std::min(size_t(prefill_chunk),c.prompt.size()-start));
        std::vector<llama_token> tokens(c.prompt.begin()+start,c.prompt.begin()+start+n);
        require(llama_decode(ctx.get(),llama_batch_get_one(tokens.data(),n))==0,c.id+": prompt decode failed");
    }
    const std::string raw_name=c.id+".f32";const auto partial=out/(raw_name+".partial");
    exclusive_file raw(partial);sha256 raw_hash;std::vector<llama_token> argmax;argmax.reserve(rows_per_case);
    const auto row_bytes=size_t(suite.vocab)*sizeof(float);
    for(size_t row=0;row<rows_per_case;++row){
        const float * logits=llama_get_logits_ith(ctx.get(),-1);argmax.push_back(finite_argmax(logits,suite.vocab));
        raw.write(logits,row_bytes);raw_hash.update(logits,row_bytes);raw.flush();
        if(row+1<rows_per_case){auto token=c.teacher[row];require(llama_decode(ctx.get(),llama_batch_get_one(&token,1))==0,c.id+": continuation decode failed");}
    }
    raw.close();const auto raw_sha=raw_hash.finish();require(!fs::exists(out/raw_name),"refusing to replace logits");fs::rename(partial,out/raw_name);
    require(fs::file_size(out/raw_name)==rows_per_case*row_bytes,"raw logits byte count differs");
    json oracle={{"source",std::string("llama.cpp ")+LLAMA_REFERENCE_REVISION},{"backend",backend},
        {"model_sha256",suite.model_sha},{"model_files",input.at("model_files")},{"vocab_size",suite.vocab},
        {"logits_file",raw_name},{"prompt_tokens",c.prompt},{"continuation_tokens",argmax},{"teacher_forcing_tokens",c.teacher},
        {"schema_version",1},{"case_id",c.id},{"domain",c.domain},{"context_tokens",c.context},
        {"allocated_context_tokens",llama_n_ctx(ctx.get())},{"rows",rows_per_case},{"prefill_chunk_tokens",prefill_chunk},
        {"micro_batch_tokens",llama_n_ubatch(ctx.get())},
        {"kv_dtype","F16"},{"threads",8},{"tokenization_verified",true},{"new_context_for_case",true},
        {"prepared_sha256",prepared_sha},{"corpus_sha256",suite.corpus_sha},{"producer_source_sha256",LLAMA_MATRIX_SOURCE_SHA256},
        {"prompt_text_sha256",hash_text(c.prompt_text)},{"continuation_text_sha256",hash_text(c.continuation_text)},
        {"oracle_raw_sha256",raw_sha},{"history","fixed first128 natural continuation tokens; argmax does not choose next input"}};
    if(backend=="cpu")oracle["cpu_attention_policy"]=cpu_reference_policy();
    const auto json_name=c.id+".json";write_json(out/json_name,oracle);
    return {{"id",c.id},{"domain",c.domain},{"context_tokens",c.context},{"prompt_tokens",c.prompt.size()},
            {"rows",rows_per_case},{"oracle_json",json_name},{"oracle_json_sha256",hash_file(out/json_name)},
            {"oracle_raw",raw_name},{"oracle_raw_sha256",raw_sha},{"oracle_raw_bytes",rows_per_case*row_bytes}};
}

static json self_test_input() {
    std::vector<llama_token> tail(rows_per_case);for(size_t i=0;i<tail.size();++i)tail[i]=llama_token(i%32);
    json c={{"id","one"},{"domain","code"},{"context_tokens",512},{"rendered_prompt_utf8","prompt"},
        {"continuation_utf8","continuation"},{"prompt_original_ids",{1,2,3}},{"prompt_tokens",{1,2,3}},
        {"continuation_original_ids",tail},{"teacher_forcing_tokens",tail},
        {"prompt_add_special",true},{"continuation_add_special",false},{"parse_special",true}};
    return {{"schema_version",1},{"suite_id","test"},{"rows",rows_per_case},{"corpus_sha256",std::string(64,'1')},
        {"model_sha256",std::string(64,'0')},{"architecture","llama"},{"vocab_size",32},{"model_context_length",4096},
        {"model_files",json::array({{{"name","model.gguf"},{"size_bytes",1},{"sha256",std::string(64,'0')}}})},
        {"cases",json::array({c})}};
}
static int self_tests() {
    size_t checks=0;
    auto test=[&](bool condition,const char * label){require(condition,std::string("self-test failed: ")+label);++checks;};
    auto rejects=[&](auto function,const char * label){bool rejected=false;try{function();}catch(const std::exception &){rejected=true;}test(rejected,label);};
    check_micro_batch(8);test(micro_batch_tokens==8&&prefill_chunk==128,"fixed physical and submitted batches");
    rejects([&]{check_micro_batch(128);},"reject old physical batch");
    rejects([&]{check_micro_batch(1);},"reject different short physical batch");
    test(hash_text("")=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","SHA256 empty");
    test(hash_text("abc")=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","SHA256 abc");
    const std::string boundary="abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    test(hash_text(boundary)=="248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1","SHA256 padding boundary");
    const std::string million(1000000,'a');const std::string million_hash="cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0";
    test(hash_text(million)==million_hash,"SHA256 million a");
    for(size_t chunk:{size_t(1),size_t(31),size_t(63),size_t(64),size_t(65),size_t(65536)}){
        sha256 h;for(size_t i=0;i<million.size();i+=chunk)h.update(million.data()+i,std::min(chunk,million.size()-i));
        test(h.finish()==million_hash,"SHA256 streaming chunk invariance");
    }
    auto base=self_test_input();test(parse_suite(base).cases.size()==1,"valid prepared input");
    auto mutate_reject=[&](auto mutate,const char * label){auto bad=base;mutate(bad);rejects([&]{parse_suite(bad);},label);};
    mutate_reject([](json & x){x["rows"]=127;},"rows exactly128");
    mutate_reject([](json & x){x["rows"]=128.0;},"rows integer type");
    mutate_reject([](json & x){x["schema_version"]=0;},"schema version");
    mutate_reject([](json & x){x["vocab_size"]=0;},"positive vocabulary");
    mutate_reject([](json & x){x["vocab_size"]=uint64_t(-1);},"unsigned integer overflow");
    mutate_reject([](json & x){x["cases"]=json::array();},"nonempty cases");
    mutate_reject([](json & x){x["cases"].push_back(x["cases"][0]);},"unique case id");
    mutate_reject([](json & x){x["cases"][0]["id"]="../escape";},"safe case id");
    mutate_reject([](json & x){x["cases"][0]["prompt_tokens"]={1,2};},"no prompt truncation");
    mutate_reject([](json & x){x["cases"][0]["prompt_original_ids"][0]=-1;},"negative token");
    mutate_reject([](json & x){x["cases"][0]["prompt_original_ids"][0]=32;},"token outside vocabulary");
    mutate_reject([](json & x){x["cases"][0]["context_tokens"]=8192;},"unsupported context");
    mutate_reject([](json & x){x["model_context_length"]=2048;x["cases"][0]["context_tokens"]=4096;},"model context admission");
    mutate_reject([](json & x){x["cases"][0]["prompt_original_ids"]=std::vector<int>(385,1);x["cases"][0]["prompt_tokens"]=x["cases"][0]["prompt_original_ids"];},"reserve128 continuation positions");
    auto exact=base;exact["cases"][0]["prompt_original_ids"]=std::vector<int>(384,1);exact["cases"][0]["prompt_tokens"]=exact["cases"][0]["prompt_original_ids"];
    test(parse_suite(exact).cases[0].prompt.size()+rows_per_case==512,"exact context fit");
    mutate_reject([](json & x){x["cases"][0]["teacher_forcing_tokens"].erase(127);},"teacher length128");
    mutate_reject([](json & x){x["cases"][0]["continuation_original_ids"].erase(127);},"continuation at least128");
    mutate_reject([](json & x){x["cases"][0]["teacher_forcing_tokens"][0]=31;},"teacher equals original prefix");
    mutate_reject([](json & x){x["cases"][0]["prompt_add_special"]=false;},"prompt special-token policy");
    mutate_reject([](json & x){x["cases"][0]["continuation_add_special"]=true;},"continuation special-token policy");
    mutate_reject([](json & x){x["cases"][0]["parse_special"]=false;},"parse special-token policy");
    mutate_reject([](json & x){x["cases"][0]["parse_special"]=1;},"boolean policy type");
    mutate_reject([](json & x){x["model_files"]=json::array();},"full model manifest required");
    mutate_reject([](json & x){x["model_files"][0]["name"]="../model.gguf";},"safe model basename");
    mutate_reject([](json & x){x["model_files"][0]["size_bytes"]=0;},"nonzero file size");
    mutate_reject([](json & x){x["model_files"][0]["sha256"]=std::string(64,'a');},"entry hash consistency");
    mutate_reject([](json & x){x["corpus_sha256"]=std::string(64,'G');},"lowercase hex SHA");
    const float tied[]={-1,2,2};test(finite_argmax(tied,3)==1,"first maximum wins ties");
    const float nan[]={0,std::numeric_limits<float>::quiet_NaN()};rejects([&]{finite_argmax(nan,2);},"reject NaN logits");
    const float inf[]={std::numeric_limits<float>::infinity()};rejects([&]{finite_argmax(inf,1);},"reject infinite logits");
    rejects([&]{finite_argmax(nullptr,0);},"reject empty logits");
    test(token_difference({1,2},{1,2}).is_null(),"token difference equality");
    test(token_difference({1,2},{1,3})==json({{"index",1},{"expected",2},{"actual",3}}),"token difference first substitution");
    test(token_difference({1,2},{1})==json({{"index",1},{"expected",2},{"actual",nullptr}}),"token difference shorter actual");
    test(token_difference({1},{1,2})==json({{"index",1},{"expected",nullptr},{"actual",2}}),"token difference longer actual");
    const auto dir=fs::temp_directory_path()/("dinf-matrix-self-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    require(fs::create_directory(dir),"cannot create self-test temp directory");
    try {
        const auto path=dir/"bytes";exclusive_file file(path);file.write(million.data(),million.size());file.flush();file.close();
        test(hash_file(path)==million_hash,"file hash matches byte hash");
        rejects([&]{exclusive_file duplicate(path);},"never overwrite output");
        test(hash_file(path)==million_hash,"failed overwrite preserves file");
        test(read_text(path)==million,"full file read");
        const auto large=dir/"large";exclusive_file empty(large);empty.close();fs::resize_file(large,max_input_bytes+1);
        rejects([&]{read_text(large);},"reject large input before allocation");
        fs::remove_all(dir);
    } catch (...) {fs::remove_all(dir);throw;}
    std::cout<<json({{"passed",true},{"checks",checks},{"models_loaded",0},{"backend_initialized",false},
                     {"micro_batch_tokens",micro_batch_tokens},{"prefill_chunk_tokens",prefill_chunk},
                     {"producer_source_sha256",LLAMA_MATRIX_SOURCE_SHA256}}).dump()<<'\n';return 0;
}

int main(int argc,char ** argv) {
    json progress;fs::path output;
    try {
        if(argc==2&&std::string(argv[1])=="--self-test")return self_tests();
        if(argc==2&&std::string(argv[1])=="--check-cpu-runtime"){
            backend_guard lifetime(true,"cpu");auto report=cpu_reference_runtime();
            report["models_loaded"]=0;report["backend_initialized"]=true;
            std::cout<<report.dump()<<'\n';return 0;
        }
        if(argc==5&&std::string(argv[1])=="--tokenize")return tokenize_probe(argv[2],argv[3],argv[4]);
        require(argc==5,"usage: llama_matrix_reference MODEL cpu|metal PREPARED.json NEW_OUTPUT_DIR");
        const std::string backend=argv[2];require(backend=="cpu"||backend=="metal","backend must be cpu or metal");
        const std::string revision=LLAMA_REFERENCE_REVISION;
        require(revision.size()>=7&&revision.size()<=40&&revision.find_first_not_of("0123456789abcdef")==std::string::npos,"build must pin llama.cpp revision");
        require(valid_hash(LLAMA_MATRIX_SOURCE_SHA256),"build must pin producer source SHA256");
        const uint32_t one=1;require(sizeof(float)==4&&std::numeric_limits<float>::is_iec559&&*reinterpret_cast<const uint8_t *>(&one)==1,"requires little-endian IEEE754 F32 host");
        const auto input_text=read_text(argv[3]);const auto input=json::parse(input_text);const auto suite=parse_suite(input);const auto prepared_sha=hash_text(input_text);
        output=argv[4];require(!fs::exists(output),"output directory already exists");require(fs::create_directories(output),"cannot create output directory");
        progress={{"schema_version",1},{"status","running"},{"suite_id",suite.id},{"rows_per_case",rows_per_case},
            {"source",std::string("llama.cpp ")+revision},{"backend",backend},{"architecture",suite.architecture},
            {"model_sha256",suite.model_sha},{"model_files",input.at("model_files")},{"corpus_sha256",suite.corpus_sha},
            {"prepared_sha256",prepared_sha},{"producer_source_sha256",LLAMA_MATRIX_SOURCE_SHA256},
            {"kv_dtype","F16"},{"threads",8},{"prefill_chunk_tokens",prefill_chunk},{"new_context_per_case",true},
            {"micro_batch_tokens",micro_batch_tokens},
            {"expected_cases",suite.cases.size()},{"cases",json::array()}};
        const auto weight_stats=model_file_stats(argv[1],suite);
        progress["model_file_stats_before"]=weight_stats;
        progress["weight_storage"]=validate_model_files(argv[1],suite);
        require(model_file_stats(argv[1],suite)==weight_stats,"model file stats changed during hash/header verification");
        progress["model_files_sha256_verified_once"]=true;
        backend_guard backend_lifetime(true,backend);
        if(backend=="cpu"){progress["cpu_attention_policy"]=cpu_reference_policy();progress["cpu_runtime_before"]=cpu_reference_runtime();}
        ggml_backend_dev_t devices[]={nullptr,nullptr};
        if(backend=="metal"){
            const auto reg=ggml_backend_reg_by_name("MTL");require(reg!=nullptr&&ggml_backend_reg_dev_count(reg)>0,"Metal backend unavailable");devices[0]=ggml_backend_reg_dev_get(reg,0);
        }
        auto mp=llama_model_default_params();mp.devices=devices;mp.n_gpu_layers=backend=="metal"?-1:0;
        auto model=std::unique_ptr<llama_model,decltype(&llama_model_free)>(llama_model_load_from_file(argv[1],mp),llama_model_free);
        require(bool(model),"model load failed");require(model_architecture(model.get())==suite.architecture,"model architecture differs from prepared input");
        require(llama_vocab_n_tokens(llama_model_get_vocab(model.get()))==suite.vocab,"model vocabulary differs from prepared input");
        require(llama_model_n_ctx_train(model.get())>0&&uint32_t(llama_model_n_ctx_train(model.get()))==suite.model_context,"model context differs from prepared input");
        progress["runtime_system_info"]=llama_print_system_info();progress["model_parameter_count"]=llama_model_n_params(model.get());progress["model_encoded_bytes"]=llama_model_size(model.get());
        // Admit every case before doing any inference. A later prompt's
        // tokenizer mismatch must not consume time on earlier numeric cases.
        for(const auto & c:suite.cases){
            require(tokenize(llama_model_get_vocab(model.get()),c.prompt_text,true)==c.prompt,c.id+": independent prompt tokenization differs");
            require(tokenize(llama_model_get_vocab(model.get()),c.continuation_text,false)==c.continuation,c.id+": independent continuation tokenization differs");
        }
        progress["all_cases_tokenization_verified_before_decode"]=true;
        for(const auto & c:suite.cases){
            std::cerr<<"CASE_START "<<c.id<<'\n';
            progress["cases"].push_back(run_case(model.get(),suite,c,input,output,backend,prepared_sha));
            std::cerr<<"CASE_COMPLETE "<<c.id<<'\n';
        }
        require(hash_file(argv[3])==prepared_sha,"prepared input changed during suite");
        progress["model_file_stats_after"]=model_file_stats(argv[1],suite);
        require(progress["model_file_stats_after"]==weight_stats,"model file stats changed during suite");
        if(backend=="cpu"){
            progress["cpu_runtime_after"]=cpu_reference_runtime();
            require(progress["cpu_runtime_after"]["policy"]==progress["cpu_runtime_before"]["policy"],"CPU runtime changed");
        }
        progress["status"]="complete";progress["completed_cases"]=suite.cases.size();
        write_json(output/"suite.json",progress);std::cout<<(output/"suite.json").string()<<'\n';return 0;
    } catch(const std::exception & error) {
        std::cerr<<error.what()<<'\n';
        // Completed cases remain intact; a failure file never acts as a suite pass.
        if(!progress.is_null()&&!output.empty()){
            try{progress["status"]="failed";progress["error"]=error.what();write_json(output/"failure.json",progress);}catch(const std::exception & write_error){std::cerr<<"failure report: "<<write_error.what()<<'\n';}
        }
        return 1;
    }
}
