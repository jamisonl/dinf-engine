// Explicit CPU reference runtime; enabled only by the pinned fixture builder.
#pragma once
#include <nlohmann/json.hpp>
#include <mach-o/dyld.h>
#include <array>
namespace dinf_reference {
using json=nlohmann::json;
namespace fs=std::filesystem;
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


static std::string hash_file(const fs::path & p) {
 require(fs::is_regular_file(p)&&fs::file_size(p)<=64*1024*1024,"runtime file bound");std::ifstream f(p,std::ios::binary);std::array<char,65536>b;sha256 h;while(f){f.read(b.data(),b.size());h.update(b.data(),size_t(f.gcount()));}require(f.eof(),"runtime hash read");return h.finish();
}
static json policy(){
 auto p=json::parse(LLAMA_CPU_REFERENCE_JSON);require(p.is_object()&&(p.at("id")=="ggml-cpu-flash-f32-v2"||p.at("id")=="ggml-cpu-f32-activations-v3"||p.at("id")=="ggml-cpu-flash-f32-query-v4"),"known CPU repair required");require(p.at("source_revision")==LLAMA_REFERENCE_REVISION,"CPU source revision mismatch");for(const char *k:{"plugin","blas_plugin"}){const auto&d=p.at(k);const fs::path path=d.at("path").get<std::string>();require(path.is_absolute()&&fs::canonical(path)==path,"canonical plugin path");require(fs::file_size(path)==d.at("size_bytes").get<uintmax_t>()&&hash_file(path)==d.at("sha256").get<std::string>(),"plugin hash/size changed");}return p;
}
static json runtime(){
 const auto p=policy();require(ggml_backend_reg_count()==2,"only CPU and BLAS registries");size_t cpu=0,blas=0;json images=json::array();for(uint32_t i=0;i<_dyld_image_count();++i){const fs::path path=_dyld_get_image_name(i);const auto name=path.filename().string();if(name.find("libggml-cpu")==0){++cpu;require(fs::canonical(path).string()==p.at("plugin").at("path"),"unexpected CPU image");}if(name=="libggml-blas.so"){++blas;require(fs::canonical(path).string()==p.at("blas_plugin").at("path"),"unexpected BLAS image");}require(name!="libggml-metal.so","no Metal image");images.push_back(path.string());}require(cpu==1&&blas==1,"exact CPU/BLAS image count");for(size_t i=0;i<ggml_backend_dev_count();++i){auto t=ggml_backend_dev_type(ggml_backend_dev_get(i));require(t!=GGML_BACKEND_DEVICE_TYPE_GPU&&t!=GGML_BACKEND_DEVICE_TYPE_IGPU,"no GPU device");}return {{"policy",p},{"runtime_images",images},{"cpu_plugin_image_count",cpu},{"blas_plugin_image_count",blas},{"gpu_backends_loaded",false},{"registry_count",2}};
}
static json load_cpu(){
 const auto p=policy();require(ggml_backend_reg_count()==0,"empty initial registry");const auto b=ggml_backend_load(p.at("blas_plugin").at("path").get<std::string>().c_str());require(b&&std::string(ggml_backend_reg_name(b))=="BLAS","explicit BLAS load");const auto c=ggml_backend_load(p.at("plugin").at("path").get<std::string>().c_str());require(c&&std::string(ggml_backend_reg_name(c))=="CPU"&&ggml_backend_reg_dev_count(c)==1,"explicit CPU load");llama_backend_init();return runtime();
}


} // namespace dinf_reference
