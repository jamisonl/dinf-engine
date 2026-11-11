#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "nlohmann/json.hpp"
#include <mach-o/dyld.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
using json=nlohmann::json;
namespace fs=std::filesystem;
static void check(bool x,const std::string&s){if(!x)throw std::runtime_error(s);}
template<class T>static void save(const fs::path&p,const std::vector<T>&v){check(!fs::exists(p),"output exists");std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),std::streamsize(v.size()*sizeof(T)));check(bool(f),"write failed");}
static float input(size_t i,uint32_t seed){uint32_t x=uint32_t(i)*1664525u+seed;x^=x>>13;x*=2246822519u;x^=x>>16;return (float(int32_t(x%20001u)-10000)/10000.f)*.7f;}
static json one(ggml_backend_t backend,const fs::path&dir,size_t length,bool constant){
 constexpr size_t D=128,H=32,HK=4;
 const std::string id=(constant?"uniform-":"random-")+std::to_string(length);const auto out=dir/id;check(fs::create_directory(out),"case dir exists");
 std::vector<float>q(D*H);std::vector<ggml_fp16_t>k(D*length*HK),v(k.size());
 for(size_t i=0;i<q.size();++i)q[i]=constant?0.f:input(i,17u)+float(i%5)*0.000071f;
 for(size_t i=0;i<k.size();++i){k[i]=ggml_fp32_to_fp16(constant?0.f:input(i,29u));v[i]=ggml_fp32_to_fp16(constant?4096.f:input(i,47u));}
 const float scale=1.f/std::sqrt(float(D));
 ggml_init_params params{2*1024*1024,nullptr,true};auto*ctx=ggml_init(params);check(ctx,"context");
 auto*tq=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,D,1,H,1);auto*tk=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,D,length,HK,1);auto*tv=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,D,length,HK,1);
 auto*result=ggml_flash_attn_ext(ctx,tq,tk,tv,nullptr,scale,0.f,0.f);ggml_flash_attn_ext_set_prec(result,GGML_PREC_F32);
 auto*graph=ggml_new_graph_custom(ctx,64,false);ggml_build_forward_expand(graph,result);auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);check(buffer,"CPU buffer");
 check(ggml_backend_dev_type(ggml_backend_get_device(backend))==GGML_BACKEND_DEVICE_TYPE_CPU,"not CPU");
 ggml_backend_tensor_set(tq,q.data(),0,q.size()*4);ggml_backend_tensor_set(tk,k.data(),0,k.size()*2);ggml_backend_tensor_set(tv,v.data(),0,v.size()*2);
 check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"graph failed");ggml_backend_synchronize(backend);std::vector<float>actual(D*H);ggml_backend_tensor_get(result,actual.data(),0,actual.size()*4);
 std::vector<double>ref32(D*H),ref16(D*H);std::vector<double>scores(length);
 for(int rounded=0;rounded<2;++rounded){auto&ref=rounded?ref16:ref32;for(size_t h=0;h<H;++h){double mx=-std::numeric_limits<double>::infinity();size_t kh=h/(H/HK);for(size_t t=0;t<length;++t){double dot=0;for(size_t d=0;d<D;++d){float query=q[h*D+d];if(rounded)query=ggml_fp16_to_fp32(ggml_fp32_to_fp16(query));dot+=double(query)*double(ggml_fp16_to_fp32(k[(kh*length+t)*D+d]));}scores[t]=dot*double(scale);mx=std::max(mx,scores[t]);}double sum=0;for(auto&s:scores){s=std::exp(s-mx);sum+=s;}for(size_t d=0;d<D;++d){double value=0;for(size_t t=0;t<length;++t)value+=(scores[t]/sum)*double(ggml_fp16_to_fp32(v[(kh*length+t)*D+d]));ref[h*D+d]=value;}}}
 save(out/"q.f32",q);save(out/"k.f16",k);save(out/"v.f16",v);save(out/"actual.f32",actual);save(out/"reference-q32.f64",ref32);save(out/"reference-q16.f64",ref16);
 size_t nonfinite=0;double max32=0,max16=0,sq32=0,sq16=0;float lo=std::numeric_limits<float>::infinity(),hi=-lo;
 for(size_t i=0;i<actual.size();++i){if(!std::isfinite(actual[i])){++nonfinite;continue;}lo=std::min(lo,actual[i]);hi=std::max(hi,actual[i]);double d32=double(actual[i])-ref32[i],d16=double(actual[i])-ref16[i];max32=std::max(max32,std::abs(d32));max16=std::max(max16,std::abs(d16));sq32+=d32*d32;sq16+=d16*d16;}
 json r={{"id",id},{"shape",{{"query",{1,H,1,D}},{"key_value",{1,HK,length,D}},{"output",{1,H,1,D}}}},{"ggml_query_ne",{D,1,H,1}},{"ggml_key_value_ne",{D,length,HK,1}},{"query_dtype","F32"},{"key_value_dtype","F16"},{"precision","GGML_PREC_F32"},{"threads",8},{"scale_f32",scale},{"nonfinite_outputs",nonfinite},{"total_outputs",actual.size()},{"all_finite",nonfinite==0},{"models_loaded",0},{"gpu_backends_initialized",0}};
 if(nonfinite==0){r["min_output"]=lo;r["max_output"]=hi;r["max_abs_to_f64_q32"]=max32;r["rms_to_f64_q32"]=std::sqrt(sq32/actual.size());r["max_abs_to_f64_q16"]=max16;r["rms_to_f64_q16"]=std::sqrt(sq16/actual.size());}
 r["constant_expected"]=constant?json(4096):json(nullptr);r["max_tensor_bytes"]=std::max(k.size()*2,q.size()*4);ggml_backend_buffer_free(buffer);ggml_free(ctx);return r;
}
int main(int argc,char**argv){try{check(argc==3,"usage: probe CPU_PLUGIN NEW_OUTPUT");const fs::path out=argv[2];check(!fs::exists(out)&&fs::create_directory(out),"output exists");auto reg=ggml_backend_load(argv[1]);check(reg&&ggml_backend_reg_dev_count(reg)==1,"CPU registry");auto dev=ggml_backend_reg_dev_get(reg,0);check(ggml_backend_dev_type(dev)==GGML_BACKEND_DEVICE_TYPE_CPU,"not CPU registry");auto backend=ggml_backend_dev_init(dev,nullptr);check(backend,"CPU init");auto set_threads=reinterpret_cast<ggml_backend_set_n_threads_t>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_set_n_threads"));check(set_threads,"CPU thread setter missing");set_threads(backend,8);
 json report={{"schema_version",1},{"backend",ggml_backend_name(backend)},{"device",ggml_backend_dev_description(dev)},{"cpu_plugin",fs::canonical(argv[1]).string()},{"threads",8},{"models_loaded",0},{"gpu_backends_initialized",0},{"cases",json::array()}};
 for(size_t n:{15u,16u,17u,256u,512u,1024u})report["cases"].push_back(one(backend,out,n,true));for(size_t n:{17u,127u,512u,1024u})report["cases"].push_back(one(backend,out,n,false));ggml_backend_free(backend);
 report["runtime_images"]=json::array();for(uint32_t i=0;i<_dyld_image_count();++i)report["runtime_images"].push_back(_dyld_get_image_name(i));std::ofstream f(out/"report.json");f<<report.dump(2)<<'\n';check(bool(f),"report write");std::cout<<report.dump()<<'\n';return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
