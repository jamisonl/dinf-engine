// Synthetic original-F32-query check; no model inputs or weights.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
namespace fs=std::filesystem;using json=nlohmann::json;
static void need(bool v,const char*s){if(!v)throw std::runtime_error(s);}
static void save(const fs::path&p,const std::vector<float>&v){need(!fs::exists(p),"fresh raw output");std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),v.size()*4);need(bool(f),"write");}
static json one(ggml_backend_t backend,const fs::path&root,size_t visible,size_t rows){
 constexpr size_t D=128,H=4,HK=2;const size_t N=(visible+255)/256*256;const std::string id="n"+std::to_string(visible)+"-q"+std::to_string(rows);auto dir=root/id;need(fs::create_directory(dir),"fresh case");std::vector<float>q(H*rows*D);std::vector<ggml_fp16_t>k(HK*N*D),v(HK*N*D),mask(rows*N);size_t rounded=0;
 for(size_t i=0;i<q.size();++i){q[i]=float(int(i*17%101)-50)/37.f;rounded+=q[i]!=ggml_fp16_to_fp32(ggml_fp32_to_fp16(q[i]));}
 for(size_t h=0;h<HK;++h)for(size_t t=0;t<visible;++t)for(size_t d=0;d<D;++d){size_t i=(h*N+t)*D+d;k[i]=ggml_fp32_to_fp16(float(int((h*137+t*23+d*11)%67)-33)/32.f);v[i]=ggml_fp32_to_fp16(float(int((h*83+t*13+d*7)%71)-35)/64.f);}
 for(size_t t=0;t<rows;++t)for(size_t j=0;j<N;++j)mask[t*N+j]=ggml_fp32_to_fp16(j<visible-rows+t+1?0.f:-INFINITY);
 const float scale=1.f/std::sqrt(float(D));std::vector<double>reference(rows*H*D);
 for(size_t h=0;h<H;++h)for(size_t t=0;t<rows;++t){const size_t count=visible-rows+t+1;std::vector<double>p(count);double mx=-std::numeric_limits<double>::infinity();for(size_t j=0;j<count;++j){double dot=0.;for(size_t d=0;d<D;++d)dot+=double(q[(h*rows+t)*D+d])*double(ggml_fp16_to_fp32(k[((h/2)*N+j)*D+d]));p[j]=dot*double(scale);mx=std::max(mx,p[j]);}double sum=0.;for(double&x:p){x=std::exp(x-mx);sum+=x;}for(size_t d=0;d<D;++d){double y=0.;for(size_t j=0;j<count;++j)y+=(p[j]/sum)*double(ggml_fp16_to_fp32(v[((h/2)*N+j)*D+d]));reference[(t*H+h)*D+d]=y;}}
 json rec={{"id",id},{"visible",visible},{"rows",rows},{"heads",H},{"kv_heads",HK},{"dim",D},{"physical",N},{"values",reference.size()},{"q_values_changed_by_half",rounded}};
 for(bool precise:{false,true}){auto ctx=ggml_init({1024*1024,nullptr,true});need(ctx,"context");auto tq=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,D,rows,H),tk=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,D,N,HK),tv=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,D,N,HK),tm=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,N,rows);auto y=ggml_flash_attn_ext(ctx,tq,tk,tv,tm,scale,0.f,0.f);if(precise)ggml_flash_attn_ext_set_prec(y,GGML_PREC_F32);auto g=ggml_new_graph(ctx);ggml_build_forward_expand(g,y);auto buf=ggml_backend_alloc_ctx_tensors(ctx,backend);need(buf,"buffer");ggml_backend_tensor_set(tq,q.data(),0,q.size()*4);ggml_backend_tensor_set(tk,k.data(),0,k.size()*2);ggml_backend_tensor_set(tv,v.data(),0,v.size()*2);ggml_backend_tensor_set(tm,mask.data(),0,mask.size()*2);need(ggml_backend_graph_compute(backend,g)==GGML_STATUS_SUCCESS,"compute");std::vector<float>actual(reference.size());ggml_backend_tensor_get(y,actual.data(),0,actual.size()*4);double maximum=0.,squares=0.;for(size_t i=0;i<actual.size();++i){need(std::isfinite(actual[i]),"finite output");const double d=double(actual[i])-reference[i];maximum=std::max(maximum,std::abs(d));squares+=d*d;}const std::string name=precise?"f32":"default";save(dir/(name+".f32"),actual);rec[name]={{"file",id+"/"+name+".f32"},{"max_abs_to_f64_original_q",maximum},{"rms_to_f64_original_q",std::sqrt(squares/actual.size())}};ggml_backend_buffer_free(buf);ggml_free(ctx);}
 return rec;
}
int main(int argc,char**argv){try{need(argc==3,"query_probe CPU_PLUGIN NEW_OUTPUT");fs::path out=argv[2];need(fs::create_directory(out),"fresh output");auto reg=ggml_backend_load(argv[1]);need(reg&&ggml_backend_reg_dev_count(reg)==1,"one CPU device");auto dev=ggml_backend_reg_dev_get(reg,0);need(ggml_backend_dev_type(dev)==GGML_BACKEND_DEVICE_TYPE_CPU,"CPU only");auto backend=ggml_backend_dev_init(dev,nullptr);need(backend,"backend");auto threads=reinterpret_cast<ggml_backend_set_n_threads_t>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_set_n_threads"));need(threads,"thread setter");threads(backend,2);json r={{"schema_version",1},{"plugin",fs::canonical(argv[1]).string()},{"model_calls",0},{"gpu_calls",0},{"threads",2},{"cases",json::array()}};for(size_t n:{1,3,31,33,257})for(size_t rows:{1,3})if(rows<=n)r["cases"].push_back(one(backend,out,n,rows));ggml_backend_free(backend);std::ofstream f(out/"report.json");f<<r.dump(2)<<'\n';need(bool(f),"report");return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
