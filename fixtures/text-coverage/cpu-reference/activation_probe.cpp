// SPDX-License-Identifier: MIT
// Independent F64 checks for CPU softplus and F32 tanh-GELU reference repairs.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using json=nlohmann::json;
namespace fs=std::filesystem;
static void require(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
template<class T>static void save(const fs::path &p,const std::vector<T>&v){require(!fs::exists(p),"output exists");std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));require(bool(f),"write failed");}
static json softplus(ggml_backend_t backend,const fs::path &out){
 const std::vector<float>input={-80,-30,-20,-17,-15,-10,-5,-1,0,1,19,20,25,80};const size_t n=input.size();
 auto ctx=ggml_init({1024*1024,nullptr,true});require(ctx,"context");
 auto state=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,1,1,1,n);auto x=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,1,1,1,n);auto dt=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,1,1,n);auto a=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,1,1);auto ids=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,n);
 // Separate zero-state sequences: x=B=C=1, A=0 gives y=softplus(dt).
 auto y=ggml_ssm_scan(ctx,state,x,dt,a,x,x,ids);auto graph=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(graph,y);auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);require(buffer,"buffer");
 std::vector<float>zeros(n,0),ones(n,1),actual(n*2);std::vector<int32_t>indexes(n);for(size_t i=0;i<n;++i)indexes[i]=i;
 ggml_backend_tensor_set(state,zeros.data(),0,n*4);ggml_backend_tensor_set(x,ones.data(),0,n*4);ggml_backend_tensor_set(dt,input.data(),0,n*4);ggml_backend_tensor_set(a,zeros.data(),0,4);ggml_backend_tensor_set(ids,indexes.data(),0,n*4);
 require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"softplus graph");ggml_backend_tensor_get(y,actual.data(),0,actual.size()*4);double worst=0;
 for(size_t i=0;i<n;++i){require(std::isfinite(actual[i])&&actual[i]==actual[i+n],"finite matching state");double expected=std::log1p(std::exp(double(input[i])));worst=std::max(worst,std::abs(double(actual[i])-expected)/expected);}
 save(out/"softplus-input.f32",input);save(out/"softplus-output.f32",actual);ggml_backend_buffer_free(buffer);ggml_free(ctx);return {{"max_relative_error",worst},{"relative_limit",2e-7},{"passed",worst<=2e-7},{"sequences",n}};
}
static json gelu(ggml_backend_t backend,const fs::path &out){
 constexpr size_t n=65536;std::vector<float>input(n),actual(n);std::vector<ggml_fp16_t>half_input(n),half_output(n);
 for(size_t i=0;i<n;++i){input[i]=float(int(i)-32768)*0.000371f;half_input[i]=ggml_fp32_to_fp16(input[i]);}
 auto ctx=ggml_init({1024*1024,nullptr,true});require(ctx,"context");auto x=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,n);auto h=ggml_new_tensor_1d(ctx,GGML_TYPE_F16,n);auto y=ggml_gelu(ctx,x);auto hy=ggml_gelu(ctx,h);auto graph=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(graph,y);ggml_build_forward_expand(graph,hy);auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);require(buffer,"buffer");ggml_backend_tensor_set(x,input.data(),0,n*4);ggml_backend_tensor_set(h,half_input.data(),0,n*2);
 require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"GELU graph");ggml_backend_tensor_get(y,actual.data(),0,n*4);ggml_backend_tensor_get(hy,half_output.data(),0,n*2);double worst=0,maxabs=0,squared=0;
 for(size_t i=0;i<n;++i){require(std::isfinite(actual[i]),"finite GELU");double v=input[i],expected=.5*v*(1+std::tanh(std::sqrt(2/std::acos(-1.))*v*(1+.044715*v*v)));double error=std::abs(double(actual[i])-expected);maxabs=std::max(maxabs,error);squared+=error*error;worst=std::max(worst,error/(1e-6+2e-6*std::abs(expected)));}
 save(out/"gelu-input.f32",input);save(out/"gelu-output.f32",actual);save(out/"gelu-output.f16",half_output);ggml_backend_buffer_free(buffer);ggml_free(ctx);return {{"max_abs",maxabs},{"rms",std::sqrt(squared/n)},{"worst_bound_ratio",worst},{"bound","1e-6 + 2e-6*abs(F64 tanh-GELU)"},{"passed",worst<=1},{"values",n}};
}
int main(int argc,char**argv){try{
 require(argc==3,"usage: activation_probe CPU_PLUGIN NEW_OUTPUT");const fs::path out=argv[2];require(!fs::exists(out)&&fs::create_directory(out),"fresh output required");require(ggml_backend_reg_count()==0,"empty registry required");auto reg=ggml_backend_load(argv[1]);require(reg&&ggml_backend_reg_dev_count(reg)==1,"CPU registry");auto dev=ggml_backend_reg_dev_get(reg,0);require(ggml_backend_dev_type(dev)==GGML_BACKEND_DEVICE_TYPE_CPU,"CPU device");auto backend=ggml_backend_dev_init(dev,nullptr);require(backend,"backend");auto set_threads=reinterpret_cast<ggml_backend_set_n_threads_t>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_set_n_threads"));require(set_threads,"thread setter");set_threads(backend,2);
 json report={{"schema_version",1},{"cpu_plugin",fs::canonical(argv[1]).string()},{"models_loaded",0},{"qualifying",false},{"threads",2}};report["softplus"]=softplus(backend,out);report["gelu"]=gelu(backend,out);report["passed"]=report["softplus"]["passed"].get<bool>()&&report["gelu"]["passed"].get<bool>();ggml_backend_free(backend);std::ofstream f(out/"report.json");f<<report.dump(2)<<'\n';require(bool(f),"report write");std::cout<<report.dump()<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
