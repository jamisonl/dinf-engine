// Independent DFlash2 component fixture against pinned llama.cpp de5122d879.
// --inspect TARGET DRAFT: metadata only, never initializes a backend.
// --derive TARGET DRAFT OUTPUT: copy raw draft tensors + target embedding/head.
// --produce DERIVED TARGET_SHA DRAFT_SHA DERIVED_SHA cpu|metal PREFIX ROWS BLOCK
// Caller verifies all three full hashes and every copied tensor range first.
// No full target model loads; synthetic feature rows test math, not text quality.
#include "llama.h"
#include "llama-ext.h"
#include "gguf.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
static void need(bool ok,const char *m){if(!ok)throw std::runtime_error(m);}
static std::string q(const std::string &s){std::string o="\"";for(unsigned char c:s){need(c>=32,"control in path");if(c=='"'||c=='\\')o+='\\';o+=char(c);}return o+'"';}
struct header {
    ggml_context * tensors=nullptr;gguf_context * meta=nullptr;
    explicit header(const char *p){meta=gguf_init_from_file(p,{true,&tensors});need(meta&&tensors,"invalid GGUF header");}
    ~header(){gguf_free(meta);ggml_free(tensors);}
};
static uint64_t num(gguf_context *m,const char *key){auto i=gguf_find_key(m,key);need(i>=0,"missing integer key");switch(gguf_get_kv_type(m,i)){case GGUF_TYPE_UINT32:return gguf_get_val_u32(m,i);case GGUF_TYPE_UINT64:return gguf_get_val_u64(m,i);case GGUF_TYPE_INT32:{auto v=gguf_get_val_i32(m,i);need(v>=0,"negative key");return uint64_t(v);}default:throw std::runtime_error("wrong integer key type");}}
static size_t bounded(const char *s,size_t max){std::string v(s);need(!v.empty()&&v.find_first_not_of("0123456789")==std::string::npos,"invalid integer");auto n=std::stoull(v);need(n>0&&n<=max,"integer outside limit");return size_t(n);}
static bool hash(const char*s){std::string v(s);return v.size()==64&&v.find_first_not_of("0123456789abcdef")==std::string::npos;}
struct copy {std::string name,path;size_t offset,bytes;ggml_tensor*tensor;};
static std::vector<copy> copies(const char *target,const char *draft,header &t,header &d){
    auto arch=gguf_find_key(d.meta,"general.architecture");need(arch>=0&&std::string(gguf_get_val_str(d.meta,arch))=="dflash","expected DFlash");
    std::vector<copy> all;
    auto add=[&](header &h,const char *path,const char *name){auto i=gguf_find_tensor(h.meta,name);need(i>=0,"missing tensor");auto *tensor=ggml_get_tensor(h.tensors,name);need(tensor,"missing tensor metadata");auto offset=gguf_get_data_offset(h.meta)+gguf_get_tensor_offset(h.meta,i);auto bytes=ggml_nbytes(tensor);need(bytes>0&&offset+bytes<=std::filesystem::file_size(path),"tensor outside source");all.push_back({name,path,offset,bytes,tensor});};
    for(int64_t i=0;i<gguf_get_n_tensors(d.meta);++i)add(d,draft,gguf_get_tensor_name(d.meta,i));
    for(const char *name:{"token_embd.weight","output.weight"}){need(gguf_find_tensor(d.meta,name)<0,"draft already has shared weights");add(t,target,name);need(all.back().tensor->ne[0]==int64_t(num(d.meta,"dflash.embedding_length")),"shared width differs");}
    need(all[all.size()-1].tensor->ne[1]==all[all.size()-2].tensor->ne[1],"shared vocabulary differs");return all;
}
static int derive(const char *target,const char *draft,const char *output){
    header t(target),d(draft);auto all=copies(target,draft,t,d);
    std::unique_ptr<gguf_context,decltype(&gguf_free)> out(gguf_init_empty(),gguf_free);gguf_set_kv(out.get(),d.meta);
    for(const auto &x:all)gguf_add_tensor(out.get(),x.tensor);
    size_t total=gguf_get_meta_size(out.get());for(const auto &x:all)total+=x.bytes+gguf_get_alignment(out.get());need(total<=size_t(10)*1024*1024*1024,"derived file exceeds 10GiB");
    std::unique_ptr<FILE,decltype(&fclose)> file(fopen(output,"wbx"),fclose);need(bool(file),"output exists or cannot create");need(gguf_write_to_file_ptr(out.get(),file.get(),true),"cannot write header");
    std::vector<char> buffer(1024*1024);std::cout<<"{\"tensor_copies\":[";
    for(size_t i=0;i<all.size();++i){const auto &x=all[i];const auto dest=gguf_get_meta_size(out.get())+gguf_get_tensor_offset(out.get(),int64_t(i));need(fseeko(file.get(),off_t(dest),SEEK_SET)==0,"seek output");std::ifstream input(x.path,std::ios::binary);input.seekg(std::streamoff(x.offset));size_t left=x.bytes;
        while(left){auto n=std::min(left,buffer.size());input.read(buffer.data(),std::streamsize(n));need(size_t(input.gcount())==n,"short source");need(fwrite(buffer.data(),1,n,file.get())==n,"short output");left-=n;}
        if(i)std::cout<<',';std::cout<<"{\"name\":"<<q(x.name)<<",\"source\":"<<q(x.path)<<",\"source_offset\":"<<x.offset<<",\"derived_offset\":"<<dest<<",\"bytes\":"<<x.bytes<<'}';}
    need(fflush(file.get())==0,"flush output");std::cout<<"],\"derived\":"<<q(output)<<",\"raw_quantized_bytes_unchanged\":true}\n";return 0;
}
static void floats(const std::string &path,const float *v,size_t n){
    static_assert(sizeof(float)==4&&std::numeric_limits<float>::is_iec559);uint32_t endian=1;need(*reinterpret_cast<uint8_t*>(&endian)==1,"little endian required");need(v&&n>0&&n<=64*1024*1024,"invalid tensor output");need(std::all_of(v,v+n,[](float x){return std::isfinite(x);}),"nonfinite values");
    std::unique_ptr<FILE,decltype(&fclose)> f(fopen(path.c_str(),"wbx"),fclose);need(bool(f),"output exists or cannot create");need(fwrite(v,sizeof(float),n,f.get())==n&&fflush(f.get())==0,"short float output");
}
// With pooling NONE, llama.cpp still renames the normalized output to
// result_embd_pooled. Observe that actual graph tensor, not its earlier name.
struct observed_buffers { std::map<std::string,std::string> names; std::vector<float> input_embeddings; };
static bool observe(ggml_tensor *t,bool ask,void *user){
    const std::string name(t->name);
    const bool wanted=name=="result_embd_pooled"||name=="result_output"||name=="dflash2_lattice"||name=="inp_noise_embd";
    if(ask)return wanted;
    if(wanted){auto *buffer=t->buffer;if(!buffer&&t->view_src)buffer=t->view_src->buffer;static_cast<observed_buffers*>(user)->names[name]=buffer?ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer)):"unavailable";}
    if(name=="inp_noise_embd"){
        if(t->type!=GGML_TYPE_F32||!ggml_is_contiguous(t)||ggml_nelements(t)<=0||ggml_nelements(t)>64*1024*1024)return false;
        auto &values=static_cast<observed_buffers*>(user)->input_embeddings;values.resize(size_t(ggml_nelements(t)));ggml_backend_tensor_get(t,values.data(),0,values.size()*sizeof(float));
    }
    return true;
}
static int produce(int argc,char **argv){
    need(argc==10,"--produce DERIVED TARGET_SHA DRAFT_SHA DERIVED_SHA cpu|metal PREFIX ROWS BLOCK");
    for(int i=3;i<=5;++i)need(hash(argv[i]),"invalid caller-verified hash");
    const std::string backend=argv[6],prefix=argv[7];need(backend=="cpu"||backend=="metal","invalid backend");
    const auto rows=bounded(argv[8],4096),block=bounded(argv[9],256);
    header h(argv[2]);const auto width=num(h.meta,"dflash.embedding_length"),trained=num(h.meta,"dflash.block_size"),mask=num(h.meta,"tokenizer.ggml.mask_token_id");
    auto idx=gguf_find_key(h.meta,"dflash.target_layers");need(idx>=0&&gguf_get_kv_type(h.meta,idx)==GGUF_TYPE_ARRAY,"missing target layers");const auto layers=gguf_get_arr_n(h.meta,idx);need(layers>1&&layers<=64&&width<=16384&&block>=2&&block<=trained,"invalid draft geometry");
    const size_t fused=width*layers;need(rows*fused<=64*1024*1024,"feature fixture too large");std::vector<float> features(rows*fused);
    for(size_t i=0;i<features.size();++i)features[i]=float(int((i*17+23)%257)-128)/128.0f;
    floats(prefix+".features.f32",features.data(),features.size());
    llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=backend=="metal"?999:0;
    std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[2],mp),llama_model_free);need(bool(model),"cannot load derived draft");
    const auto vocab=llama_vocab_n_tokens(llama_model_get_vocab(model.get()));need(mask<uint64_t(vocab)&&vocab>7,"invalid mask/vocabulary");
    observed_buffers observed;
    auto cp=llama_context_default_params();cp.cb_eval=observe;cp.cb_eval_user_data=&observed;cp.n_ctx=uint32_t(std::max<size_t>(512,rows+block));cp.n_batch=uint32_t(std::max(rows,block));cp.n_ubatch=cp.n_batch;cp.n_seq_max=1;cp.n_threads=4;cp.n_threads_batch=4;cp.type_k=GGML_TYPE_F16;cp.type_v=GGML_TYPE_F16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.embeddings=true;cp.pooling_type=LLAMA_POOLING_TYPE_NONE;cp.offload_kqv=backend=="metal";cp.op_offload=backend=="metal";
    llama_model_set_attn_rot(model.get(),false);
    std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);need(bool(ctx),"cannot create draft context");llama_set_causal_attn(ctx.get(),false);llama_set_embeddings_nextn(ctx.get(),true,false);
    std::unique_ptr<llama_batch_ext,decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx.get()),llama_batch_ext_free);need(bool(batch),"cannot create batch");
    for(size_t i=0;i<rows;++i){auto b=llama_batch_ext_add_embd(batch.get(),0,{features.data()+i*fused,1,fused});llama_pos pos[4]={int32_t(i),int32_t(i),int32_t(i),0};need(b>=0&&llama_batch_ext_set_pos(batch.get(),b,pos),"cannot add features");}
    need(llama_process(ctx.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get())==0,"draft injection failed");llama_batch_ext_clear(batch.get());
    for(size_t i=0;i<block;++i){auto b=llama_batch_ext_add_token(batch.get(),0,i==0?7:int32_t(mask));llama_pos pos[4]={int32_t(rows+i),int32_t(rows+i),int32_t(rows+i),0};need(b>=0&&llama_batch_ext_set_pos(batch.get(),b,pos)&&llama_batch_ext_set_output_logits(batch.get(),b,true),"cannot add noise token");}
    need(llama_process(ctx.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get())==0,"draft decode failed");
    need(observed.input_embeddings.size()==block*width,"missing actual graph input embeddings");
    floats(prefix+".input_embeddings.f32",observed.input_embeddings.data(),observed.input_embeddings.size());
    floats(prefix+".hidden.f32",llama_get_embeddings(ctx.get()),block*width);floats(prefix+".logits.f32",llama_get_logits(ctx.get()),block*size_t(vocab));floats(prefix+".lattice.f32",llama_get_embeddings_nextn(ctx.get()),block*width);
    need(observed.names.size()==4,"missing compute backend evidence");
    std::cout<<"{\"schema_version\":1,\"producer\":\"llama.cpp DFlash2 component\",\"revision\":\"de5122d879d00d2895c36be7e0cee94105aeeabd\",\"backend\":"<<q(backend)<<",\"target_sha256\":"<<q(argv[3])<<",\"draft_sha256\":"<<q(argv[4])<<",\"derived_sha256\":"<<q(argv[5])<<",\"rows\":"<<rows<<",\"block\":"<<block<<",\"anchor\":7,\"mask\":"<<mask<<",\"width\":"<<width<<",\"layers\":"<<layers<<",\"vocab\":"<<vocab<<",\"top_k\":"<<llama_model_dflash_selector_top_k(model.get())<<",\"prefix\":"<<q(prefix)<<",\"cache_dtype\":\"F16\",\"features\":\"synthetic_exact_dyadic_v1\",\"full_target_loaded\":false,\"compute_buffers\":{";
    bool first=true;for(const auto &entry:observed.names){if(!first)std::cout<<',';first=false;std::cout<<q(entry.first)<<':'<<q(entry.second);}std::cout<<"}}\n";return 0;
}
int main(int argc,char **argv){try{
    if(argc==5&&std::string(argv[1])=="--derive")return derive(argv[2],argv[3],argv[4]);
    if(argc==4&&std::string(argv[1])=="--inspect"){header t(argv[2]),d(argv[3]);auto all=copies(argv[2],argv[3],t,d);size_t bytes=0;for(const auto&x:all)bytes+=x.bytes;std::cout<<"{\"tensor_bytes\":"<<bytes<<",\"tensor_count\":"<<all.size()<<",\"width\":"<<num(d.meta,"dflash.embedding_length")<<",\"block\":"<<num(d.meta,"dflash.block_size")<<",\"window\":"<<num(d.meta,"dflash.attention.sliding_window")<<"}\n";return 0;}
    if(argc>1&&std::string(argv[1])=="--produce")return produce(argc,argv);
    throw std::runtime_error("use --inspect, --derive, or --produce");
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
