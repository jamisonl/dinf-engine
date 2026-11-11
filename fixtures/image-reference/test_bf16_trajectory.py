import copy,hashlib,importlib.util,json,tempfile,time,sys,runpy
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parent;spec=importlib.util.spec_from_file_location('trajectory',B/'check_bf16_trajectory.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);g=json.loads((B/'bf16-trajectory-gates.json').read_text());started=time.monotonic();results=[]
with tempfile.TemporaryDirectory(prefix='bf16-trajectory-test-')as tmp:
    O=Path(tmp);F=O/'fixture';argv=sys.argv;sys.argv=['generate_inputs.py',str(F)]
    try:runpy.run_path(str(B/'generate_inputs.py'),run_name='__main__')
    finally:sys.argv=argv
    paths=[O/n for n in ['native_uncached','native_cached','engine_uncached','engine_cached']];reports=[];dirty={}
    for i,p in enumerate(paths):
        p.mkdir();outputs={}
        for n,shape in m.SHAPES.items():
            if n in ['sigmas','timesteps']:raw=np.asarray(g['schedule_anchor'][n]['values'],dtype='<f4').tobytes()
            elif n.startswith('input_'):raw=(F/('latents.f32'if n=='input_latents'else'conditioning.f32')).read_bytes()
            else:
                a=np.ones(shape,dtype='<f4');a.ravel()[0]=0;raw=a.tobytes()
            file=n+'.f32';(p/file).write_bytes(raw);outputs[n]={'file':file,'shape':shape,'sha256':hashlib.sha256(raw).hexdigest(),'finite':True}
        role='native'if i<2 else'engine';runtime,backend=m.RUNTIMES[role]
        reports.append({'schema_version':1,'runtime':runtime,'backend':backend,'dtype':'bf16','mode':m.MODES[i%2],'fixture_sha256':g['fixture_sha256'],'outputs':outputs})
    current=copy.deepcopy(reports)
    def reset():
        global current
        for p,b in dirty.items():p.write_bytes(b)
        dirty.clear();current=copy.deepcopy(reports)
        for p,r in zip(paths,current):(p/'result.json').write_text(json.dumps(r))
    def edit_report(i,fn):
        fn(current[i]);(paths[i]/'result.json').write_text(json.dumps(current[i]))
    def edit_raw(i,n,raw):
        p=paths[i]/(n+'.f32');dirty.setdefault(p,p.read_bytes());p.write_bytes(raw);edit_report(i,lambda r:r['outputs'][n].update(sha256=hashlib.sha256(raw).hexdigest()))
    def call(gates=None):return m.compare_four(*paths,g if gates is None else gates,F/'input.json')
    def test(name,mutate,invalid=False,check=None):
        reset();mutate()
        try:r=call()
        except (ValueError,KeyError,FileNotFoundError,TypeError):
            if not invalid:raise
        else:
            assert not invalid,name
            if check:check(r)
            else:assert r['valid']and not r['passed'],name
        results.append(name)
    def passed(r):assert r['valid']and r['passed']
    test('all four complete equal trajectories pass',lambda:None,check=passed)
    for label,fn in [
        ('missing final step',lambda r:r['outputs'].pop('latent_3')),
        ('extra fifth step',lambda r:r['outputs'].update(prediction_4=r['outputs']['prediction_3'])),
        ('wrong mode',lambda r:r.update(mode='denoise_single')),
        ('wrong runtime self comparison',lambda r:r.update(runtime='diffusers-0.42.0/torch-2.15.0',backend='mps')),
        ('wrong dtype',lambda r:r.update(dtype='f32')),
        ('wrong fixture',lambda r:r.update(fixture_sha256='0'*64)),
        ('wrong full shape',lambda r:r['outputs']['latent_3'].update(shape=[65536])),
        ('unsafe basename',lambda r:r['outputs']['latent_3'].update(file='../latent_3.f32')),
        ('same-shape output alias',lambda r:r['outputs']['latent_3'].update(file='prediction_0.f32')),
        ('swapped output descriptors',lambda r:r['outputs'].update(prediction_0=r['outputs']['latent_0'],latent_0=r['outputs']['prediction_0'])),
        ('wrong raw hash',lambda r:r['outputs']['latent_3'].update(sha256='0'*64)),
        ('false finite descriptor',lambda r:r['outputs']['latent_3'].update(finite=False)),
    ]:test(label,lambda fn=fn:edit_report(3,fn),invalid=True)
    def alias_all_outputs(r):
        for name in r['outputs']:
            if name.startswith(('prediction_', 'latent_')):
                r['outputs'][name]['file']='prediction_0.f32'
    test('all eight output keys cannot alias one raw tensor',lambda:edit_report(3,alias_all_outputs),invalid=True)
    raw=(paths[3]/'prediction_2.f32').read_bytes()
    test('matching hash short raw rejected',lambda:edit_raw(3,'prediction_2',raw[:-4]),invalid=True)
    test('matching hash NaN rejected',lambda:edit_raw(3,'prediction_2',np.full(65536,np.nan,dtype='<f4').tobytes()),invalid=True)
    test('matching hash nonBF16 value rejected',lambda:edit_raw(3,'prediction_2',np.full(65536,np.float32(1.0001),dtype='<f4').tobytes()),invalid=True)
    def wrong_input():
        b=np.zeros(65536,dtype='<f4').tobytes()
        for i in range(4):edit_raw(i,'input_latents',b)
    test('shared but wrong input rejected against fixture',wrong_input,invalid=True)
    def wrong_schedule():
        for i in range(4):edit_raw(i,'timesteps',(np.asarray(g['schedule_anchor']['timesteps']['values'],dtype='<f4')+np.float32(.01)).tobytes())
    test('common scheduler drift rejected against anchor',wrong_schedule,invalid=True)
    def bad_final():
        b=np.full(65536,2,dtype='<f4').tobytes()
        for i in [2,3]:edit_raw(i,'prediction_3',b)
    def final_check(r):
        assert r['valid']and not r['passed'];assert r['checks']['engine_cache']['passed'];assert not r['checks']['uncached_component']['rows']['prediction_3']['passed']
    test('late finite prediction failure kept',bad_final,check=final_check)
    def one_cache_word():
        a=np.frombuffer(raw,dtype='<f4').copy();a[123]=np.float32(1.0078125);edit_raw(3,'prediction_2',a.tobytes())
    def cache_check(r):
        assert r['valid']and not r['passed'];assert r['checks']['cached_component']['passed'];assert not r['checks']['engine_cache']['rows']['prediction_2']['passed']
    test('cache oneBF16word fails despite passing component limits',one_cache_word,check=cache_check)
    def signed_zero():
        a=np.frombuffer(raw,dtype='<f4').copy();a[0]=np.float32(-0.);edit_raw(1,'prediction_2',a.tobytes())
    def zero_check(r):
        z=r['checks']['native_cache']['rows']['prediction_2'];assert r['valid']and not r['passed']and z['relative_l2']==0 and not z['bit_exact']
    test('cache signed zero requires exact bytes',signed_zero,check=zero_check)
    reset()
    link=paths[3]/'prediction_1.f32'; original=link.read_bytes();link.unlink();link.symlink_to(paths[2]/'prediction_1.f32')
    try:call()
    except ValueError:results.append('symlink raw rejected')
    else:raise AssertionError('symlink accepted')
    finally:link.unlink();link.write_bytes(original)
    reset()
    try:m.compare_four(paths[0],paths[0],paths[2],paths[3],g,F/'input.json')
    except ValueError:results.append('same directory cannot qualify cache')
    else:raise AssertionError('same directory accepted')
    for key,value in [('steps',40),('component',{'relative_l2_max':.02,'cosine_min':.999}),('same_runtime_cache','tolerance')]:
        bad=copy.deepcopy(g);bad[key]=value
        try:call(bad)
        except ValueError:results.append('weakened '+key+' rejected')
        else:raise AssertionError(key)
    for name in ('sigmas','timesteps'):
        for kind in ('changed_values','wrong_origin_hash','rounded_anchor_values'):
            bad=copy.deepcopy(g);anchor=bad['schedule_anchor'][name]
            if kind=='changed_values':anchor['values'][0]=float(np.float32(anchor['values'][0]+1))
            elif kind=='wrong_origin_hash':anchor['origin_sha256']='0'*64
            else:anchor['values'][0]+=1e-10
            try:call(bad)
            except ValueError:results.append(name+' '+kind+' rejected')
            else:raise AssertionError(name+' '+kind)
    large=O/'oversized-input.json'
    with large.open('wb') as stream:stream.truncate(1024*1024+1)
    linked=O/'linked-input.json';linked.symlink_to(F/'input.json')
    for name,path in [('oversized',large),('symlink',linked),('directory',F)]:
        # Invalid size/type must fail before opening, hashing, or parsing the file.
        with patch.object(Path,'open',side_effect=AssertionError('invalid fixture was opened')),patch.object(m.hashlib,'sha256',side_effect=AssertionError('invalid fixture was hashed')):
            try:m.load_fixture(path,g)
            except ValueError:results.append(name+' fixture rejected before open/hash')
            else:raise AssertionError(name+' fixture accepted')
print(json.dumps({'passed':True,'cases':results,'case_count':len(results),'seconds':time.monotonic()-started,'model_calls':0},indent=2))
