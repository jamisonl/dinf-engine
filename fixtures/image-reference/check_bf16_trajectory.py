#!/usr/bin/env python3
"""Fixed four-step BF16 denoiser check; old one-step checks stay separate."""
import argparse
import hashlib
import json
import stat
from pathlib import Path
import numpy as np

RUNTIMES = {'native': ('diffusers-0.42.0/torch-2.15.0', 'mps'), 'engine': ('dinf-engine', 'metal')}
MODES = ('denoise_uncached', 'denoise_cache')
SHAPES = {'sigmas': [5], 'timesteps': [4], 'input_latents': [1,1024,64], 'input_conditioning': [1,32,4096]}
SHAPES.update({f'{kind}_{i}': [1,1024,64] for i in range(4) for kind in ('prediction','latent')})

def need(value, message):
    if not value:
        raise ValueError(message)

def read_bounded(path, limit):
    path = Path(path)
    info = path.lstat()
    need(stat.S_ISREG(info.st_mode) and info.st_size <= limit, 'bounded regular file required')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    need(len(raw) <= limit, 'file exceeded size limit')
    return raw

def read_json(path, expected_sha256=None):
    raw = read_bounded(path, 1024 * 1024)
    if expected_sha256 is not None:
        need(hashlib.sha256(raw).hexdigest() == expected_sha256, 'fixture identity')
    return json.loads(raw)

def read_tensor(base, item, shape, bf16):
    name=item['file'];need(isinstance(name,str) and Path(name).name==name and name not in ('.','..'), 'raw basename')
    raw=read_bounded(base/name,4*int(np.prod(shape)))
    need(len(raw)==4*int(np.prod(shape)), 'raw size/type')
    need(item['shape']==shape and item.get('finite') is True, 'full shape/finite descriptor')
    need(hashlib.sha256(raw).hexdigest()==item['sha256'], 'raw hash')
    a=np.frombuffer(raw,dtype='<f4');need(np.isfinite(a).all(), 'nonfinite raw')
    if bf16:
        need(np.all((a.view('<u4')&0xffff)==0), 'non-BF16 value')
    return a

def load_fixture(path, gates):
    path=Path(path)
    f=read_json(path,gates['fixture_sha256']);need(f['schema_version']==1 and f['steps']==4, 'four-step fixture')
    expected={}
    for output,key in [('input_latents','latents'),('input_conditioning','conditioning')]:
        v=dict(f[key],finite=True);read_tensor(path.parent,v,SHAPES[output],True);expected[output]=v['sha256']
    return expected

def validate_gates(g):
    need(g['schema_version']==1 and g['declared_before_model_runs'] is True and g['steps']==4 and g['dtype']=='bf16', 'fixed trajectory profile')
    need(g['component']=={'relative_l2_max':.01,'cosine_min':.9999} and g['scheduler_max_abs']==1e-6, 'original component/scheduler limits')
    need(g['same_runtime_cache']=='byte_exact_all_outputs', 'exact cache gate')
    for name in ('sigmas', 'timesteps'):
        anchor = g['schedule_anchor'][name]
        need(anchor['shape'] == SHAPES[name], 'scheduler anchor shape')
        values = np.asarray(anchor['values'], dtype='f8')
        need(values.shape == tuple(SHAPES[name]) and np.isfinite(values).all(), 'scheduler anchor finite')
        raw = values.astype('<f4').tobytes()
        need(np.array_equal(values, np.frombuffer(raw, dtype='<f4')), 'scheduler anchor must contain exact F32 values')
        need(hashlib.sha256(raw).hexdigest() == anchor['origin_sha256'], 'scheduler anchor hash')

def load_run(base, role, mode, fixture_inputs, gates):
    base=Path(base);r=read_json(base/'result.json')
    need(r['schema_version']==1 and r['fixture_sha256']==gates['fixture_sha256'] and r['dtype']=='bf16', 'result profile/fixture')
    need((r['runtime'],r['backend'])==RUNTIMES[role] and r['mode']==mode, 'runtime/backend/mode')
    need(set(r['outputs'])==set(SHAPES), 'exact four-step output set')
    arrays={}
    for name,shape in SHAPES.items():
        item=r['outputs'][name]
        need(item['file']==name+'.f32', 'output file must match output name')
        arrays[name]=read_tensor(base,item,shape,name not in ('sigmas','timesteps'))
        if name in fixture_inputs:
            need(item['sha256']==fixture_inputs[name], 'input differs from fixed fixture')
        if name in ('sigmas','timesteps'):
            anchor=gates['schedule_anchor'][name];need(anchor['shape']==shape, 'scheduler anchor shape');a=np.asarray(anchor['values'],dtype='f8');need(a.shape==(shape[0],) and np.isfinite(a).all(), 'scheduler anchor finite')
            need(np.max(abs(arrays[name].astype('f8')-a))<=gates['scheduler_max_abs'], 'scheduler differs from original anchor')
    return r,arrays

def metrics(a,b):
    x=a.astype('f8');y=b.astype('f8');d=y-x;nx=np.linalg.norm(x);ny=np.linalg.norm(y)
    return {'relative_l2':float(np.linalg.norm(d)/nx) if nx else (0. if not np.any(d) else float('inf')),
            'cosine':float(np.dot(x,y)/(nx*ny)) if nx and ny else (1. if np.array_equal(x,y) else 0.),
            'max_abs':float(np.max(abs(d))), 'rms':float(np.sqrt(np.mean(d*d)))}

def compare_pair(left,right,gates,fixture,kind):
    validate_gates(gates);need(Path(left).resolve()!=Path(right).resolve(), 'distinct run directories')
    inputs=load_fixture(fixture,gates)
    if kind in MODES:
        roles=('native','engine');modes=(kind,kind);cache=False
    else:
        need(kind in RUNTIMES,'comparison kind');roles=(kind,kind);modes=MODES;cache=True
    r,x=load_run(left,roles[0],modes[0],inputs,gates);a,y=load_run(right,roles[1],modes[1],inputs,gates);rows={}
    for name in SHAPES:
        row=metrics(x[name],y[name]);exact=r['outputs'][name]['sha256']==a['outputs'][name]['sha256']
        if cache or name.startswith('input_'):ok=exact
        elif name in ('sigmas','timesteps'):ok=row['max_abs']<=gates['scheduler_max_abs']
        else:ok=row['relative_l2']<=gates['component']['relative_l2_max'] and row['cosine']>=gates['component']['cosine_min']
        rows[name]=dict(row,bit_exact=exact,passed=bool(ok),shape=SHAPES[name])
    return {'valid':True,'passed':all(v['passed'] for v in rows.values()),'kind':kind,'cache_comparison':cache,'rows':rows}

def compare_four(native_uncached,native_cached,engine_uncached,engine_cached,gates,fixture):
    paths=[Path(p) for p in [native_uncached,native_cached,engine_uncached,engine_cached]];need(len({p.resolve() for p in paths})==4,'four distinct runs')
    a,b,c,d=paths;checks={
        'uncached_component':compare_pair(a,c,gates,fixture,'denoise_uncached'),
        'cached_component':compare_pair(b,d,gates,fixture,'denoise_cache'),
        'native_cache':compare_pair(a,b,gates,fixture,'native'),
        'engine_cache':compare_pair(c,d,gates,fixture,'engine'),
    }
    return {'valid':True,'passed':all(v['passed']for v in checks.values()),'steps':4,'checks':checks,'scope':gates['scope']}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['native_uncached','native_cached','engine_uncached','engine_cached','gates','fixture','report']:p.add_argument(n,type=Path)
    a=p.parse_args();result={'valid':False,'passed':False}
    try:result=compare_four(a.native_uncached,a.native_cached,a.engine_uncached,a.engine_cached,read_json(a.gates),a.fixture)
    except Exception as e:result['error']=f'{type(e).__name__}: {e}'
    with a.report.open('x')as f:json.dump(result,f,indent=2)
    print(json.dumps(result));return 0 if result['passed']else 1

if __name__=='__main__':raise SystemExit(main())
