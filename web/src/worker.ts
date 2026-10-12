/// <reference lib="webworker" />
import { GPUExecutor } from './gpu';
import type { GPUPlan, NumericBindings } from './gpu';
import type { WorkerRequest, WorkerResponse, Snapshot, ExperimentConfig, BenchmarkReport, BenchmarkRow, Capabilities } from './types';

type Pyodide = {loadPackage:(name:string)=>Promise<void>;unpackArchive:(data:ArrayBuffer,format:string)=>void;runPython:(code:string)=>unknown;globals:{set:(key:string,value:unknown)=>void;delete:(key:string)=>void}};
let python:Pyodide|undefined;
let booting:Promise<Capabilities>|undefined;
let base='';
let capabilities:Capabilities={webgpu:false};
let gpu:GPUExecutor|undefined;
let backend='cpu';
let gpuDisabled=false;
let trainReadNames:string[]=[];
let activeRun=0;
let chain=Promise.resolve();
const evalCache=new Map<string,{executor:GPUExecutor;spec:{plan:GPUPlan;bindings:NumericBindings;readNames:string[]}}>();

function py<T=unknown>(command:string,payload:unknown={}):T {
  if(!python)throw new Error('Python is not loaded. Initialize the runtime first.');
  python.globals.set('_forge_command',command);python.globals.set('_forge_payload',JSON.stringify(payload));
  try{return JSON.parse(python.runPython('_forge_call(_forge_command, _forge_payload)') as string) as T;}
  finally {python.globals.delete('_forge_command');python.globals.delete('_forge_payload');}
}
async function boot(url:string,disableGPU=false):Promise<Capabilities>{
  if(booting)return booting;
  base=url;
  gpuDisabled=disableGPU;
  booting=(async()=>{
    const runtimeUrl=new URL('pyodide/pyodide.mjs',base).href;
    const runtime=await import(/* @vite-ignore */ runtimeUrl);
    python=await runtime.loadPyodide({indexURL:new URL('pyodide/',base).href}) as Pyodide;
    await python.loadPackage('numpy');
    const response=await fetch(new URL('assets/forge.zip',base));if(!response.ok)throw new Error('Forge Python package is missing. Run scripts/prepare_assets.py.');
    python.unpackArchive(await response.arrayBuffer(),'zip');
    python.runPython('from forge_ml.bridge import call as _forge_call');
    const digitsResponse=await fetch(new URL('assets/digits.json',base));if(!digitsResponse.ok)throw new Error('Digits asset missing.');
    const versions=py<{python:string;numpy:string}>('assets',{digits:await digitsResponse.json()});
    capabilities={webgpu:false,pyodide:'0.27.7',...versions};
    // Acceptance can disable the entire GPU path without changing the browser.
    if(gpuDisabled){capabilities.reason='WebGPU explicitly disabled for CPU acceptance';return capabilities;}
    try{const probe=await GPUExecutor.create();const info=probe.capabilities.adapter as Record<string,unknown>|null;capabilities.webgpu=true;capabilities.adapter=info?`${info.vendor||''} ${info.architecture||''} ${info.description||''}${info.isFallbackAdapter?' (software adapter)':''}`.trim():'adapter details unavailable';probe.destroy();}
    catch(error){capabilities.reason=String(error);}
    return capabilities;
  })();
  return booting;
}
function clearGPU(){for(const v of evalCache.values())v.executor.destroy();evalCache.clear();gpu?.destroy();gpu=undefined;}
async function setup(result:{plan:GPUPlan;bindings:NumericBindings},selected:string){
  if(selected==='webgpu'&&gpuDisabled)throw new Error('WebGPU is disabled in this worker');
  clearGPU();backend=selected;
  trainReadNames=Object.keys(result.plan.outputs).filter(k=>k==='loss'||k==='gradient_norm'||k.startsWith('gradient_norm:')||k.startsWith('parameter_norm:'));
  if(backend==='webgpu'){gpu=await GPUExecutor.create();await gpu.prepare(result.plan,result.bindings);}
}
async function sync(){if(gpu)py('sync',{bindings:await gpu.readBindings()});}
async function evaluateGPU(kind:string,pixels?:number[]):Promise<Record<string,number[]>>{
  if(!gpu)throw new Error('GPU not initialized');
  const cacheKey=kind;
  let item=evalCache.get(cacheKey);
  if(!item || kind==='drawing'){
    const spec=py<{plan:GPUPlan;bindings:NumericBindings;readNames:string[]}>('eval-plan',{kind,pixels});
    if(!item || item.spec.plan.cache_key!==spec.plan.cache_key){
      item?.executor.destroy();const executor=gpu.spawn();await executor.prepare(spec.plan,spec.bindings);item={executor,spec};evalCache.set(cacheKey,item);
    }else item.spec=spec;
  }
  await item.executor.copyBindingsFrom(gpu);
  const inputs=Object.fromEntries(Object.entries(item.spec.bindings).filter(([k])=>item!.spec.plan.bindings[k]?.kind==='input'));
  const result=await item.executor.run(inputs,item.spec.readNames,false);
  return result.values;
}
async function snapshot():Promise<Snapshot>{
  if(!gpu)return py<Snapshot>('snapshot');
  const probabilities=(await evaluateGPU('display')).probabilities;
  const train=(await evaluateGPU('train')).loss[0];const validation=(await evaluateGPU('validation')).loss[0];
  const value=py<Snapshot>('snapshot',{probabilities,losses:{train,validation}});
  value.notes?.push('GPU parameters stay resident between steps. Display predictions and losses also execute on WebGPU. Tensor inspection explicitly runs a separate NumPy snapshot.');
  return value;
}
function median(a:number[]){const s=[...a].sort((a,b)=>a-b);return s[Math.floor(s.length/2)];}
async function benchmark(repetitions:number):Promise<BenchmarkReport>{
  if(!Number.isInteger(repetitions)||repetitions<3||repetitions>30)throw new Error('Use 3–30 benchmark repetitions');
  if(!gpu){const report=py<BenchmarkReport>('benchmark',{repetitions});report.rows.forEach(r=>r.backend='Pyodide NumPy');report.environment={...report.environment,userAgent:navigator.userAgent,backend:'Pyodide NumPy'};return report;}
  const cases=py<{name:string;configuration:string;plan:GPUPlan;bindings:NumericBindings;compilationMs:number;shapes:unknown}[]>('bench-plans');
  const rows:BenchmarkRow[]=[];
  for(const c of cases){
    const executor=gpu.spawn();
    try{
      await executor.prepare(c.plan,c.bindings);
      const reads=[Object.keys(c.plan.outputs)[0]];
      for(let i=0;i<3;i++)await executor.run({},reads);
      const samples=[];
      for(let i=0;i<repetitions;i++)samples.push((await executor.run({},reads)).timing);
      const t=samples.map(s=>s.endToEndMs),mean=t.reduce((a,b)=>a+b,0)/t.length;
      rows.push({name:c.name,backend:'WebGPU',configuration:c.configuration,medianMs:median(t),minMs:Math.min(...t),maxMs:Math.max(...t),stdMs:Math.sqrt(t.reduce((a,b)=>a+(b-mean)**2,0)/t.length),compilationMs:c.compilationMs+executor.preparation.pipelineMs,uploadMs:executor.preparation.uploadMs,executionMs:median(samples.map(s=>s.executeMs)),readbackMs:median(samples.map(s=>s.readbackMs)),kernels:c.plan.kernels.length,scratchBytes:c.plan.stats.scratch_bytes,repetitions,...{samples,shapes:c.shapes,dtype:'float32',compiler:c.plan.options}});
    }finally{executor.destroy();}
    await new Promise(resolve=>setTimeout(resolve,0));
  }
  return {rows,timingMethod:'Synchronized performance.now wall clock; queue.onSubmittedWorkDone plus mapped readback. 3 warmups. Resident data; initial upload reported separately.',environment:{userAgent:navigator.userAgent,...gpu.capabilities,backend:'WebGPU',resourceStats:gpu.resourceStats},timestamp:new Date().toISOString(),notes:['Compilation includes Python plan generation and GPU pipeline creation.','End-to-end samples include queue completion, update commit and one output readback. Initial model upload is reported separately.','Inspections are excluded. A lower dispatch or allocation count does not promise lower latency.']};
}
async function dispatch(message:WorkerRequest):Promise<unknown>{
  const payload=(message.payload||{}) as Record<string,unknown>;
  if(message.command==='boot')return boot(String(payload._baseUrl),payload.disableGPU===true);
  await boot(String(payload._baseUrl));
  if(message.command==='resources')return {training:gpu?.resourceStats??null,evaluation:[...evalCache.values()].map(v=>v.executor.resourceStats),cacheSize:evalCache.size};
  if(message.command==='validate')return validateFixtures(String(payload.backend??'cpu'));
  if(message.command==='init'){
    let checkpoint=payload.checkpoint;
    if(checkpoint==='digits'){const r=await fetch(new URL('assets/digits-checkpoint.json',base));if(!r.ok)throw new Error('Digits checkpoint unavailable');checkpoint=await r.json();}
    const config=payload.config as ExperimentConfig;
    const result=py<{plan:GPUPlan;bindings:NumericBindings}>('init',{config,checkpoint});
    await setup(result,config.backend);return snapshot();
  }
  if(message.command==='import'){
    const result=py<{config:ExperimentConfig;plan:GPUPlan;bindings:NumericBindings}>('import',payload);
    await setup(result,result.config.backend);return snapshot();
  }
  if(message.command==='step'){
    const steps=Number(payload.steps??1);if(!Number.isInteger(steps)||steps<1||steps>5)throw new Error('Worker groups contain 1–5 steps');
    if(!gpu)return py('step',{steps});
    for(let i=0;i<steps;i++){
      const batch=py<NumericBindings>('batch');
      const result=await gpu.run(batch,trainReadNames);
      if(!Number.isFinite(result.values.loss[0])||!Number.isFinite(result.values.gradient_norm[0]))throw new Error('Nonfinite GPU training result. Reset and lower learning rate.');
      const parameters=Object.fromEntries(Object.keys(result.values).filter(k=>k.startsWith('gradient_norm:')).map(k=>[k.slice(14),{gradientNorm:result.values[k][0],valueNorm:result.values['parameter_norm:'+k.slice(14)][0]}]));
      py('gpu-complete',{ms:result.timing.endToEndMs,gradientNorm:result.values.gradient_norm[0],parameters});
      await new Promise(resolve=>setTimeout(resolve,0));
      if(message.runId!==activeRun)throw new Error('Obsolete training group stopped');
    }
    return snapshot();
  }
  if(message.command==='snapshot')return snapshot();
  if(message.command==='export'||message.command==='inspect'){await sync();return py(message.command,payload);}
  if(message.command==='predict')return gpu?{probabilities:(await evaluateGPU('drawing',payload.pixels as number[])).probabilities}:py('predict',payload);
  if(message.command==='benchmark')return benchmark(Number(payload.repetitions??7));
  if(message.command==='loss-test'){if(!gpu)throw new Error('Choose WebGPU to inject device loss');gpu.injectDeviceLoss();await new Promise(r=>setTimeout(r,20));return {lost:gpu.isLost,message:'Device loss injected. Reset or select CPU; last completed exported checkpoint remains outside the worker.'};}
  throw new Error('Unknown command');
}
self.onmessage=(event:MessageEvent<WorkerRequest>)=>{
  const message=event.data;activeRun=Math.max(activeRun,message.runId);
  chain=chain.then(async()=>{
    const response:WorkerResponse={id:message.id,runId:message.runId,generation:message.generation};
    try{if(message.runId<activeRun)throw new Error('Obsolete request discarded');response.result=await dispatch(message);}
    catch(error){response.error=error instanceof Error?error.message:String(error);}
    self.postMessage(response);
  });
};

interface FixtureStep {values:Record<string,number[]>;bindings:Record<string,number[]>}
interface Fixture {name:string;plan:GPUPlan;bindings:NumericBindings;expected:FixtureStep[];rtol:number;atol:number}
async function validateFixtures(selected:string){
  if(selected==='webgpu'&&gpuDisabled)throw new Error('WebGPU is disabled in this worker');
  const response=await fetch(new URL('validation/gpu-fixtures.json',base));if(!response.ok)throw new Error('Generate validation fixtures first.');
  const cases=(await response.json()).fixtures as Fixture[];
  const result:Record<string,unknown>[]=[];
  let device:GPUExecutor|undefined;
  if(selected==='webgpu')device=await GPUExecutor.create();
  try{
    for(const c of cases){
      let actual:FixtureStep[]=[];let executor:GPUExecutor|undefined;
      try{
        if(device){
          executor=device.spawn();await executor.prepare(c.plan,c.bindings);
          for(let i=0;i<c.expected.length;i++){
            const run=await executor.run({},Object.keys({...c.plan.outputs,...c.plan.updates}));
            actual.push({values:run.values,bindings:await executor.readBindings()});
          }
        }else actual=py<FixtureStep[]>('validate-case',{case:c});
        let maxAbsolute=0,maxRelative=0,count=0;
        c.expected.forEach((expected,step)=>{
          for(const category of ['values','bindings'] as const){
            for(const [name,reference] of Object.entries(expected[category])){
              const a=actual[step][category][name];if(!a||a.length!==reference.length)throw new Error(`Missing/incorrect shape: ${name}`);
              reference.forEach((v,i)=>{const error=Math.abs(a[i]-v);maxAbsolute=Math.max(maxAbsolute,error);maxRelative=Math.max(maxRelative,error/Math.max(Math.abs(v),1e-12));count++;
                if(!Number.isFinite(a[i])||error>c.atol+c.rtol*Math.abs(v))throw new Error(`${name}[${i}] step ${step}: ${a[i]} vs ${v}, absolute ${error}`);
              });
            }
          }
        });
        result.push({name:c.name,pass:true,scalars:count,maxAbsolute,maxRelative,atol:c.atol,rtol:c.rtol});
      }catch(error){result.push({name:c.name,pass:false,error:String(error)});}
      finally{executor?.destroy();}
      await new Promise(resolve=>setTimeout(resolve,0));
    }
    return {backend:selected==='webgpu'?'WebGPU':'Pyodide NumPy',environment:{userAgent:navigator.userAgent,...(device?.capabilities??{}),sharedArrayBuffer:typeof SharedArrayBuffer!=='undefined',crossOriginIsolated:self.crossOriginIsolated},fixtures:result,pass:result.every(r=>r.pass),timestamp:new Date().toISOString()};
  }finally{device?.destroy();}
}
