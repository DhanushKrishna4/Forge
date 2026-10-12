import { ForgeClient } from './client';
import { DEFAULT_CONFIG } from './types';
import type { Snapshot,Capabilities,ExperimentConfig,BenchmarkReport } from './types';
const status=document.querySelector('#status')!;
const checks=document.querySelector('#checks')!;
const output=document.querySelector('#result')!;
let saved:unknown;
const buttons=[...document.querySelectorAll<HTMLButtonElement>('button[data-run]')];
const download=document.querySelector<HTMLButtonElement>('#download')!;
const assert=(ok:unknown,message:string)=>{if(!ok)throw new Error(message);};
async function run(backend:'cpu'|'webgpu'){
  buttons.forEach(b=>b.disabled=true);download.disabled=true;checks.innerHTML='';
  const client=new ForgeClient();let runId=0;
  const report:{backend:string;checks:{name:string;pass:boolean;details?:unknown;error?:string}[];[key:string]:unknown}={backend,checks:[],timestamp:new Date().toISOString(),url:location.href,userAgent:navigator.userAgent,crossOriginIsolated:window.crossOriginIsolated,sharedArrayBuffer:typeof SharedArrayBuffer!=='undefined'};
  const call=<T=unknown>(command:Parameters<ForgeClient['request']>[0],payload:unknown={})=>client.request<T>(command,payload,runId);
  async function check(name:string,fn:()=>Promise<unknown>){
    status.textContent=name;const li=document.createElement('li');li.textContent=name+' …';checks.append(li);
    try{const details=await fn();report.checks.push({name,pass:true,details});li.className='pass';li.textContent=name+' — pass';}
    catch(error){report.checks.push({name,pass:false,error:String(error)});li.className='fail';li.textContent=name+' — '+String(error);}
    output.textContent=JSON.stringify(report,null,2);
  }
  try{
    const caps=await call<Capabilities>('boot',{disableGPU:backend==='cpu'});report.capabilities=caps;
    if(backend==='cpu')assert(!caps.webgpu&&caps.reason?.includes('disabled'),'CPU acceptance must disable GPU access');
    if(backend==='webgpu'&&!caps.webgpu)throw new Error('WebGPU unavailable; GPU verification has NOT passed: '+caps.reason);
    await check('All operators, gradients, optimizer state and updates against native NumPy',async()=>{
      const r=await call<{pass:boolean;fixtures:unknown[]}>('validate',{backend});assert(r.pass,'Numerical fixture failed: '+JSON.stringify(r));return r;
    });
    let initial:Snapshot;let trained:Snapshot;let checkpoint:unknown;
    const config:ExperimentConfig={...DEFAULT_CONFIG,backend};
    await check('Initialize without training; real data and a decision boundary',async()=>{runId++;initial=await call<Snapshot>('init',{config});assert(initial.metrics.step===0,'Auto-training is forbidden');assert(initial.samples!.length>100&&initial.boundary!.probabilities.length===1600,'Missing real samples/predictions');return initial.metrics;});
    await check('Bounded training changes predictions and reduces loss',async()=>{
      for(let i=0;i<20;i++)trained=await call<Snapshot>('step',{steps:3});
      assert(trained!.metrics.step===60,'Step count');assert(trained!.metrics.trainLoss<initial!.metrics.trainLoss*.8,'Loss must drop >20% on fixture');
      assert(trained!.boundary!.probabilities.some((v,i)=>Math.abs(v-initial!.boundary!.probabilities[i])>.1),'Predictions unchanged');
      assert((trained!.metrics.gradientNorm??0)>0,'No actual gradients');return {initial:initial!.metrics,trained:trained!.metrics};
    });
    await check('Pause and single-step preserve bounded completion',async()=>{
      const a=await call<Snapshot>('snapshot');await new Promise(r=>setTimeout(r,100));const b=await call<Snapshot>('snapshot');assert(a.metrics.step===b.metrics.step,'Training continued after group ended');const c=await call<Snapshot>('step',{steps:1});assert(c.metrics.step===b.metrics.step+1,'Single-step failed');return c.metrics;
    });
    await check('Export/import resumes Adam state and RNG identically on same backend',async()=>{
      checkpoint=await call('export');const expected=await call<Snapshot>('step',{steps:1});const expectedExport=await call<{payload:{bindings:unknown}}>('export');
      runId++;await call('import',{experiment:checkpoint});const actual=await call<Snapshot>('step',{steps:1});const actualExport=await call<{payload:{bindings:unknown}}>('export');assert(expected.metrics.step===actual.metrics.step,'Resume counter');assert(JSON.stringify(expectedExport.payload.bindings)===JSON.stringify(actualExport.payload.bindings),'Resumed parameters or optimizer state differ');return {step:actual.metrics.step};
    });
    await check('Invalid and oversized experiment imports fail and preserve the session',async()=>{
      for(const experiment of [{version:999},'x'.repeat(8*1024*1024+1)]){let rejected=false;try{await call('import',{experiment});}catch{rejected=true;}assert(rejected,'Bad import was accepted');}
      const r=await call<Snapshot>('snapshot');assert(r.metrics.step>0,'Session lost after invalid import');return {rejected:2};
    });
    await check('Tensor and gradient inspection, then repeated reset without growing owned buffers',async()=>{
      const snap=await call<Snapshot>('snapshot');const data=await call<{values?:number[];gradient?:number[]}>('inspect',{nodeId:snap.graph.parameterIds![0]});assert(data.values?.length&&data.gradient?.length,'Missing inspection');
      const counts=[];
      for(let i=0;i<3;i++){runId++;await call('init',{config});await call('step',{steps:1});counts.push(await call('resources'));}
      assert(JSON.stringify(counts[0])===JSON.stringify(counts[2]),'Owned resources accumulated across resets');return counts;
    });
    await check('Late results cannot overwrite a newer run',async()=>{
      const old=call('step',{steps:5}).catch(()=>null);runId++;const next=await call<Snapshot>('init',{config});await old;assert(next.metrics.step===0,'New run contaminated');const now=await call<Snapshot>('snapshot');assert(now.metrics.step===0,'Stale group modified new run');return now.metrics.step;
    });
    await check('Compiler option changes rebuild the plan and preserve initial predictions',async()=>{
      runId++;const before=await call<Snapshot>('init',{config});
      const baseline={...config,compiler:{dce:false,fold:false,cse:false,fusion:false,reuse:false,tiled:false}};
      runId++;const after=await call<Snapshot>('init',{config:baseline});
      assert(before.plan.cache_key!==after.plan.cache_key,'Compiler options did not change cache key');
      assert(before.boundary!.probabilities.every((v,i)=>Math.abs(v-after.boundary!.probabilities[i])<1e-5),'Pass toggles changed predictions');
      return {optimized:before.plan.kernels.length,baseline:after.plan.kernels.length};
    });
    await check('Digits random initialization supports complete training',async()=>{
      runId++;const config2={...config,dataset:'digits',hidden:[64,32],activation:'relu',learningRate:.003,batchSize:64};
      const a=await call<Snapshot>('init',{config:config2});let b=a;for(let i=0;i<10;i++)b=await call<Snapshot>('step',{steps:3});assert(b.metrics.trainLoss<a.metrics.trainLoss,'Digits training not learning');return {initial:a.metrics,after:b.metrics};
    });
    await check('Forge-trained digits checkpoint, confusion matrix and drawing probabilities',async()=>{
      runId++;const r=await call<Snapshot>('init',{config:{...config,dataset:'digits'},checkpoint:'digits'});assert(r.digits!.testAccuracy!>.9,'Digits held-out accuracy below target');assert(r.digits!.confusion.flat().reduce((a,b)=>a+b,0)===266,'Validation split changed');const p=await call<{probabilities:number[]}>('predict',{pixels:r.digits!.examples[0].pixels});assert(Math.abs(p.probabilities.reduce((a,b)=>a+b,0)-1)<1e-4,'Probabilities not normalized');checkpoint=await call('export');return {testAccuracy:r.digits!.testAccuracy,step:r.metrics.step,prediction:p};
    });
    if(backend==='webgpu')await check('Injected device loss reports failure, then CPU recovery restores completed checkpoint',async()=>{
      await call('loss-test');let failed=false;try{await call('step',{steps:1});}catch{failed=true;}assert(failed,'Lost device still claims success');runId++;const restored=await call<Snapshot>('import',{experiment:checkpoint,backend:'cpu'});assert(restored.backend==='cpu'&&restored.metrics.step===310,'CPU recovery did not preserve checkpoint');return restored.metrics;
    });
    await check('Worker replacement discards old requests and restores host checkpoint',async()=>{
      client.restart();runId++;await call('boot',{disableGPU:true});const result=await call<Snapshot>('import',{experiment:checkpoint,backend:'cpu'});assert(result.metrics.step===310,'Checkpoint outside worker did not survive');return {generation:client.generation,step:result.metrics.step};
    });
    await check('Watchdog replaces a nonresponsive worker and retains the host checkpoint',async()=>{
      const impatient=new ForgeClient({requestTimeoutMs:1});impatient.lastCheckpoint=checkpoint;
      try{let message='';try{await impatient.request('boot',{disableGPU:true});}catch(error){message=String(error);}
        assert(message.includes('watchdog timed out')&&impatient.generation===1,'Watchdog did not reject and replace worker');assert(impatient.lastCheckpoint===checkpoint,'Host checkpoint lost');return {generation:impatient.generation,message};
      }finally{impatient.destroy();}
    });
  }catch(error){report.fatal=String(error);}
  finally{client.destroy();report.pass=!report.fatal&&report.checks.every(r=>r.pass);saved=report;output.textContent=JSON.stringify(report,null,2);status.textContent=report.pass?'All acceptance checks passed.':'Acceptance finished with failures; inspect the results.';buttons.forEach(b=>b.disabled=false);download.disabled=false;}
}
async function runBenchmark(backend:'cpu'|'webgpu'){
  buttons.forEach(b=>b.disabled=true);download.disabled=true;checks.innerHTML='';output.textContent='Measuring…';status.textContent='Running '+backend+' benchmarks: 3 warmups and 15 measured repetitions per configuration.';
  const client=new ForgeClient();
  try{await client.request('boot',{disableGPU:backend==='cpu'});await client.request('init',{config:{...DEFAULT_CONFIG,backend}});saved={...await client.request<BenchmarkReport>('benchmark',{repetitions:15}),url:location.href};output.textContent=JSON.stringify(saved,null,2);status.textContent='Benchmark measurements complete.';}
  catch(error){saved={error:String(error)};output.textContent=JSON.stringify(saved,null,2);status.textContent='Benchmark failed.';}
  finally{client.destroy();buttons.forEach(b=>b.disabled=false);download.disabled=false;}
}
document.querySelector<HTMLButtonElement>('#cpu')!.onclick=()=>void run('cpu');document.querySelector<HTMLButtonElement>('#gpu')!.onclick=()=>void run('webgpu');
document.querySelector<HTMLButtonElement>('#bench-cpu')!.onclick=()=>void runBenchmark('cpu');document.querySelector<HTMLButtonElement>('#bench-gpu')!.onclick=()=>void runBenchmark('webgpu');
download.onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(saved,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='forge-browser-validation.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
