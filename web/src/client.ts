import type { Command, WorkerRequest, WorkerResponse } from './types';

/** Owns worker lifetime; run + generation checks keep obsolete results out of UI. */
export class ForgeClient {
  generation = 0;
  private worker!: Worker;
  private sequence = 0;
  private pending = new Map<number,{resolve:(v:unknown)=>void;reject:(e:Error)=>void;timer:ReturnType<typeof setTimeout>;runId:number;generation:number}>();
  lastCheckpoint: unknown = null;
  private makeWorker() {
    this.worker=new Worker(new URL('./worker.ts',import.meta.url),{type:'module'});
    this.worker.onmessage=(event:MessageEvent<WorkerResponse>)=>{
      const m=event.data,p=this.pending.get(m.id);
      if(!p || m.generation!==this.generation || p.generation!==m.generation || p.runId!==m.runId)return;
      clearTimeout(p.timer);this.pending.delete(m.id);
      if(m.error)p.reject(new Error(m.error));else {if(m.result && m.id && typeof m.result==='object' && (m.result as {format?:string}).format==='forge-experiment')this.lastCheckpoint=m.result;p.resolve(m.result);}
    };
    this.worker.onerror=(event)=>this.rejectAll(new Error(event.message||'Python worker failed. Use Recover worker to restore your last checkpoint.'));
  }
  constructor(private options:{requestTimeoutMs?:number}={}){this.makeWorker();}
  request<T=unknown>(command:Command,payload:unknown={},runId=0):Promise<T>{
    const id=++this.sequence,generation=this.generation;
    return new Promise<T>((resolve,reject)=>{
      const timer=setTimeout(()=>this.restart(new Error('Worker watchdog timed out and replaced the worker. Submitted GPU work may have completed. Restore the last completed checkpoint or reset.')),this.options.requestTimeoutMs??(command==='boot'||command==='benchmark'?180000:60000));
      this.pending.set(id,{resolve:v=>resolve(v as T),reject,timer,runId,generation});
      const message:WorkerRequest={id,command,payload:{...(payload as object),_baseUrl:new URL(import.meta.env.BASE_URL,location.origin).href},runId,generation};
      this.worker.postMessage(message);
    });
  }
  private rejectAll(error:Error){for(const p of this.pending.values()){clearTimeout(p.timer);p.reject(error);}this.pending.clear();}
  restart(reason=new Error('Worker replaced; obsolete requests discarded.')){this.worker.terminate();this.rejectAll(reason);this.generation++;this.makeWorker();}
  destroy(){this.worker.terminate();this.rejectAll(new Error('Client closed.'));}
}
