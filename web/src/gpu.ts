/** Generic execution of Python-generated plans. All ML semantics live in Python. */
export interface PlanNode {
  id: number; op: string; inputs: number[]; shape: number[]; dtype: string;
  attrs: Record<string, unknown>; name?: string; kind: string; provenance?: number[];
}
export interface GPUPlan {
  version: number; nodes: PlanNode[];
  kernels: {id:number;op:string;node_ids:number[];inputs:number[];output:number;wgsl:string;dispatch:[number,number,number]}[];
  buffers: {id:number;bytes:number;kind:string;dtype:string;nodes:number[]}[];
  tensor_buffers: Record<string,number>;
  bindings: Record<string,{node:number;buffer:number;shape:number[];dtype:string;kind:string}>;
  outputs: Record<string,number>; updates: Record<string,number>;
  label_specs: Record<string,number>; gpu_compatible:boolean;
  options: {max_allocation_bytes:number;[key:string]:unknown};
  cache_key:string; stats: Record<string,number>;
  [key:string]:unknown;
}
export type NumericBindings = Record<string, number[] | Float32Array | Int32Array | {data:number[];shape?:number[];dtype?:string}>;
export interface RunResult {
  values: Record<string, number[]>;
  timing: {uploadMs:number;executeMs:number;readbackMs:number;endToEndMs:number};
  resources: {ownedBytes:number;bufferCount:number;pipelineCount:number;stagingBytes:number};
}
interface SharedDevice {device:GPUDevice;refs:number;lost:string|null;ownedBytes:number;stagingBytes:number;capabilities:Record<string,unknown>}
const product = (shape:number[]) => shape.reduce((a,b)=>a*b,1);
const floats = (value:NumericBindings[string], dtype:string, length:number, name:string) => {
  const raw = Array.isArray(value) || ArrayBuffer.isView(value) ? value : value.data;
  if (raw.length !== length) throw new Error(`Binding ${name} has ${raw.length} elements; expected ${length}.`);
  for (const x of raw) if (!Number.isFinite(x) || (dtype==='int32' && (!Number.isInteger(x) || x < -2147483648 || x > 2147483647))) throw new Error(`Binding ${name} contains invalid ${dtype} values.`);
  const array = dtype==='int32' ? new Int32Array(raw) : new Float32Array(raw);
  for (const x of array) if (!Number.isFinite(x)) throw new Error(`Binding ${name} exceeds float32 range.`);
  return array;
};

export class GPUExecutor {
  private shared: SharedDevice;
  private plan:GPUPlan|null=null;
  private buffers = new Map<number,GPUBuffer>();
  private compiled:{pipeline:GPUComputePipeline;group:GPUBindGroup;dispatch:[number,number,number]}[]=[];
  private nodeById=new Map<number,PlanNode>();
  private disposed=false;
  private busy=false;
  private ownedBytes=0;
  private stagingBytes=0;
  public preparation = {pipelineMs:0,uploadMs:0,totalMs:0};
  private constructor(shared:SharedDevice) {this.shared=shared;shared.refs++;}
  static async create():Promise<GPUExecutor> {
    if (!navigator.gpu) throw new Error('WebGPU is unavailable. Select the NumPy CPU backend.');
    const adapter = await navigator.gpu.requestAdapter({powerPreference:'low-power'});
    if (!adapter) throw new Error('No WebGPU adapter is available. Select the NumPy CPU backend.');
    let device:GPUDevice;
    try {device=await adapter.requestDevice();} catch(error) {throw new Error(`WebGPU device request failed: ${String(error)}`);}
    const limits:Record<string,number>={};
    for (const key of ['maxBufferSize','maxStorageBufferBindingSize','maxStorageBuffersPerShaderStage','maxComputeWorkgroupsPerDimension','maxComputeInvocationsPerWorkgroup','maxComputeWorkgroupSizeX','maxComputeWorkgroupSizeY','maxComputeWorkgroupStorageSize']) limits[key]=Number((device.limits as unknown as Record<string,number>)[key]);
    const info=adapter.info;
    const shared:SharedDevice={device,refs:0,lost:null,ownedBytes:0,stagingBytes:0,capabilities:{limits,features:Array.from(device.features),adapter:info ? {vendor:info.vendor,architecture:info.architecture,device:info.device,description:info.description,isFallbackAdapter:(info as unknown as {isFallbackAdapter?:boolean}).isFallbackAdapter ?? (adapter as unknown as {isFallbackAdapter?:boolean}).isFallbackAdapter ?? null} : null,timingMethod:'synchronized wall clock',timestampQueriesUsed:false}};
    device.lost.then(event=>{shared.lost=`WebGPU device lost (${event.reason}): ${event.message || 'Reset the experiment or select NumPy CPU.'}`;});
    return new GPUExecutor(shared);
  }
  get capabilities() {return this.shared.capabilities;}
  get resourceStats() {return {ownedBytes:this.ownedBytes,bufferCount:this.buffers.size,pipelineCount:this.compiled.length,stagingBytes:this.stagingBytes,deviceOwnedBytes:this.shared.ownedBytes,deviceStagingBytes:this.shared.stagingBytes};}
  get isLost() {return this.shared.lost!==null;}
  spawn() {this.check();return new GPUExecutor(this.shared);}
  private check() {if(this.disposed) throw new Error('WebGPU executor is destroyed.');if(this.shared.lost) throw new Error(this.shared.lost);}
  private clearPlan() {for(const buffer of this.buffers.values()) buffer.destroy();this.buffers.clear();this.compiled=[];this.nodeById.clear();this.shared.ownedBytes-=this.ownedBytes;this.ownedBytes=0;this.plan=null;}
  async prepare(plan:GPUPlan,bindings:NumericBindings):Promise<void> {
    this.check();if(this.busy) throw new Error('Cannot prepare while executor is busy.');
    this.busy=true;this.clearPlan();const start=performance.now();
    try {
      if(plan.version!==1 || !plan.gpu_compatible) throw new Error('This plan is not a supported portable float32 WebGPU graph.');
      const limits=this.shared.device.limits;
      const total=plan.buffers.reduce((a,b)=>a+b.bytes,0);
      if(total>plan.options.max_allocation_bytes || this.shared.ownedBytes+this.shared.stagingBytes+total>128*1024*1024) throw new Error('Plan exceeds the Forge allocation cap.');
      if(limits.maxComputeInvocationsPerWorkgroup<64 || limits.maxComputeWorkgroupSizeX<64 || limits.maxComputeWorkgroupSizeY<8 || limits.maxComputeWorkgroupStorageSize<512) throw new Error('Adapter does not support this plan’s workgroup requirements.');
      for(const b of plan.buffers) {
        if(b.bytes>limits.maxBufferSize || b.bytes>limits.maxStorageBufferBindingSize || b.bytes<4 || b.bytes%4) throw new Error('Buffer exceeds device storage limits or alignment.');
      }
      for(const k of plan.kernels) {
        if(k.inputs.length+1>limits.maxStorageBuffersPerShaderStage) throw new Error('Fusion exceeds device storage binding limits; lower max_bindings.');
        if(k.dispatch.some(x=>x>limits.maxComputeWorkgroupsPerDimension || x<1)) throw new Error('Dispatch exceeds device workgroup limits.');
        if(k.inputs.some(id=>plan.tensor_buffers[id]===plan.tensor_buffers[k.output])) throw new Error('Unsafe dispatch storage overlap.');
      }
      this.plan=plan;this.nodeById=new Map(plan.nodes.map(n=>[n.id,n]));
      const device=this.shared.device;
      device.pushErrorScope('validation');
      let scopePopped=false;
      try {
        for(const b of plan.buffers) {
          const buffer=device.createBuffer({label:`Forge ${b.kind} ${b.id}`,size:b.bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
          this.buffers.set(b.id,buffer);this.ownedBytes+=b.bytes;this.shared.ownedBytes+=b.bytes;
        }
        const pipelineStart=performance.now();
        // Deliberately sequential: bounded driver compilation memory on 8 GB hosts.
        for(const kernel of plan.kernels) {
          const module=device.createShaderModule({label:`Forge kernel ${kernel.id}`,code:kernel.wgsl});
          const info=await module.getCompilationInfo();
          const errors=info.messages.filter(m=>m.type==='error');
          if(errors.length) throw new Error(`WGSL kernel ${kernel.id}: ${errors.map(m=>`${m.lineNum}:${m.linePos} ${m.message}`).join('; ')}`);
          const pipeline=await device.createComputePipelineAsync({label:`Forge kernel ${kernel.id}`,layout:'auto',compute:{module,entryPoint:'main'}});
          const ids=[...kernel.inputs,kernel.output];
          const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:ids.map((id,binding)=>({binding,resource:{buffer:this.bufferFor(id),size:product(this.nodeById.get(id)!.shape)*4}}))});
          this.compiled.push({pipeline,group,dispatch:kernel.dispatch});
        }
        this.preparation.pipelineMs=performance.now()-pipelineStart;
        const uploadStart=performance.now();
        for(const node of plan.nodes) if(node.op==='constant') {
          const values=floats(node.attrs.value as number[],node.dtype,product(node.shape),'constant');
          device.queue.writeBuffer(this.bufferFor(node.id),0,values);
        }
        for(const [name,spec] of Object.entries(plan.bindings)) {
          const value=bindings[name] ?? (spec.kind==='state' ? new Float32Array(product(spec.shape)) : undefined);
          if(value===undefined) throw new Error(`Missing initial binding ${name}.`);
          this.upload(name,value);
        }
        await device.queue.onSubmittedWorkDone();
        this.preparation.uploadMs=performance.now()-uploadStart;
        const error=await device.popErrorScope();scopePopped=true;if(error) throw new Error(`WebGPU validation: ${error.message}`);
      } finally {if(!scopePopped) await device.popErrorScope();}
      this.preparation.totalMs=performance.now()-start;this.check();
    } catch(error) {this.clearPlan();throw error;} finally {this.busy=false;}
  }
  private bufferFor(id:number):GPUBuffer {
    const buffer=this.buffers.get(this.plan!.tensor_buffers[id]);
    if(!buffer) throw new Error(`Plan has no buffer for tensor ${id}.`);
    return buffer;
  }
  private upload(name:string,value:NumericBindings[string]) {
    const spec=this.plan!.bindings[name];if(!spec) throw new Error(`Unknown binding ${name}.`);
    const data=floats(value,spec.dtype,product(spec.shape),name);
    const classes=this.plan!.label_specs[name];
    if(classes!==undefined) for(const label of data) if(label<0 || label>=classes) throw new Error(`Class labels in ${name} must lie in [0,${classes}).`);
    this.shared.device.queue.writeBuffer(this.buffers.get(spec.buffer)!,0,data);
  }
  async run(bindings:NumericBindings={},readNames:string[]=Object.keys(this.plan?.outputs??{}),commit=true):Promise<RunResult> {
    this.check();if(!this.plan) throw new Error('Prepare a plan before execution.');if(this.busy) throw new Error('Concurrent execution is not allowed.');
    this.busy=true;const start=performance.now();const device=this.shared.device;
    device.pushErrorScope('validation');let scopePopped=false;
    try {
      for(const [name,value] of Object.entries(bindings)) this.upload(name,value);
      await device.queue.onSubmittedWorkDone();const uploaded=performance.now();
      const encoder=device.createCommandEncoder({label:'Forge complete graph'});
      const pass=encoder.beginComputePass();
      for(const kernel of this.compiled) {pass.setPipeline(kernel.pipeline);pass.setBindGroup(0,kernel.group);pass.dispatchWorkgroups(...kernel.dispatch);}
      pass.end();
      device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();const executed=performance.now();
      const values=await this.readNodes(Object.fromEntries(readNames.map(name=>{
        const id=this.plan!.outputs[name]??this.plan!.updates[name];if(id===undefined) throw new Error(`Unknown output ${name}.`);return [name,id];
      })));
      const read=performance.now();
      // Read requested values before committing: an output can be an old parameter.
      // Rare direct-binding updates also need snapshots to support e.g. a <- b,
      // b <- a without sequential copy clobbering. Ordinary optimizer DAG outputs
      // already reside in separate, protected output buffers.
      if(commit && Object.keys(this.plan.updates).length) {
        const commitEncoder=device.createCommandEncoder({label:'Forge out-of-place commit'});
        const targets=new Set(Object.keys(this.plan.updates).map(name=>this.buffers.get(this.plan!.bindings[name].buffer)!));
        const snapshots=new Map<GPUBuffer,GPUBuffer>();let extraBytes=0;
        try {
          for(const id of Object.values(this.plan.updates)) {
            const source=this.bufferFor(id);
            if(targets.has(source) && !snapshots.has(source)) {
              const bytes=product(this.nodeById.get(id)!.shape)*4;extraBytes+=bytes;
              if(this.ownedBytes+extraBytes>this.plan.options.max_allocation_bytes || this.shared.ownedBytes+this.shared.stagingBytes+bytes>128*1024*1024) throw new Error('Commit snapshots exceed allocation cap.');
              const snapshot=device.createBuffer({size:bytes,usage:GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
              snapshots.set(source,snapshot);this.stagingBytes+=bytes;this.shared.stagingBytes+=bytes;commitEncoder.copyBufferToBuffer(source,0,snapshot,0,bytes);
            }
          }
          for(const [name,id] of Object.entries(this.plan.updates)) {
            const spec=this.plan.bindings[name];const source=this.bufferFor(id);
            commitEncoder.copyBufferToBuffer(snapshots.get(source)??source,0,this.buffers.get(spec.buffer)!,0,product(spec.shape)*4);
          }
          device.queue.submit([commitEncoder.finish()]);await device.queue.onSubmittedWorkDone();
        } finally {for(const buffer of snapshots.values()) buffer.destroy();const released=Array.from(snapshots.values()).reduce((sum,b)=>sum+b.size,0);this.stagingBytes-=released;this.shared.stagingBytes-=released;}
      }
      const finished=performance.now();
      const error=await device.popErrorScope();scopePopped=true;if(error) throw new Error(`WebGPU execution: ${error.message}`);this.check();
      return {values,timing:{uploadMs:uploaded-start,executeMs:executed-uploaded+finished-read,readbackMs:read-executed,endToEndMs:finished-start},resources:this.resourceStats};
    } finally {if(!scopePopped) await device.popErrorScope();this.busy=false;}
  }
  private async readNodes(names:Record<string,number>):Promise<Record<string,number[]>> {
    if(!Object.keys(names).length) return {};
    const device=this.shared.device;const entries=Object.entries(names);
    let offset=0;
    const locations=entries.map(([name,id])=>{const n=this.nodeById.get(id)!;const bytes=product(n.shape)*4;const here=offset;offset+=bytes;return {name,id,node:n,bytes,offset:here};});
    if(this.ownedBytes+offset>this.plan!.options.max_allocation_bytes || this.shared.ownedBytes+this.shared.stagingBytes+offset>128*1024*1024) throw new Error('Readback would exceed the Forge allocation cap; request fewer values.');
    const staging=device.createBuffer({label:'Forge readback staging',size:offset,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});this.stagingBytes+=offset;this.shared.stagingBytes+=offset;
    try {
      const encoder=device.createCommandEncoder();
      for(const l of locations) encoder.copyBufferToBuffer(this.bufferFor(l.id),0,staging,l.offset,l.bytes);
      device.queue.submit([encoder.finish()]);await staging.mapAsync(GPUMapMode.READ);
      const range=staging.getMappedRange();const result:Record<string,number[]>=Object.create(null) as Record<string,number[]>;
      for(const l of locations) {
        // Copy into JS arrays before unmapping invalidates mapped storage.
        result[l.name]=Array.from(l.node.dtype==='int32' ? new Int32Array(range,l.offset,l.bytes/4) : new Float32Array(range,l.offset,l.bytes/4));
        if(result[l.name].some(x=>!Number.isFinite(x))) throw new Error(`Nonfinite GPU result ${l.name}; check operation domains and learning rate.`);
      }
      staging.unmap();return result;
    } finally {staging.destroy();this.stagingBytes-=offset;this.shared.stagingBytes-=offset;}
  }
  async readBindings(names?:string[]):Promise<Record<string,number[]>> {
    this.check();if(!this.plan) throw new Error('No prepared plan.');if(this.busy) throw new Error('Read bindings only between completed steps.');
    const selected=names??Object.keys(this.plan.bindings).filter(n=>['parameter','state'].includes(this.plan!.bindings[n].kind));
    this.busy=true;
    try {return await this.readNodes(Object.fromEntries(selected.map(name=>{const spec=this.plan!.bindings[name];if(!spec) throw new Error(`Unknown binding ${name}.`);return [name,spec.node];})));} finally {this.busy=false;}
  }
  async copyBindingsFrom(other:GPUExecutor,names?:string[]):Promise<void> {
    this.check();other.check();if(this.shared!==other.shared) throw new Error('GPU-to-GPU binding copies require a shared device; use spawn().');
    if(!this.plan || !other.plan || this.busy || other.busy) throw new Error('Copy bindings between prepared, idle executors.');
    const selected=names??Object.keys(this.plan.bindings).filter(n=>this.plan!.bindings[n].kind!=='input' && other.plan!.bindings[n]);
    this.busy=true;other.busy=true;
    try {
      const encoder=this.shared.device.createCommandEncoder();
      for(const name of selected) {
        const dst=this.plan.bindings[name],src=other.plan.bindings[name];
        if(!dst || !src || dst.dtype!==src.dtype || JSON.stringify(dst.shape)!==JSON.stringify(src.shape)) throw new Error(`Binding mismatch for ${name}.`);
        encoder.copyBufferToBuffer(other.buffers.get(src.buffer)!,0,this.buffers.get(dst.buffer)!,0,product(dst.shape)*4);
      }
      this.shared.device.queue.submit([encoder.finish()]);await this.shared.device.queue.onSubmittedWorkDone();
    } finally {this.busy=false;other.busy=false;}
  }
  injectDeviceLoss() {this.check();this.shared.device.destroy();}
  destroy() {if(this.disposed) return;this.clearPlan();this.disposed=true;this.shared.refs--;if(this.shared.refs===0) this.shared.device.destroy();}
}
