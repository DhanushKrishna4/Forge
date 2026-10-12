"""Inspectable conservative graph compiler and whole-training-step memory planner.

Fusion never crosses a reduction/matmul or observable value. Scratch reuse requires
last_use < birth (strict): an input can never alias an output in its dispatch.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import defaultdict

import numpy as np

from .wgsl import LEAVES, OPS, POINTWISE, emit_kernel

DEFAULT_OPTIONS = dict(dce=True, fold=True, cse=True, fusion=True, reuse=True, tiled=True,
                       max_fusion=8, max_bindings=8, instrument=False, max_allocation_bytes=128*1024*1024, max_kernel_work=64*1024*1024)


def _ident(value):
    ident = value.id if hasattr(value,'id') else value
    if not isinstance(ident,(int,np.integer)) or isinstance(ident,(bool,np.bool_)): raise ValueError('Output references must be integer node ids or Tensors')
    return int(ident)


def _apply(node, inputs):
    from .interpreter import apply_op
    return apply_op(node['op'], inputs, node['attrs'], node['dtype'], tuple(node['shape']))


def _validate(nodes):
    if len(nodes) > 10000:
        raise ValueError('Compiler graph exceeds 10000 nodes')
    names = set()
    for i, n in enumerate(nodes):
        if n['id'] != i or any(not isinstance(j,int) or j < 0 or j >= i for j in n['inputs']):
            raise ValueError('Graph must be acyclic, topologically ordered with stable contiguous ids')
        if n['op'] not in OPS: raise ValueError(f'Unsupported compiler operation: {n["op"]}')
        if n['dtype'] not in {'float32','float64','int32'}: raise ValueError('Unsupported dtype')
        if len(n['shape']) > 2 or any(not isinstance(x,int) or x <= 0 for x in n['shape']):
            raise ValueError('Only positive static rank 0-2 shapes are supported')
        if math.prod(n['shape']) > 1_048_576: raise ValueError('Tensor exceeds element limit')
        if n['op'] in {'input','parameter','state'}:
            if not n.get('name') or n['name'] in names: raise ValueError('Binding names must be unique')
            names.add(n['name'])
        _validate_operation(n, nodes)
        if n['op'] == 'constant':
            value = np.asarray(n['attrs']['value'])
            if value.size != math.prod(n['shape']) or not np.isfinite(value).all():
                raise ValueError('Invalid constant storage')


def _validate_operation(n, nodes):
    """Infer shape/type independently of recorded metadata, without tensor allocation."""
    op=n['op']; inputs=[nodes[j] for j in n['inputs']]; attrs=n['attrs']; shape=tuple(n['shape'])
    arity = 0 if op in LEAVES else 2 if op in {'add','sub','mul','div','greater','matmul','cross_entropy'} else 3 if op=='where' else 1
    if len(inputs)!=arity: raise ValueError(f'Invalid arity for {op}')
    if op in LEAVES: return
    if n['dtype'] not in {'float32','float64'} and op not in {'reshape','transpose','broadcast_to'}: raise ValueError('Arithmetic outputs must be floating point')
    if op=='one_hot':
        classes=attrs.get('classes')
        if inputs[0]['dtype']!='int32' or len(inputs[0]['shape'])!=1 or not isinstance(classes,int) or classes<1: raise ValueError('Invalid one_hot signature')
        expected=(inputs[0]['shape'][0],classes)
    elif op=='cross_entropy':
        logits,labels=inputs
        if logits['dtype']!=n['dtype'] or labels['dtype']!='int32' or len(logits['shape'])!=2 or labels['shape']!=[logits['shape'][0]]: raise ValueError('Invalid cross_entropy signature')
        expected=()
    else:
        if any(a['dtype']!=n['dtype'] for a in inputs): raise ValueError(f'Mixed arithmetic dtypes in {op}')
        shapes=[tuple(a['shape']) for a in inputs]
        if op in {'add','sub','mul','div','greater','where'}: expected=np.broadcast_shapes(*shapes)
        elif op=='matmul':
            if any(len(a)!=2 for a in shapes) or shapes[0][1]!=shapes[1][0]: raise ValueError('Invalid matmul shapes')
            expected=(shapes[0][0],shapes[1][1])
        elif op in {'sum','mean'}:
            axes=attrs.get('axis'); keep=attrs.get('keepdims')
            if not isinstance(axes,(list,tuple)) or not isinstance(keep,bool) or len(set(axes))!=len(axes) or any(not isinstance(a,int) or a<0 or a>=len(shapes[0]) for a in axes): raise ValueError('Invalid normalized reduction axes')
            expected=tuple(1 if a in axes else d for a,d in enumerate(shapes[0])) if keep else tuple(d for a,d in enumerate(shapes[0]) if a not in axes)
        elif op=='reshape':
            expected=tuple(attrs.get('shape',()))
            if math.prod(expected)!=math.prod(shapes[0]): raise ValueError('Reshape changes element count')
        elif op=='transpose':
            if len(shapes[0])!=2 or tuple(attrs.get('axes',()))!=(1,0): raise ValueError('Only matrix transpose is supported')
            expected=shapes[0][::-1]
        elif op=='broadcast_to':
            expected=tuple(attrs.get('shape',()))
            if np.broadcast_shapes(shapes[0],expected)!=expected: raise ValueError('Invalid broadcast target')
        else:
            expected=shapes[0]
            if op=='softmax' and (not isinstance(attrs.get('axis'),int) or not 0<=attrs['axis']<len(expected)): raise ValueError('Invalid softmax axis')
    if shape!=expected: raise ValueError(f'Incorrect recorded shape for {op}: {shape}, expected {expected}')


def compile_graph(graph, outputs, options=None, updates=None):
    """Compile one immutable graph. Weights are bindings, never part of the key.

    `outputs` includes every value that needs to remain observable. `updates`
    maps persistent binding names to newly computed values, committed afterward.
    """
    options = {**DEFAULT_OPTIONS, **(options or {})}
    if set(options) - set(DEFAULT_OPTIONS): raise ValueError('Unknown compiler option')
    if any(not isinstance(options[name],bool) for name in ('dce','fold','cse','fusion','reuse','tiled','instrument')): raise ValueError('Pass options must be boolean')
    if any(not isinstance(options[name],int) or isinstance(options[name],bool) for name in ('max_fusion','max_bindings','max_allocation_bytes','max_kernel_work')): raise ValueError('Resource limits must be integers')
    if not 1 <= options['max_fusion'] <= 32: raise ValueError('max_fusion must be in [1,32]')
    if not 4 <= options['max_bindings'] <= 16: raise ValueError('max_bindings must be in [4,16]')
    if not 1024 <= options['max_allocation_bytes'] <= 128*1024*1024: raise ValueError('Invalid allocation cap')
    if not isinstance(options['max_kernel_work'],int) or not 1 <= options['max_kernel_work'] <= 64*1024*1024: raise ValueError('Invalid per-kernel work cap')
    source = copy.deepcopy(graph.to_dict()['nodes'])
    _validate(source)
    source_original = copy.deepcopy(source)
    for name,value in [*outputs.items(),*(updates or {}).items()]:
        if not isinstance(name,str) or not name or len(name)>200: raise ValueError('Output/update names must be nonempty strings of at most 200 characters')
        if hasattr(value,'graph') and value.graph is not graph: raise ValueError('Output Tensor belongs to a different graph')
    outputs = {k:_ident(v) for k,v in outputs.items()}
    updates = {k:_ident(v) for k,v in (updates or {}).items()}
    if set(outputs)&set(updates): raise ValueError('Output and update names must be distinct')
    if not outputs and not updates: raise ValueError('At least one output or update is required')
    if any(i < 0 or i >= len(source) for i in [*outputs.values(), *updates.values()]):
        raise ValueError('Unknown output id')
    original_roots = set(outputs.values()) | set(updates.values())
    if options['instrument']:
        outputs.update({f'tensor:{n["id"]}':n['id'] for n in source})
    roots = set(outputs.values()) | set(updates.values())
    needed = set(roots)
    for n in reversed(source):
        if n['id'] in needed: needed.update(n['inputs'])
    active = [n for n in source if not options['dce'] or n['id'] in needed]
    # Targets remain persistent even in the degenerate update p <- constant case.
    update_targets = {n['name']:n for n in source if n['op'] in {'parameter','state'}}
    for name, ident in updates.items():
        if name not in update_targets: raise ValueError(f'Update target {name} is not a parameter/state')
        target = update_targets[name]
        if source[ident]['shape'] != target['shape'] or source[ident]['dtype'] != target['dtype']:
            raise ValueError(f'Update shape/dtype mismatch for {name}')
        if all(n['id'] != target['id'] for n in active): active.append(target)
    active.sort(key=lambda n:n['id'])
    passes = [dict(name='validation', enabled=True,before=len(source),after=len(source),removed=[],detail='Typed static DAG; supported operations and bounded shapes.'),
              dict(name='dce',enabled=options['dce'],before=len(source),after=len(active),removed=[n['id'] for n in source if n not in active],detail='Keeps requested outputs, gradients, updates, and their dependencies.')]
    folded = []
    constant_bytes = sum(math.prod(n['shape'])*np.dtype(n['dtype']).itemsize for n in active if n['op']=='constant')
    if constant_bytes>options['max_allocation_bytes']: raise ValueError('Immutable constants exceed allocation cap before compilation')
    constants = {}
    for node in active:
        if node['op'] == 'constant':
            constants[node['id']] = np.asarray(node['attrs']['value'],dtype=node['dtype']).reshape(node['shape'])
        elif options['fold'] and node['inputs'] and all(j in constants for j in node['inputs']):
            bytes_needed = math.prod(node['shape'])*np.dtype(node['dtype']).itemsize
            if constant_bytes+bytes_needed>options['max_allocation_bytes']: raise ValueError('Constant folding would exceed allocation cap')
            value = _apply(node,[constants[j] for j in node['inputs']])
            # Nonfinite constants violate serialization and domain validation.
            if np.isfinite(value).all():
                constants[node['id']] = value
                constant_bytes += bytes_needed
                node.update(op='constant',inputs=[],attrs={'value':value.reshape(-1).tolist()},kind='constant')
                folded.append(node['id'])
    passes.append(dict(name='fold',enabled=options['fold'],before=len(active),after=len(active),removed=folded,detail=f'{len(folded)} immutable expressions replaced with constants; bindings are never folded.'))
    aliases = {}; seen = {}; retained = []
    for node in active:
        node['inputs'] = [aliases.get(j,j) for j in node['inputs']]
        key = json.dumps([node['op'],node['inputs'],node['attrs'],node['shape'],node['dtype']],sort_keys=True,separators=(',',':'))
        if options['cse'] and node['op'] not in {'input','parameter','state'} and key in seen:
            aliases[node['id']] = seen[key]
        else:
            seen[key] = node['id']; retained.append(node)
    passes.append(dict(name='cse',enabled=options['cse'],before=len(active),after=len(retained),removed=list(aliases),detail='Exact operation, ordered inputs, attributes, shape and dtype; no algebraic rewrites.'))
    nodes = {n['id']:n for n in retained}
    outputs = {k:aliases.get(v,v) for k,v in outputs.items()}
    updates = {k:aliases.get(v,v) for k,v in updates.items()}
    protected = set(outputs.values()) | set(updates.values())
    consumers = defaultdict(set)
    for n in retained:
        for j in n['inputs']: consumers[j].add(n['id'])
    groups = {n['id']:[n['id']] for n in retained if n['op'] not in LEAVES}
    owner = {i:i for i in groups}
    if options['fusion'] and not options['instrument']:
        for n in retained:
            ident = n['id']
            if ident not in groups or n['op'] not in POINTWISE: continue
            for j in n['inputs']:
                if j not in groups or j in protected or len(consumers[j]) != 1: continue
                members = groups[ident] + groups[j]
                if nodes[j]['op'] not in POINTWISE or nodes[j]['shape'] != n['shape']: continue
                external = {p for x in members for p in nodes[x]['inputs'] if p not in members}
                if len(members) > options['max_fusion'] or len(external) + 1 > options['max_bindings']: continue
                groups[ident] = sorted(members)
                del groups[j]
                for x in members: owner[x] = ident
    ordered = sorted(groups.items())
    kernels = []
    for root, members in ordered:
        external = sorted({j for i in members for j in nodes[i]['inputs'] if j not in members})
        if len(external)+1 > options['max_bindings']: raise ValueError('Kernel exceeds configured storage binding limit')
        work = math.prod(nodes[root]['shape']) * len(members)
        if nodes[root]['op']=='matmul': work *= nodes[nodes[root]['inputs'][0]]['shape'][1]
        elif nodes[root]['op'] in {'sum','mean','cross_entropy'}: work = math.prod(nodes[nodes[root]['inputs'][0]]['shape']) * (2 if nodes[root]['op']=='cross_entropy' else 1)
        elif nodes[root]['op']=='softmax': work *= nodes[root]['shape'][nodes[root]['attrs']['axis']] * 2
        if work > options['max_kernel_work']: raise ValueError(f'Kernel {root} exceeds work cap ({work} > {options["max_kernel_work"]})')
        shader, dispatch = emit_kernel(nodes,members,external,options['tiled'])
        kernels.append(dict(id=len(kernels),op='fused' if len(members)>1 else nodes[root]['op'],node_ids=members,inputs=external,output=root,wgsl=shader,dispatch=dispatch,estimated_scalar_work=work))
    unfused = sum(n['op'] not in LEAVES for n in retained)
    passes.append(dict(name='fusion',enabled=options['fusion'] and not options['instrument'],before=unfused,after=len(kernels),removed=[i for i in owner if owner[i]!=i],detail='GPU only; same-shape single-consumer pointwise regions, bounded operations and bindings. Instrumentation disables fusion.'))
    passes.append(dict(name='schedule',enabled=True,before=len(kernels),after=len(kernels),removed=[],detail='Topological kernel order over the entire forward, backward, and update DAG.'))
    end = len(kernels)
    births = {n['id']:-1 for n in retained if n['op'] in LEAVES}
    last = dict(births)
    for k in kernels:
        births[k['output']] = k['id']; last[k['output']] = k['id']
        for j in k['inputs']: last[j] = max(last.get(j,-1),k['id'])
    for j in protected: last[j] = end
    buffers = []; mapping = {}; lifetimes = []; free = []; allocations = {}
    def nbytes(n): return math.prod(n['shape']) * np.dtype(n['dtype']).itemsize
    for ident in sorted(births,key=lambda i:(births[i],i)):
        node = nodes[ident]; birth = births[ident]; death = last[ident]; num = nbytes(node)
        kind = node['op'] if node['op'] in LEAVES else ('inspection' if options['instrument'] and ident not in original_roots else 'output' if ident in protected else 'scratch')
        if kind == 'scratch' and options['reuse']:
            candidates = [b for b in buffers if b['kind']=='scratch' and allocations[b['id']] < birth and b['bytes'] >= num and b['dtype']==node['dtype']]
        else: candidates = []
        if candidates:
            buf = min(candidates,key=lambda b:b['bytes'])
        else:
            buf = dict(id=len(buffers),bytes=max(4,num),kind=kind,dtype=node['dtype'],nodes=[])
            buffers.append(buf)
        buf['nodes'].append(ident); allocations[buf['id']] = death; mapping[str(ident)] = buf['id']
        lifetimes.append(dict(node=ident,buffer=buf['id'],start=birth,end=death,bytes=num,capacity=buf['bytes'],kind=kind))
    total = sum(b['bytes'] for b in buffers)
    if total > options['max_allocation_bytes']: raise ValueError(f'GPU allocation plan {total} exceeds cap')
    scratch = sum(b['bytes'] for b in buffers if b['kind']=='scratch')
    unreused = sum(l['bytes'] for l in lifetimes if l['kind']=='scratch')
    peak = max((sum(l['bytes'] for l in lifetimes if l['start']<=t<=l['end']) for t in range(-1,end+1)),default=0)
    passes.append(dict(name='reuse',enabled=options['reuse'],before=unreused,after=scratch,removed=[],detail='Scratch bytes; reuse only when previous last use is strictly before new dispatch. Persistent/input/output buffers stay separate.'))
    bindings = {n['name']:dict(node=n['id'],buffer=mapping[str(n['id'])],shape=n['shape'],dtype=n['dtype'],kind=n['op']) for n in retained if n['op'] in {'input','parameter','state'}}
    kernel_for = {i:k['id'] for k in kernels for i in k['node_ids']}
    source_map = {}
    for n in source:
        ident = aliases.get(n['id'],n['id'])
        status = 'dead' if ident not in nodes else 'cse' if n['id'] in aliases else 'folded' if ident in folded else 'fused' if ident in owner and owner[ident]!=ident else 'materialized'
        source_map[str(n['id'])] = dict(node=ident,kernel=kernel_for.get(ident),status=status)
    label_specs = {}
    for n in retained:
        if n['op'] in {'cross_entropy','one_hot'}:
            label = n['inputs'][1 if n['op']=='cross_entropy' else 0]
            classes = nodes[n['inputs'][0]]['shape'][1] if n['op']=='cross_entropy' else n['attrs']['classes']
            pending = [label]
            while pending:
                label_node = nodes[pending.pop()]
                if label_node['op']=='constant':
                    value = np.asarray(label_node['attrs']['value'])
                    if np.any(value<0) or np.any(value>=classes): raise ValueError('Constant class labels out of range')
                elif label_node.get('name'):
                    name = label_node['name']; label_specs[name] = min(classes,label_specs.get(name,classes))
                else: pending.extend(label_node['inputs'])
    key_data = dict(nodes=retained,outputs=outputs,updates=updates,options=options)
    for index, p in enumerate(passes):
        p['unit'] = 'bytes' if p['name']=='reuse' else 'kernels' if p['name'] in {'fusion','schedule'} else 'nodes'
        p['changed'] = list(p['removed'])
        if p['name']=='fold': p['removed'] = []
        previous = passes[index-1]['after_nodes'] if index else [n['id'] for n in source_original]
        p['before_nodes'] = list(previous)
        removed = set(p['removed']) if p['name'] in {'dce','cse','fusion'} else set()
        p['after_nodes'] = [i for i in previous if i not in removed]
    plan = dict(version=1,source_nodes=source_original,nodes=retained,kernels=kernels,buffers=buffers,tensor_buffers=mapping,bindings=bindings,outputs=outputs,updates=updates,passes=passes,lifetimes=lifetimes,source_map=source_map,options=options,label_specs=label_specs,
                gpu_compatible=all(n['dtype'] in {'float32','int32'} for n in retained),
                stats=dict(source_nodes=len(source),optimized_nodes=len(retained),kernel_count=len(kernels),fused_nodes=unfused-len(kernels),source_logical_bytes=sum(nbytes(n) for n in source_original),logical_bytes=sum(nbytes(n) for n in retained),materialized_bytes=sum(l['bytes'] for l in lifetimes),peak_live_bytes=peak,allocated_bytes=total,scratch_bytes=scratch,unreused_scratch_bytes=unreused,saved_scratch_bytes=unreused-scratch),
                cache_key=hashlib.sha256(json.dumps(key_data,sort_keys=True,separators=(',',':')).encode()).hexdigest())
    validate_plan_memory(plan)
    return plan


def validate_plan_memory(plan):
    """Executable memory invariant, also useful for imported plan diagnostics."""
    mapping = plan['tensor_buffers']
    for kernel in plan['kernels']:
        if mapping[str(kernel['output'])] in {mapping[str(i)] for i in kernel['inputs']}:
            raise ValueError('A dispatch input overlaps its output storage')
    by_buffer = defaultdict(list)
    for life in plan['lifetimes']: by_buffer[life['buffer']].append(life)
    for entries in by_buffer.values():
        entries.sort(key=lambda life:life['start'])
        for a,b in zip(entries,entries[1:]):
            if a['end']>=b['start']: raise ValueError('Overlapping tensor lifetimes share a buffer')


def execute_plan(plan, bindings, simulate_memory=False):
    """Run optimized graph in NumPy; fusion is only a GPU optimization.

    The optional schedule replay stores kernel outputs in planned buffers to test
    reuse over forward and backward lifetimes. Returns outputs and uncommitted
    updates. Never mutates the caller's bindings.
    """
    nodes = {n['id']:n for n in plan['nodes']}; values = {}; buffers = {}
    retained_bytes = sum(math.prod(n['shape'])*np.dtype(n['dtype']).itemsize for n in plan['nodes'])
    if not simulate_memory and retained_bytes>min(plan['options']['max_allocation_bytes'],128*1024*1024):
        raise ValueError('NumPy retained tensor arrays exceed allocation cap; reduce the graph or model size')
    for node in plan['nodes']:
        if node['op'] not in LEAVES: continue
        if node['op']=='constant':
            value = np.asarray(node['attrs']['value'],dtype=node['dtype']).reshape(node['shape'])
        else:
            if node['name'] not in bindings: raise ValueError(f'Missing binding {node["name"]}')
            value = np.asarray(bindings[node['name']])
            if value.shape != tuple(node['shape']) or value.dtype != np.dtype(node['dtype']): raise ValueError(f'Binding shape/dtype mismatch: {node["name"]}')
            if not np.isfinite(value).all(): raise ValueError('Bindings must be finite')
        if simulate_memory: buffers[plan['tensor_buffers'][str(node['id'])]] = value.copy()
        else: values[node['id']] = value
    for name,classes in plan['label_specs'].items():
        labels = np.asarray(bindings[name])
        if np.any(labels<0) or np.any(labels>=classes): raise ValueError('Class labels out of range')
    if simulate_memory:
        for kernel in plan['kernels']:
            local = {j:buffers[plan['tensor_buffers'][str(j)]] for j in kernel['inputs']}
            for ident in kernel['node_ids']:
                node = nodes[ident]; local[ident] = _apply(node,[local[j] for j in node['inputs']])
            buffers[plan['tensor_buffers'][str(kernel['output'])]] = local[kernel['output']].copy()
        return {name:buffers[plan['tensor_buffers'][str(ident)]].copy() for name,ident in {**plan['outputs'],**plan['updates']}.items()}
    for node in plan['nodes']:
        if node['op'] not in LEAVES: values[node['id']] = _apply(node,[values[j] for j in node['inputs']])
    return {name:values[ident].copy() for name,ident in {**plan['outputs'],**plan['updates']}.items()}
