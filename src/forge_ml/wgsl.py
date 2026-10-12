"""Portable f32 WGSL emission. No shader feature beyond core WebGPU is used."""
from __future__ import annotations

import math

POINTWISE = frozenset({'add','sub','mul','div','neg','relu','tanh','sigmoid','exp','log','sqrt','greater','where','broadcast_to'})
LEAVES = frozenset({'input','parameter','state','constant'})
OPS = POINTWISE | LEAVES | {'matmul','sum','mean','reshape','transpose','softmax','cross_entropy','one_hot'}


def size(shape):
    return math.prod(shape)


def broadcast_index(source, target, index='i'):
    """Map a contiguous output offset to the corresponding broadcast operand."""
    if not source:
        return '0u'
    padded = [1] * (len(target) - len(source)) + list(source)
    terms = []
    for axis, dim in enumerate(padded):
        if dim != 1:
            out_stride = size(target[axis + 1:])
            in_stride = size(padded[axis + 1:])
            terms.append(f'((({index}) / {out_stride}u) % {dim}u) * {in_stride}u')
    return '(' + ' + '.join(terms) + ')' if terms else '0u'


def _expression(op, args):
    if op in {'add','sub','mul','div'}:
        return f'({args[0]} {dict(add="+",sub="-",mul="*",div="/")[op]} {args[1]})'
    if op == 'neg': return f'(-{args[0]})'
    if op == 'relu': return f'max({args[0]}, 0.0)'
    if op == 'sigmoid':
        e = f'exp(-abs({args[0]}))'
        return f'select({e} / (1.0 + {e}), 1.0 / (1.0 + {e}), {args[0]} >= 0.0)'
    if op in {'tanh','exp','log','sqrt'}: return f'{op}({args[0]})'
    if op == 'greater': return f'select(0.0, 1.0, {args[0]} > {args[1]})'
    if op == 'where': return f'select({args[2]}, {args[1]}, {args[0]} != 0.0)'
    if op == 'broadcast_to': return args[0]
    raise ValueError(f'Unsupported pointwise op {op}')


def emit_kernel(nodes, members, external, tiled=False):
    root = nodes[members[-1]]
    output = root['id']
    n = size(root['shape'])
    declarations = []
    names = {}
    for slot, ident in enumerate(external):
        names[ident] = f'b{slot}'
        dtype = 'i32' if nodes[ident]['dtype'] == 'int32' else 'f32'
        declarations.append(f'@group(0) @binding({slot}) var<storage, read> b{slot}: array<{dtype}>;')
    output_dtype = 'i32' if root['dtype']=='int32' else 'f32'
    declarations.append(f'@group(0) @binding({len(external)}) var<storage, read_write> result: array<{output_dtype}>;')
    header = '\n'.join(declarations) + '\n'
    args = [names[i] for i in root['inputs'] if i in names]
    body = ''
    dispatch = [(n + 63) // 64, 1, 1]
    if root['op'] in POINTWISE:
        lines = []
        locals_ = set()
        for ident in members:
            node = nodes[ident]
            operands = [f'v{j}' if j in locals_ else f'{names[j]}[{broadcast_index(nodes[j]["shape"],node["shape"])}]' for j in node['inputs']]
            if node['op']=='sigmoid':
                lines.append(f'let sigmoid_e{ident} = exp(-abs({operands[0]}));')
                expression = f'select(sigmoid_e{ident} / (1.0 + sigmoid_e{ident}), 1.0 / (1.0 + sigmoid_e{ident}), {operands[0]} >= 0.0)'
            else: expression = _expression(node['op'], operands)
            lines.append(f'let v{ident} = {expression};')
            locals_.add(ident)
        body = '\n'.join(lines) + f'\nresult[i] = v{output};'
    elif root['op'] == 'matmul':
        m, k = nodes[root['inputs'][0]]['shape']
        _, p = nodes[root['inputs'][1]]['shape']
        a, b = args
        if tiled:
            # Every invocation enters both barriers, including masked edge lanes.
            text = header + f'''
var<workgroup> tile_a: array<f32, 64>;
var<workgroup> tile_b: array<f32, 64>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
  let row = gid.y; let col = gid.x;
  let slot = lid.y * 8u + lid.x;
  var acc = 0.0;
  for (var base = 0u; base < {k}u; base += 8u) {{
    tile_a[slot] = 0.0; tile_b[slot] = 0.0;
    if (row < {m}u && base + lid.x < {k}u) {{ tile_a[slot] = {a}[row * {k}u + base + lid.x]; }}
    if (col < {p}u && base + lid.y < {k}u) {{ tile_b[slot] = {b}[(base + lid.y) * {p}u + col]; }}
    workgroupBarrier();
    for (var t = 0u; t < 8u; t += 1u) {{ acc += tile_a[lid.y * 8u + t] * tile_b[t * 8u + lid.x]; }}
    workgroupBarrier();
  }}
  if (row < {m}u && col < {p}u) {{ result[row * {p}u + col] = acc; }}
}}
'''
            return text, [(p + 7) // 8, (m + 7) // 8, 1]
        body = f'let row = i / {p}u; let col = i % {p}u;\nvar acc = 0.0;\nfor (var t=0u; t < {k}u; t+=1u) {{ acc += {a}[row*{k}u+t] * {b}[t*{p}u+col]; }}\nresult[i] = acc;'
    elif root['op'] in {'sum','mean'}:
        source = nodes[root['inputs'][0]]['shape']
        axes = list(root['attrs']['axis'])
        remaining = [d for ax,d in enumerate(source) if ax not in axes]
        reduced = [source[ax] for ax in axes]
        terms = []
        for ax, dim in enumerate(source):
            reduced_axis = ax in axes
            group = axes if reduced_axis else [x for x in range(len(source)) if x not in axes]
            shape = reduced if reduced_axis else remaining
            position = group.index(ax)
            index = 'r' if reduced_axis else 'i'
            terms.append(f'(({index} / {size(shape[position+1:])}u) % {dim}u) * {size(source[ax+1:])}u')
        idx = ' + '.join(terms) or '0u'
        count = size(reduced)
        divisor = f' / {float(count)}' if root['op'] == 'mean' else ''
        body = f'var acc = 0.0;\nfor (var r=0u; r < {count}u; r+=1u) {{ acc += {args[0]}[{idx}]; }}\nresult[i] = acc{divisor};'
    elif root['op'] == 'reshape':
        body = f'result[i] = {args[0]}[i];'
    elif root['op'] == 'transpose':
        rows, cols = nodes[root['inputs'][0]]['shape']
        body = f'result[i] = {args[0]}[(i % {rows}u) * {cols}u + i / {rows}u];'
    elif root['op'] == 'softmax':
        shape = root['shape']; axis = root['attrs']['axis']; count = shape[axis]
        stride = size(shape[axis+1:])
        body = f'''let base = i - ((i / {stride}u) % {count}u) * {stride}u;
var maximum = {args[0]}[base];
for (var r=1u; r < {count}u; r+=1u) {{ maximum = max(maximum, {args[0]}[base+r*{stride}u]); }}
var total = 0.0;
for (var r=0u; r < {count}u; r+=1u) {{ total += exp({args[0]}[base+r*{stride}u] - maximum); }}
result[i] = exp({args[0]}[i] - maximum) / total;'''
    elif root['op'] == 'cross_entropy':
        batch, classes = nodes[root['inputs'][0]]['shape']
        body = f'''var loss = 0.0;
for (var row=0u; row < {batch}u; row+=1u) {{
  let base = row*{classes}u; var maximum = {args[0]}[base];
  for (var c=1u; c < {classes}u; c+=1u) {{ maximum = max(maximum, {args[0]}[base+c]); }}
  var total = 0.0;
  for (var c=0u; c < {classes}u; c+=1u) {{ total += exp({args[0]}[base+c] - maximum); }}
  // maximum >= target: max is exact here and blocks unsafe driver reassociation
  // of a small accumulated loss with the large unshifted logits.
  let target_gap = max(maximum - {args[0]}[base+u32({args[1]}[row])], 0.0);
  loss += log(total) + target_gap;
}}
result[i] = loss / {float(batch)};'''
    elif root['op'] == 'one_hot':
        classes = root['attrs']['classes']
        body = f'result[i] = select(0.0, 1.0, i32(i % {classes}u) == {args[0]}[i / {classes}u]);'
    else:
        raise ValueError(f'WGSL backend does not support {root["op"]}')
    return header + f'@compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\nlet i = gid.x;\nif (i >= {n}u) {{ return; }}\n{body}\n}}\n', dispatch
