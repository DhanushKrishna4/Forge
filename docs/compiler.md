# Compiler and WebGPU execution

Forge compiles the **combined forward, reverse-mode backward, and optimizer-update graph** in Python. The unoptimized NumPy interpreter and the compiler consume the same immutable nodes. TypeScript executes the emitted plan; it does not infer shapes, implement autodiff, or implement optimizer formulas.

## Public boundary

```python
from forge_ml.compiler import compile_graph, execute_plan

plan = compile_graph(graph, {"loss": loss, "dW": gradient},
                     updates=updates,
                     options={"fusion": True, "reuse": True})
values = execute_plan(plan, bindings)  # NumPy, does not commit updates
bindings.update({name: values[name] for name in plan["updates"]})
```

Output and update names must be distinct. An output is explicitly observable and held through the end of execution. Every update maps an existing parameter/state binding name to a shape- and dtype-matching new value. The caller commits updates only after the whole graph succeeds. The browser executor does that commit internally by default. Imported experiments contain validated configuration/data, not executable arbitrary plans or WGSL.

`execute_plan(..., simulate_memory=True)` replays the generated kernel schedule, retaining only group-local fused values and writing each group result into its planned buffer slot. It is used in memory-equivalence tests. Ordinary NumPy execution checks its retained semantic-array capacity before allocating and honors DCE, folding, and CSE; GPU fusion and scratch-buffer reuse do not claim to speed up NumPy.

## Pipeline

1. **Validation** independently infers every operation's shape/dtype from its inputs. It checks normalized axes, reshape element counts, matrix multiplication dimensions, ordered dependencies, constants, binding names, resource bounds, and the complete supported operation set. Parameters and state remain external bindings.
2. **DCE** walks backward from outputs and update expressions. A requested gradient is a root even if it is not required by an optimizer. Update targets remain allocated in degenerate binding-copy updates.
3. **Constant folding** uses the same `interpreter.apply_op` as normal execution, only when every operand is an immutable constant. Mutable inputs, parameters, and optimizer state are never folded. Cumulative immutable storage is checked before each folded array is materialized. No reassociation, `x/x` rewriting, or other real-number-only identities are used.
4. **CSE** hashes operation, *ordered* input identities, normalized attributes, shape, and dtype. Only exact expressions are merged. Mutable named leaves retain their distinct identities.
5. **Pointwise fusion** combines same-shape pointwise producers with their sole consumer. Observable outputs, fan-out, matmul, reduction, softmax, transpose, and reshape form boundaries. A group emits intermediate `let` values in dependency order. Defaults are at most 8 operations and 8 storage bindings including its output; limits are configurable within hard caps. Instrumentation disables fusion.
6. **Scheduling** orders kernel roots topologically across the entire training DAG. A node's last use can be well after its forward computation because backward expressions still read it.
7. **Lifetimes/reuse** assign separate persistent/input/constant/output buffers and reuse scratch capacity only when the prior tensor's last use is *strictly before* the new tensor's birth. Best fit picks the smallest compatible free capacity. A dispatch's input storage therefore cannot alias its output. There are no storage views or implicit in-place updates.
8. **WGSL emission** generates core-WebGPU float32 shaders. The plan includes actual sources, bindings, dispatch geometry, source mapping, and buffer capacities. CPU float64 plans are retained for validation and explicitly marked incompatible with WebGPU.

Options `dce`, `fold`, `cse`, `fusion`, `reuse`, and `tiled` can each be disabled. Setting all six to `False` provides an executable comparison baseline. `instrument=True` preserves every original graph node as a named `tensor:ID` output; this includes additional allocations and must not be used for performance measurements.

The inspectable `passes` records have `before`, `after`, `unit`, changed/removed source IDs, before/after node ID sets, and explanations. These numbers distinguish nodes, dispatches, and scratch bytes. Folding replaces expressions, so its replacement IDs appear in `changed`, not `removed`. Original records remain in `source_nodes`; optimized records remain in `nodes`. The `source_map` marks each original node as materialized, fused, CSE-aliased, folded, or dead. An inspector must not invent a value for a fused/dead tensor; it can run instrumentation explicitly.

## Plan schema (version 1)

- `nodes`, `source_nodes`: JSON-safe node records with IDs, operation, inputs, shape, dtype, attributes, kind, name, provenance.
- `kernels`: ID, operation/group type, contributing `node_ids`, ordered external `inputs`, single `output`, generated `wgsl`, dispatch counts, and an explicit *estimated* scalar-work bound.
- `buffers`: ID, byte capacity, storage kind, dtype, and tensors assigned to that capacity.
- `tensor_buffers`: materialized node ID → buffer ID.
- `bindings`: binding name → node/buffer/shape/dtype/kind.
- `outputs`, `updates`: externally observable name → optimized node ID.
- `lifetimes`: node, buffer, inclusive start/end schedule indices, logical bytes, buffer capacity, kind. Leaves begin at −1, outputs are held through the end index.
- `stats`: source and optimized node counts, kernels, fused nodes, logical source/optimized tensor bytes, materialized tensor bytes, peak simultaneously live materialized bytes, total allocated capacity, and reused/unreused scratch capacities.
- `cache_key`: SHA-256 of optimized graph structure, shapes, dtypes, constants, output/update roots, and compiler options. Parameter/input/state values do not enter the key. The executor validates plan resource requirements against the actual device before preparing it; cached executors are tied to their device.

These are **Forge-owned buffers**, not process RSS, total browser memory, physical GPU residency, or driver allocations. `logical_bytes` includes optimized semantic tensors that fusion may not materialize; `materialized_bytes` counts materialized tensors before storage reuse. `allocated_bytes` counts actual planned storage capacities. Staging buffers used for explicit readback or rare direct-binding update snapshots are tracked at runtime and destroyed in `finally` blocks. Shared-device accounting includes every spawned inference/benchmark executor and staging buffer, capped at 128 MiB in aggregate. Per-plan lower caps are honored too.

## Kernels and numerical details

Elementwise kernels map output coordinates to trailing-aligned input coordinates, substituting zero for broadcast axes. The same mechanism implements explicit broadcast-to and all pointwise backward/optimizer expressions. Sigmoid uses `exp(-abs(x))` and selects the stable positive/negative branch, so large finite inputs cannot overflow its internal exponential. Comparisons use float32 0/1 masks; `where` uses WGSL `select`.

Baseline matmul assigns one output to each invocation and sums its inner dimension sequentially. Tiled matmul uses 8×8 workgroups and two 64-element workgroup arrays. All invocations, including out-of-range edge lanes, execute both barriers in uniform control flow. Edge lanes load zeros; only valid rows/columns store a result. No early return appears before a tiled barrier. Non-square and non-multiple-of-eight dimensions are included in the GPU fixture suite.

Reductions deliberately use a serial inner loop **per output**, avoiding atomics and workgroup barriers. Sum/mean over every axis require one invocation; axis reductions parallelize independent outputs. Every reduction coordinate is generated from kept and reduced axes. Empty axis tuples are identity reductions. This is correct for partial shapes and is a bounded baseline, not a claim of optimal large-reduction performance.

Softmax computes a per-axis maximum and denominator for each output. Cross-entropy computes shifted log-sum-exp and the nonnegative target gap `max(row_max - target, 0)`, then adds the gap to the small logarithm. The outer max is exact because the target cannot exceed the row maximum; it prevents a native shader compiler from reassociating the subtraction with the small loss accumulator. Plain parentheses did not prevent this cancellation on the tested Apple Metal driver. Cross-entropy is never emitted as `log(softmax(x))`. The backward graph emits the batch-scaled probability-minus-one-hot expression from Python.

Scalar-work bounds limit expensive serial kernels. Defaults allow at most 64 million estimated scalar loop operations per kernel; the option can lower this limit. Shape validation additionally bounds tensors to 1,048,576 elements and compiler graphs to 10,000 nodes. The bound is a safeguard/estimate, not a measured execution time. Default demonstrations are substantially smaller. Domain requirements for division/log/sqrt are unchanged by compilation; uploaded arrays and returned results must be finite, and labels are checked before submission.

## TypeScript executor

```typescript
const trainer = await GPUExecutor.create();
await trainer.prepare(plan, initialBindings);
const { values, timing } = await trainer.run(batch, ["loss", "gradient_norm"]);
const checkpoint = await trainer.readBindings();
const inference = trainer.spawn(); // same device, separate plan/resources
await inference.prepare(inferencePlan, inferenceBindings);
await inference.copyBindingsFrom(trainer); // GPU-to-GPU, no weight readback
const prediction = await inference.run(newInputs, ["probabilities"], false);
inference.destroy();
trainer.destroy();
```

`create` discovers an adapter and records actual device limits, features, and available adapter information. No optional feature or timestamp support is required. `prepare` validates capacity/binding/dispatch limits, allocates buffers, checks shader compilation diagnostics, creates asynchronous pipelines and bind groups, and uploads constants/bindings. State omitted from initial bindings defaults to zero. Shader or validation failure releases all partial plan resources.

`run` uploads supplied bindings, dispatches the complete graph, awaits queue completion, reads requested old-step outputs, and commits updates. Parameter and optimizer-state buffers persist across steps. Ordinary optimizer update outputs already have dedicated storage; rare updates that directly refer to old bindings are snapshotted before any target copy, so simultaneous swaps work correctly. Reading an old parameter as an output returns its precommit value. `commit=false` executes without changing persistent bindings.

Readback uses a bounded staging buffer with `COPY_DST | MAP_READ`, awaits mapping, copies values into ordinary JS arrays, unmaps, and destroys the buffer. No mapped view escapes the call. Device loss, allocation/workload excess, compilation problems, and invalid labels produce explicit errors. `injectDeviceLoss` supports recovery-path testing. Destroying the final executor destroys the shared device; destroying a spawned executor only destroys its resources. A CPU recovery path is selected for the whole graph by the application, never inside individual kernels.

Timing uses synchronized wall-clock measurements, including queue completion and update copies. Preparation separates pipeline creation and initial upload. A run separates new uploads, completed execution/commit, and readback; end-to-end latency includes all three. These timings are not GPU timestamp measurements. When there are no requested outputs, readback is explicitly empty.

## Reproducible verification

```sh
PYTHONPATH=src .venv/bin/python -m pytest tests/test_compiler.py -q
PYTHONPATH=src .venv/bin/python scripts/generate_gpu_fixtures.py
```

The native suite checks each pass individually and together, randomized bounded graphs, float32/float64 outputs and every gradient, persistent mutable parameters, fresh multi-step SGD/momentum/Adam updates, fan-out, instrumentation, and scratch replay. It requires actual dispatch reduction from fusion and actual allocated-scratch reduction from reuse. GPU fixtures serialize generated plans plus NumPy references for four optimizer steps, all required operations, batch size one, odd matmul dimensions, reduction edges, stable large/tied logits, common-logit translations up to one million, and binding swaps. Ordinary GPU fixtures use `rtol=3e-5, atol=3e-6` for float32 accumulation/transcendental differences. Four-step Adam fixtures use `rtol=2e-4, atol=2e-5` because the exp-based bias correction subtracts a value near one (`beta2=0.999`), amplifying a transcendental ULP before updates accumulate. Exact binding-copy fixtures use zero tolerance. Browser reports preserve measured absolute and relative errors alongside these bounds; tolerance is not itself a measured error. Native schedule replay verifies lifetimes but **does not validate GPU shaders**; the separately reported real browser conformance run is required for that claim.

Reference semantics were checked against the [WebGPU explainer](https://gpuweb.github.io/gpuweb/explainer/) and [WGSL specification](https://www.w3.org/TR/WGSL/), including storage access, workgroup barriers, masked edge handling, and host readback boundaries.

### Hardware regression: preserving shifted cross-entropy

The first Apple `metal-3` run found one real shader failure: the large/tied-logit fixture produced loss `4.0057172775` versus NumPy `4.0058646202`; a float64 calculation gives `4.0058644661`. The NumPy reference was correct. Reassociating the shader's accumulator with the original ±10,000 logits reproduces the observed error, despite parentheses around the target subtraction.

The corrected WGSL computes `target_gap = max(row_max - target, 0.0)` before adding `log(sum(exp(logits-row_max)))`. This max does not clip the mathematical loss: `row_max >= target` is an exact invariant for finite logits. Its operation boundary prevents the observed driver reassociation from exposing the small loss accumulator to a large unshifted logit. The interpreter and reference values were unchanged, and no tolerance was widened.

The [subsequent hardware run](../benchmarks/browser-webgpu-dev.json) passed all 25 numerical fixtures and 14 browser acceptance checks. The original failing fixture's maximum absolute error across loss, probabilities, and gradients became `2.98e-8`. Four additional common-shift fixtures (0, +10,000, −10,000, +1,000,000) also had maximum absolute error `2.98e-8`, under tighter `rtol=3e-6, atol=5e-7` bounds. These are measurements on that hardware/browser, not universal accuracy or bitwise-reproducibility guarantees.
