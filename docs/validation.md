# Validation record

The native verification run on 2026-10-08 completed **302 tests in 3.52 seconds** with Python 3.13.12, NumPy 2.2.6, pytest 8.4.2 and development-only PyTorch 2.8.0 on macOS arm64. This is a recorded run, not a timing guarantee. Browser acceptance checks are reported separately; native execution alone does not verify a GPU shader or browser interaction.

```sh
# Entire native suite, including PyTorch when the reference dependencies are installed:
PYTHONPATH=src .venv/bin/python -m pytest -q

# Fast tensor, differentiation and optimizer checks:
PYTHONPATH=src .venv/bin/python -m pytest tests/test_core.py tests/test_optim.py -q

# Independent reference results, including printed absolute/relative errors:
PYTHONPATH=src .venv/bin/python -m pytest tests/test_reference.py -v -s

# Compiler equivalence, training artifacts and safe experiment restoration:
PYTHONPATH=src .venv/bin/python -m pytest tests/test_compiler.py tests/test_training.py tests/test_serialization.py -q
```

The reference suite uses `pytest.importorskip` only when PyTorch is absent. PyTorch was installed and all reference tests ran in the recorded verification. It is not a Forge runtime dependency and is not downloaded into the browser.

## Numerical checks

Forward results are compared with NumPy across float32 and float64, scalar/vector/matrix inputs, all required pointwise operations, reductions, materialized layout changes and broadcasting. The suite validates immutable graph records, exact binding shapes/dtypes, domain errors, label range/shape, resource bounds, and unsupported control flow.

Central finite differences in float64 cover every differentiable operation, softmax along either matrix axis, stable cross-entropy, reshape/transpose, broadcast reductions over every expanded axis, matrix multiplication, fan-out, repeated parameters and unused targets. Ordinary fixtures avoid nondifferentiable points; ReLU's zero derivative at zero is asserted separately. Loss cases include logits scaled by 1,000, ties at 10,000, batch size one, one class and varying class counts.

Finite-difference perturbations are `1e-6`, or `1e-4` for large-logit fixtures. Typical tolerances are relative `2e-6` and absolute `2e-8`. Float64 cancellation in the difference quotient explains these tolerances; they are not arbitrary changes to accommodate failures.

Independent PyTorch comparisons use identical explicit input, label and parameter arrays. They cover composite pointwise functions, scalar/vector/matrix shapes, reduction/broadcast/layout combinations, shared parameters, softmax Jacobian-vector products, cross-entropy and five complete MLP training steps. Every gradient, parameter and optimizer state is checked for SGD, momentum and Adam.

Measured maximum errors across each five-step MLP comparison:

| Optimizer | Dtype | Maximum absolute error | Maximum relative error |
|---|---|---:|---:|
| SGD | float32 | 1.1920929e-7 | 3.368041e-6 |
| Momentum | float32 | 1.1920929e-7 | 3.368041e-6 |
| Adam | float32 | 6.5565109e-7 | 4.6651895e-4 |
| SGD | float64 | 2.220446e-16 | 4.0906189e-14 |
| Momentum | float64 | 2.220446e-16 | 6.1876514e-15 |
| Adam | float64 | 2.220446e-16 | 3.1053103e-14 |

The printed relative-error denominator is `max(abs(reference),1e-12)`. Adam's largest relative float32 error occurs on a small reference value; the assertion combines relative `3e-5` and absolute `3e-7` tolerances. Float64 references use relative `3e-11`, absolute `3e-12`. The table includes losses, logits, gradients and parameters; optimizer moments/counters are also asserted. It does not claim that every number reaches the largest error, or that long GPU trajectories remain identical.

## Compiler and state checks

Compiler tests compare all passes disabled, each pass individually enabled, all passes combined, and reuse disabled/enabled. They include randomized bounded graphs, forward/backward fan-out, requested gradient roots, immutable-only constant folding, changing mutable parameter values, bounded fusion and inspection materialization. Schedule replay uses the planned storage slots to detect unsafe reuse across backward consumers. Dedicated fixtures require fewer dispatches from fusion and lower allocated scratch capacity from reuse.

Multi-step SGD/momentum/Adam tests compare both explicit NumPy formulas and independent PyTorch state. Checkpoint tests resume from a saved RNG/parameter/state snapshot and require identical subsequent batches and values on the native backend. A numerical failure preserves the last completed parameter/state version and restores the RNG. Exported dictionaries are detached snapshots; later configuration/history mutation cannot corrupt them.

Import tests reject wrong versions/checksums, unknown configuration, excessive shapes or byte counts, missing/extra state, mismatched Adam counters, nonfinite/invalid data, malformed RNG, changed dataset partitions and invalid histories. A checkpoint never deserializes executable code. Digits assets require the exact original seed-2026 split, not merely any complete partition.

## Real training results

A small acceptance fixture trains each 2D generator for 30 Adam updates from seeded random initialization: 100 points, noise 0.05, two 12-unit tanh layers, batch 32, learning rate 0.02. Observed training losses were:

| Dataset | Initial loss | Loss after 30 updates |
|---|---:|---:|
| XOR | 0.72934276 | 0.20340766 |
| Two moons | 0.74753088 | 0.10239514 |
| Spiral | 0.75863761 | 0.56589341 |

The acceptance assertions require changed predictions, changed parameters, finite gradients and a substantial decrease on these fixtures. They are not claims about every seed or arbitrary configuration.

The delivered digits artifact was trained with Forge's NumPy interpreter using `64 → 64 → 32 → 10`, ReLU, Adam, batch 64 and learning rate 0.003. The fixed data split is 1,252 training, 266 validation and 279 test examples. An 800-step bounded run selected the checkpoint at step 310 by lowest validation loss, without consulting test accuracy. Re-evaluation of the saved checkpoint reproduces its saved confusion matrix and **272/279 = 97.4910% held-out test accuracy**, with loss **0.09598914**.

The associated measured pilot completed 10 steps at 2.4103 ms/step and estimated 1.9042 seconds for the remaining fixed budget. The recorded training/selection interval was 1.9139 seconds. Peak process RSS was 46,727,168 bytes, while persistent parameter/optimizer arrays occupied 78,844 bytes. These are different measurements; neither is presented as browser or GPU memory. The machine-readable evidence is [digits-checkpoint.metrics.json](../web/public/assets/digits-checkpoint.metrics.json).

A separate saved two-moons run used two 32-unit tanh layers and 600 Adam updates. Its fixed 36-example test partition reached 100% accuracy, loss 0.00026832. That small generated-data result should not be generalized to arbitrary draws. Pilot time was 2.2295 ms/step, peak process RSS 38,748,160 bytes and persistent tensor storage 14,620 bytes. See [moons-checkpoint.metrics.json](../benchmarks/moons-checkpoint.metrics.json).

To reproduce artifacts, without making ordinary site builds retrain them:

```sh
PYTHONPATH=src .venv/bin/python scripts/prepare_digits.py
PYTHONPATH=src .venv/bin/python -m forge_ml.cli train --dataset digits --steps 800 --seed 42 --output web/public/assets/digits-checkpoint.json
PYTHONPATH=src .venv/bin/python -m forge_ml.cli train --dataset moons --steps 600 --seed 42 --output benchmarks/moons-checkpoint.json
```

## Installed package

The [wheel verification report](../benchmarks/package-validation.json), recorded on 2026-10-08, passed in a temporary virtual environment with the source checkout absent from Python's import path. It installed `forge_ml-0.1.0-py3-none-any.whl` and NumPy 2.2.6, constructed a 53-node graph, generated 13 kernels, checked compiled outputs against the interpreter, and completed one Adam step. The loss decreased from 1.0719291 to 1.0637667. CLI help and its missing-digits-asset error also passed; the temporary environment was removed afterward.

```sh
.venv/bin/python -m build
.venv/bin/python scripts/check_wheel.py --report benchmarks/package-validation.json
```

The report includes the exact verified wheel's SHA-256. A rebuilt wheel has its own checksum and should be checked again before distribution.

## Browser acceptance

Saved production-server runs on 2026-10-08 passed **13/13 CPU checks** and **14/14 WebGPU checks** in Chrome 155. Both used Pyodide 0.27.7, Python 3.12.7 and NumPy 2.0.2, with `crossOriginIsolated=false` and `sharedArrayBuffer=false`. CPU acceptance explicitly disabled WebGPU. GPU acceptance reported vendor `apple`, architecture `metal-3` and `isFallbackAdapter=false`; the browser did not expose a specific model name.

The reports are [CPU acceptance](../benchmarks/browser-cpu.json) and [WebGPU acceptance](../benchmarks/browser-webgpu.json). Their URLs identify the actual static production build served at `http://127.0.0.1:4173/Forge/validation.html`. These runs loaded the production worker, Python archive, local WASM/runtime assets, digits dataset and checkpoint through the repository subpath. Earlier development runs remain in the `*-dev.json` files as historical evidence.

Each run passed all **25 numerical fixtures**. The largest observed absolute errors across these fixtures were 1.1920929e-7 on Pyodide NumPy and 2.8833747e-6 on WebGPU. The reports preserve each fixture's tolerance, absolute/relative error and checked scalar count. Ordinary GPU cases use relative 3e-5 and absolute 3e-6 tolerances; Adam cases allow relative 2e-4 and absolute 2e-5 for accumulated float32 moment/bias-correction differences. Large translated-logit cross-entropy cases use tighter relative 3e-6 and absolute 5e-7 checks. The simultaneous binding-swap fixture is exact.

Both backends also passed seeded 2D learning, pause/single-step boundaries, exact same-backend checkpoint continuation, invalid/oversized import rejection, tensor/gradient inspection, repeated reset with stable owned-buffer counts, stale-request rejection, compiler-option rebuilding, random-initialized digits training, checkpoint inference, worker replacement and an injected nonresponsive-worker watchdog. The GPU suite additionally injected device loss and restored the last completed checkpoint onto CPU. These are bounded acceptance cases, not exhaustive verification of every browser, configuration or device.

## Interface review

The production interface was reviewed at a 1280×800 desktop viewport and a 390×844 mobile viewport, with no horizontal overflow at either size. The mobile toolbar retains readable import/export names and the Stop control. Recorded screenshots show the [desktop playground](screenshots/desktop-playground.png) and [mobile digits workspace](screenshots/mobile-digits.png).

The interaction review exercised training/stop/reset, adding and moving custom points through the accessible point editor, forward/backward graph navigation, gradient and shader inspection, compiler-pass inspection and the memory timeline. The desktop screenshot records a real GPU-trained moons experiment at step 327, with displayed training loss 0.0018, validation loss 0.0028 and validation accuracy 100%. This is one seeded demonstration, separate from the saved benchmark fixtures and held-out digits result. The viewport/control review does not claim a comprehensive assistive-technology or cross-browser accessibility certification.

## Performance evidence

Browser conformance loads generated fixture plans and their native reference arrays into Pyodide NumPy or the real WebGPU executor. It compares every requested output, every parameter gradient, optimizer state and parameter values across multiple updates. Fixtures include the full operator set, odd matrix sizes, reduction boundaries, batch size one, stable losses and simultaneous binding updates. GPU execution must report an actual adapter and distinguish hardware from software. A missing/skipped GPU run is not a pass.

The benchmark suite separates plan/pipeline compilation, initial upload, completed execution, readback and end-to-end latency. It performs warmups, records repeated samples and reports median/variability. Instrumented inspection is excluded. CPU and GPU comparisons identify transfer costs separately; fewer dispatches or fewer allocated bytes are structural results, not promises of lower latency.

The [native report](../benchmarks/native.json) contains 14 measured rows: unoptimized and optimized forms of an elementwise chain, common subexpressions, scratch reuse, three matrix sizes and a complete Adam training step. Each row records 15 samples after three warmups, plus compiler options and tensor shapes. For the elementwise fixture, compilation reduces dispatches from 12 to 2 and planned scratch capacity from 180,224 to 16,384 bytes. The scratch-reuse fixture reduces that capacity from 45,056 to 8,192 bytes without changing its 12 dispatches.

The production [Pyodide report](../benchmarks/pyodide.json) records 15 samples per configuration, with 20 executions per sample after three individual warmups. Each total elapsed time is divided by 20; the report retains both totals and per-execution averages. This avoids the sub-resolution zero timings found in the earlier single-execution development report. The reported variation describes these batch averages, not individual request latency. All production CPU samples are positive.

For the complete Adam step, the measured Pyodide medians were 1.855 ms unoptimized and 1.635 ms optimized. These are synchronous CPU computation/update timings with no GPU transfers or browser startup. The production [WebGPU report](../benchmarks/webgpu.json) records 15 synchronized samples after three warmups on the same reported Apple Metal adapter. Its timings include queue completion, update commit and one output readback; plan/pipeline compilation and initial upload are reported separately. Timestamp queries were not used. Selected measured outcomes are:

| Fixture | Unoptimized median | Optimized median | Standard deviation, unoptimized → optimized | Dispatches | Planned scratch bytes |
|---|---:|---:|---:|---:|---:|
| Elementwise chain | 0.9 ms | 0.8 ms | 0.047 → 0.081 ms | 12 → 2 | 180,224 → 16,384 |
| Scratch reuse | 0.6 ms | 0.6 ms | 0.102 → 0.250 ms | 12 → 12 | 45,056 → 8,192 |
| Complete MLP Adam step | 1.8 ms | 1.3 ms | 0.072 → 0.162 ms | 189 → 86 | 60,536 → 10,504 |

The complete Adam-step pipeline/plan compilation measured 46.6 ms unoptimized and 27.3 ms optimized, with 0.2 ms initial upload for each. Compilation was outside repeated execution timing. Medians for individual phases need not sum to the total median. The unchanged scratch-reuse latency demonstrates that reducing planned capacity does not guarantee a speed improvement. Native, Pyodide and GPU timings have different completion/transfer boundaries and should not be treated as interchangeable throughput measurements. For exact timing boundaries, see [compiler verification and timing methodology](compiler.md).
