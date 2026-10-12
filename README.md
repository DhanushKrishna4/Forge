# Forge

An inspectable Python deep-learning framework and WebGPU compiler, with a static browser playground. Build a small network, train it, follow its gradients, inspect generated shaders and memory lifetimes, and measure optimization tradeoffs.

[Source on GitHub](https://github.com/DhanushKrishna4/Forge)

Forge implements its tensor graph, reverse-mode differentiation, SGD/momentum/Adam updates, compiler passes, memory planner and WGSL generation in Python. NumPy supplies arrays and numerical primitives. Pyodide runs that same package in a browser worker; TypeScript handles the UI and dispatches Python-generated plans. There is no ML backend, account, API key, paid service, or hidden PyTorch/JAX execution.

## Run locally

Tested with Python 3.13.12 and Node.js 26.4.0. Commands start in the repository root; `uv` is an optional fast installer.

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python scripts/prepare_assets.py --runtime
npm --prefix web ci
npm --prefix web run dev
```

Open [Forge locally](http://127.0.0.1:5173/Forge/). The first asset preparation downloads a pinned, unmodified Pyodide 0.27.7 runtime plus NumPy 2.0.2 (~16 MiB). Thereafter browser computation is entirely local and assets can be served offline. The native package pins NumPy 2.2.6; browser/native parity is tested with identical explicit arrays. Only one experiment trains at a time. Nothing trains automatically.

On macOS, if a sync utility marks an editable `.pth` hidden, Python 3.13 may ignore it. Clear that flag (`chflags nohidden .venv/lib/python3.13/site-packages/*forge*.pth`) or use `PYTHONPATH=src` for the commands below.

## Explore

- **Playground:** XOR, moons, spiral and editable custom points; 1–3 hidden layers, activations and optimizers; train/pause/step/stop/reset; real losses, accuracy, predictions and gradient norms. The digits workspace includes a Forge-trained checkpoint, validation examples, confusion matrix, probabilities and an 8×8 drawing canvas. Drawings can differ substantially from the training distribution.
- **Graph:** navigate shared dependencies in forward/backward graphs, inspect bounded values and gradients through a separate instrumented NumPy execution, compare compiler passes and inspect the actual generated WGSL.
- **Memory:** full forward/backward/update lifetimes and scratch-buffer reuse. Counts describe Forge-owned planned capacities, not total browser memory or physical GPU residency.
- **Benchmarks:** optimized/unoptimized fixtures with warm-up, repeated synchronized measurements, medians, variability, compilation/upload/execution/readback phases and downloadable JSON. GPU speedups are workload-dependent; fewer dispatches can still be slower.

The browser needs WebAssembly and module workers. WebGPU additionally needs a supported adapter exposed by the browser in a secure context (HTTPS or localhost); the app reports detected capabilities. Chrome 155 on macOS is the recorded browser environment. Other browser/device combinations require their own acceptance run. CPU training remains available without WebGPU.

CPU and WebGPU are selected for the whole training graph. WebGPU weights/state remain resident between steps; display inference copies weights GPU-to-GPU. An unavailable/lost device produces a clear CPU recovery option. Inspection and checkpoint exports are explicit readbacks, outside timed benchmarks.

## Numerical contract

Float32, contiguous tensors of ranks 0–2, positive static dimensions and NumPy-style broadcasting. Native float64 exists for validation. Supported operations: `+ - * /`, negation, rank-2 matmul, ReLU/tanh/sigmoid, exp/log/sqrt, sum/mean with explicit axes and keepdims, reshape, matrix transpose, broadcast-to, stable softmax, stable mean softmax cross-entropy, and MSE. Comparisons, selection and one-hot/reduction-to-shape expressions support derivatives. Integer labels are nondifferentiable and shape/range checked; ReLU's derivative at zero is 0.

No strided aliasing, graph in-place mutation, dynamic shapes, higher-order derivatives, general Python compilation, convolution, transformers or distributed training. Log requires positive inputs, sqrt nonnegative inputs, division nonzero denominators. There is no clipping that changes their mathematics. See [operator contract](docs/operators.md), [autodiff](docs/autodiff.md), [architecture](docs/architecture.md) and [compiler](docs/compiler.md).

Default limits: 100,000 trainable parameters, 1,048,576 elements/tensor, 10,000 compiled nodes, 2,048 custom rows, 1–128 batch size, 1–64 units per hidden layer, 10,000 native training steps (5,000 interactive), 8 cached shapes/plans, 8-operation fusion and 128 MiB aggregate owned GPU capacity including staging. Work estimates also bound serial shader loops. Runtime/driver overhead is separate.

## Reproduce

```sh
# Fast native contract/compiler/training/persistence checks
PYTHONPATH=src .venv/bin/python -m pytest -q --ignore=tests/test_reference.py
# Development-only independent reference (never a runtime dependency)
.venv/bin/python -m pip install -r requirements-reference.txt
PYTHONPATH=src .venv/bin/python -m pytest tests/test_reference.py -v -s
# Standalone examples
PYTHONPATH=src .venv/bin/python examples/tensors.py
PYTHONPATH=src .venv/bin/python examples/xor.py
# Regenerate data, then train ONE bounded experiment at a time
PYTHONPATH=src .venv/bin/python scripts/prepare_digits.py
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src .venv/bin/python -m forge_ml.cli train --dataset digits --steps 800 --output web/public/assets/digits-checkpoint.json
# Real native benchmark data (no inspection)
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src .venv/bin/python -m forge_ml.cli benchmark --repetitions 15 --output benchmarks/native.json
# Regenerate native fixtures / bundle the exact current Python sources
PYTHONPATH=src .venv/bin/python scripts/generate_gpu_fixtures.py
.venv/bin/python scripts/prepare_assets.py
```

The 64→64→32→10 digits MLP has 6,570 parameters. It was trained for 800 steps, and step 310 was selected by validation loss. The fixed split contains 1,252 training / 266 validation / 279 test images. Its held-out result is **272/279 correct (97.491%)**, above the 90% target. Test data were evaluated only after checkpoint selection. The live confusion matrix uses validation data; the saved test result is shown separately. Exact configuration, pilot time/RSS, losses, confusion matrix and dataset hash are in [`digits-checkpoint.metrics.json`](web/public/assets/digits-checkpoint.metrics.json). Builds use that artifact and never retrain it.

The current native suite passes **302 tests**. The saved production-browser reports pass **13 CPU checks** with WebGPU explicitly disabled and **14 WebGPU checks** on an Apple Metal adapter; both include 25 numerical fixtures. The runs served the static build at `/Forge/` without SharedArrayBuffer. Full verification is recorded in [docs/validation.md](docs/validation.md), alongside tolerances and measured limitations. Actual benchmark samples live in [benchmarks/](benchmarks/); a missing GPU adapter is never counted as a pass.

Production [WebGPU measurements](benchmarks/webgpu.json) on that adapter recorded a complete Adam-step median of **1.8 ms unoptimized → 1.3 ms optimized**, across 15 samples after three warmups. Those times include queue completion and one output readback; initial upload and compilation are separate. The scratch-reuse fixture reduced planned capacity from 45,056 to 8,192 bytes with the same 0.6 ms median. These results demonstrate the measured tradeoff for these fixtures, not a universal GPU speedup.

## Installable Python package

The numerical package builds as a wheel and source distribution:

```sh
.venv/bin/python -m build
.venv/bin/python scripts/check_wheel.py --report benchmarks/package-validation.json
```

The saved [package verification](benchmarks/package-validation.json) confirms an isolated wheel installation outside this checkout, a complete graph/autodiff/Adam step, compiled-plan equivalence and the `forge` console command. An installed wheel needs an explicit `--digits-asset /absolute/path/to/digits.json` for digits training; prepared browser data are separate from the numerical package.

## Browser tests and static production

```sh
.venv/bin/python scripts/prepare_assets.py --runtime
PYTHONPATH=src .venv/bin/python scripts/generate_gpu_fixtures.py
npm --prefix web run build
python3 scripts/serve.py --port 4173 --base /Forge/
```

Open [the production acceptance page](http://127.0.0.1:4173/Forge/validation.html) and run CPU acceptance, then WebGPU acceptance. Download the report to record that browser/device. This uses the actual production worker and includes numerical fixture parity, learning, bounded pause/step, checkpoint continuation, bad imports, inspection, repeated reset/resource checks, stale requests, digits training/checkpoint/prediction, and GPU device-loss recovery. It deliberately serves without cross-origin isolation or SharedArrayBuffer. Open [the production application](http://127.0.0.1:4173/Forge/) separately.

Export/import is bounded, versioned JSON with shape/dtype/value/structure validation and SHA-256. RNG 128-bit integers are decimal strings; checksums use exact binary64 number representations so Python↔JavaScript number spelling does not invalidate files. No pickle or imported executable code. Reproducibility means controlled data/state and justified numerical agreement, not universal CPU/GPU bit identity.

[Deployment instructions](docs/deployment.md) describe the manual GitHub Pages workflow, repository base path and caching. The production output is `web/dist/`.

MIT for Forge; see [third-party notices](THIRD_PARTY_NOTICES.md) for runtime licenses and CC BY 4.0 digits attribution.
