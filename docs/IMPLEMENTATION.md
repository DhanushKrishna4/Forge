# Forge implementation status

Started 2026-10-02 in an empty workspace. Status refreshed 2026-10-08. This record distinguishes implemented features from the environments in which they have been verified.

## Delivered components

- [x] Immutable typed tensor graph, rank 0–2 operator contract, NumPy execution and explicit domain/resource validation.
- [x] Reverse-mode differentiation, gradient accumulation through shared dependencies, SGD, momentum and Adam expressed in the same graph.
- [x] Independent finite-difference and PyTorch references for operations, gradients and complete optimizer steps.
- [x] Python compiler validation, dead-code elimination, constant folding, common-subexpression elimination, bounded fusion, scheduling, full-training lifetimes, scratch reuse and WGSL generation.
- [x] Generic TypeScript WebGPU execution of the Python-generated plan, resident parameters/state, GPU-to-GPU evaluation copies and atomic completed-step updates.
- [x] Native XOR, moons, spiral and digits training, saved trained checkpoints, measured pilot/resource reports and validation-based checkpoint selection.
- [x] The same Python package in a Pyodide worker; CPU and GPU whole-graph training, bounded requests, stale-result protection, watchdog and device-loss recovery.
- [x] Interactive 2D/custom and digits workspaces; graph, gradient, generated-shader and lifetime inspectors; benchmark controls and experiment export/import.
- [x] Versioned bounded JSON persistence with portable checksums and lossless RNG state.
- [x] Native and production-browser numerical conformance, training, checkpoint continuation and recovery checks.
- [x] Native and production-browser benchmark reports with actual samples and declared timing boundaries.
- [x] Wheel/source packaging, verified isolated wheel installation, locked dependencies, static build configuration and manual-only Pages workflow.
- [x] Production-browser CPU and WebGPU acceptance under `/Forge/`, including runtime/data/checkpoint loading and recovery.
- [x] Final production-browser benchmark measurements with CPU clock-resolution mitigation and synchronized GPU completion.
- [x] Production UI review at 1280×800 and 390×844, including overflow, readable controls, accessible point editing and inspector interactions.

## Verification evidence

| Evidence | Recorded result |
|---|---|
| Native suite, 2026-10-08 | 302 passing tests in 3.52 seconds, including installed development-only PyTorch references. |
| [CPU browser acceptance](../benchmarks/browser-cpu.json) | 13/13 checks and 25/25 numerical fixtures; WebGPU explicitly disabled. |
| [WebGPU browser acceptance](../benchmarks/browser-webgpu.json) | 14/14 checks and 25/25 numerical fixtures on the reported Apple `metal-3` adapter, `isFallbackAdapter=false`. |
| Browser environment | Chrome 155; Pyodide 0.27.7, Python 3.12.7, NumPy 2.0.2; no cross-origin isolation or SharedArrayBuffer. Production-server URLs under `/Forge/` are preserved in the reports. |
| [Digits checkpoint](../web/public/assets/digits-checkpoint.metrics.json) | 800-step budget; selected step 310 by validation loss; 272/279 correct on the held-out test split (97.491%). |
| [Native benchmark](../benchmarks/native.json) | 14 rows, 15 samples per row after three warmups; real timings and planned dispatch/scratch statistics. |
| [Production browser benchmarks](../benchmarks/webgpu.json) | 14 rows per backend, 15 samples each; synchronized GPU completion/readback and separate [Pyodide samples](../benchmarks/pyodide.json) averaging 20 executions per sample to reduce clock quantization. |
| [UI screenshots](screenshots/desktop-playground.png) | Desktop 1280×800 and [mobile 390×844](screenshots/mobile-digits.png) reviewed without horizontal overflow; training controls and inspector workflows exercised. |
| [Installed wheel](../benchmarks/package-validation.json) | Clean-environment import, compiled-plan equivalence, one Adam update and CLI checks passed; verified wheel SHA-256 recorded. |

Production acceptance verified asset loading, complete training and recovery from the local static build. A local production pass does not establish a future hosted deployment. The full numerical tolerances, timings and limitations are in [validation](validation.md).

## Integration decisions

The Python package is authoritative for tensor semantics, differentiation, optimizer formulas and compilation. TypeScript owns the interface, worker protocol and GPU resources. Browser requests cross the Python/JavaScript boundary as coarse JSON commands rather than per-operation calls. Inspection is a separately labeled NumPy execution and is excluded from performance measurements.

Experiment SHA-256 normalizes JSON values using type tags, sorted object keys and exact binary64 hexadecimal numeric encoding, including signed-zero normalization. This survives Python/JavaScript number-spelling changes. PCG64's 128-bit `state` and `inc` are bounded decimal strings in exports, restored as validated integers. Unsafe plain JSON integers are rejected.

The CLI accepts `--digits-asset` for installed-wheel usage; source checkouts retain the prepared repository asset as a default. Browser runtime/data assets remain separate from the numerical wheel. Ordinary builds package prepared data and the trained checkpoint without retraining.

The public source repository is [DhanushKrishna4/Forge](https://github.com/DhanushKrishna4/Forge). [Deployment instructions](deployment.md) document the manual workflow and the checks required for a hosted release.
