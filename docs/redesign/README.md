# Forge website redesign

The interface now follows an experiment from its visible prediction surface into the model, computation graph, generated kernels and memory plan. The graphite, copper and blue palette remains, with a new composition, navigation, typography scale and responsive layout. The Python engine, worker protocol, numerical defaults, trained assets and compiler behavior are unchanged.

## What changed

- **Playground:** a large live decision surface sits beside validation accuracy, losses and training controls. The model trace connects real layer shapes and parameter counts to their tensors in the graph inspector. Gradient norms and compiled-plan summaries provide the next level of detail.
- **Configuration:** dataset, network, training and execution settings live in a grouped drawer. Pending changes are explicit; applying them starts a fresh experiment. On narrow screens the training controls appear above the plot.
- **Graph:** a searchable operation index, dependency diagram, tensor values and gradients, source semantics, generated WGSL and compiler-pass membership replace the previous compact inspector arrangement.
- **Memory and benchmarks:** allocation totals lead into a filterable lifetime ledger. Benchmark bars show measured medians alongside complete sample statistics, phase timings and environment details.
- **Digits:** examples, drawing and prediction probabilities share the experiment layout. Saved checkpoint test accuracy remains explicitly labeled as recorded metadata, separate from current validation results.

The portfolio review informed the standard of finish and interaction care; it supplied no layout, assets or visual motifs. A substantive second pass tightened the header and typography, improved graph readability and search, corrected graph shortcuts, and refined mobile controls and the digits layout. [The first implementation capture](first-pass-desktop.jpg) records the earlier pass.

## Before and after

These are full-page captures at the listed viewport sizes, so image heights differ with content. Playground comparisons use the same initial Moons experiment: seed 42, 240 points, noise 0.12, two 16-unit tanh layers, Adam at 0.01 and batch size 32 on CPU.

| Viewport / state | Before | After |
| --- | --- | --- |
| Desktop, 1440 × 1000, step 0 | [Before](before-desktop.jpg) | [After](after-desktop.jpg) |
| Laptop, 1280 × 900, step 0 | [Before](before-laptop.jpg) | [After](after-laptop.jpg) |
| Tablet, 768 × 1024, step 0 | [Before](before-tablet.jpg) | [After](after-tablet.jpg) |
| Mobile, 390 × 844, step 0 | [Before](before-mobile.jpg) | [After](after-mobile.jpg) |
| Desktop, one training step | [Before](before-result.jpg) | [After](after-result.jpg) |
| Desktop, graph inspector | [Before](before-graph.jpg) | [After](after-graph.jpg) |

Additional views: [settings](after-settings.jpg), [memory](after-memory.jpg), [measured benchmarks](after-benchmarks.jpg), and [digits checkpoint](after-digits.jpg).

## Verification

| Check | Result |
| --- | --- |
| Native Python regression suite | 302 tests passed |
| TypeScript check and production build | Passed |
| Local production CPU acceptance | [13/13 checks passed](acceptance-cpu.json), including 25/25 numerical fixtures |
| Local production WebGPU acceptance | [14/14 checks passed](acceptance-webgpu.json), including 25/25 numerical fixtures |
| Responsive layout | Reviewed at 1440, 1280, 768 and 390 pixels; no horizontal page overflow observed |
| Browser console during interaction review | No errors observed |

Acceptance ran against the static production build under `/Forge/` in Chrome 155. The reports record Pyodide 0.27.7, Python 3.12.7 and NumPy 2.0.2 without cross-origin isolation or SharedArrayBuffer. CPU acceptance explicitly disabled WebGPU; GPU acceptance used an Apple Metal adapter with `isFallbackAdapter=false`.

The acceptance cases cover operators, gradients, optimizer state and updates against native NumPy; bounded training and single steps; exact same-backend checkpoint continuation; invalid and oversized imports; tensor capture; stable owned-buffer counts after resets; stale-result rejection; compiler option changes; digits training and checkpoint inference; worker replacement and watchdog recovery. The GPU run also injected device loss and restored the completed checkpoint on CPU.

Manual interaction review additionally exercised:

- Train, pause, single-step and reset. The matching one-step CPU comparison retained train loss **0.7129**, validation loss **0.6818** and validation accuracy **52.8%**.
- Draft settings, disabled training until apply, drawer dismissal with Escape and restored focus.
- Layer-to-graph navigation, kernel and source views, tensor/gradient capture, graph search and memory filters.
- A real [14-row CPU benchmark run](benchmark-cpu.json) with seven repetitions per row, including a verified JSON download from the production interface.
- Custom-point pointer editing and keyboard-accessible coordinate editing, followed by downloading and re-importing an experiment with its optimizer state.
- Loading the trained digits checkpoint, filtering incorrect predictions, selecting examples, drawing a digit, running inference and clearing the drawing.

## Scope of the evidence

These reports establish local production behavior; they do not themselves verify a new hosted deployment. Browser review was limited to Chromium on macOS. Safari, Firefox, other devices and comprehensive assistive-technology testing remain unverified. Timings are specific to the recorded browser, hardware and fixture settings; compiler dispatch or allocation reductions do not imply a universal speedup. Screenshots show real runtime states, not substitute numerical evidence.
