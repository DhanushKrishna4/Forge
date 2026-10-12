# Static builds and GitHub Pages

Forge produces ordinary static files. Python runs inside Pyodide in the visitor's browser worker; WebGPU also runs in that worker when available. Hosting does not execute Python, provide an inference service, require a database, or need application API keys.

The public source repository is [DhanushKrishna4/Forge](https://github.com/DhanushKrishna4/Forge). GitHub Pages uses the manual deployment workflow described below. Hosted browser acceptance must be verified separately from a local production build.

## Reproducible local setup

The recorded development environment uses Python 3.13.12 and Node.js 26.4.0. Runtime, development and reference dependencies are separated:

- `requirements-runtime.txt`: Forge plus native NumPy 2.2.6.
- `requirements-dev.txt`: locked test/build dependencies and scikit-learn used only to prepare the dataset.
- `requirements-reference.txt`: locked development dependencies plus PyTorch 2.8.0 for independent validation.
- `web/package-lock.json`: exact frontend dependency graph used by `npm ci`.
- `web/public/assets/runtime-manifest.json`: pinned Pyodide 0.27.7 and its NumPy 2.0.2 asset hashes. Browser NumPy comes from the compatible Pyodide package set, not the native wheel.

From the repository root:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
npm --prefix web ci
PYTHONPATH=src .venv/bin/python -m pytest --ignore=tests/test_reference.py -q
```

Use the reference dependency set when running the independent comparisons:

```sh
.venv/bin/python -m pip install -r requirements-reference.txt
PYTHONPATH=src .venv/bin/python -m pytest tests/test_reference.py -v -s
```

The installable wheel can be verified independently of an editable checkout:

```sh
.venv/bin/python -m build
.venv/bin/python scripts/check_wheel.py --report benchmarks/package-validation.json
```

The check creates a temporary virtual environment, installs the wheel and pinned NumPy, imports with Python's isolated `-I` mode outside the repository, runs one small graph/autodiff/Adam step and its compiled equivalent, checks the console command, and removes the environment. The first run may download NumPy; it does not install a reference framework or retrain a demonstration.

An installed package does not bundle the browser's digits asset. Supply its prepared JSON explicitly when training outside the source checkout:

```sh
forge train --dataset digits --digits-asset /absolute/path/to/Forge/web/public/assets/digits.json --steps 800 --seed 42 --output ./digits-checkpoint.json
```

Inside the source checkout the existing repository asset is the default. A missing asset produces an actionable error before training starts. `scripts/prepare_digits.py` remains the separate dataset-regeneration command.

On Linux, the CI reference job first selects the CPU-only PyTorch wheel from the official PyTorch CPU index. That installed version satisfies the reference lock while avoiding CUDA packages; Forge does not use CUDA. No GPU or paid service is needed for the reference suite.

## Build the deployable directory

```sh
PYTHONPATH=src .venv/bin/python scripts/prepare_assets.py --runtime
PYTHONPATH=src .venv/bin/python scripts/generate_gpu_fixtures.py
FORGE_BASE=/Forge/ npm --prefix web run build
```

`prepare_assets.py` always regenerates `web/public/assets/forge.zip` from the current Python source files. `--runtime` downloads missing files from the pinned Pyodide release, verifies the NumPy package checksum and writes a runtime manifest. Existing downloaded runtime files are reused. The first preparation needs network access; the generated site serves its own runtime assets afterward.

`generate_gpu_fixtures.py` creates bounded deterministic operator/backend reference fixtures. It does not retrain a demonstration checkpoint. The ordinary build uses the small digits dataset and genuinely trained checkpoint already included in `web/public/assets/`; it never runs dataset preparation or model training. Artifact regeneration commands are in [the validation record](validation.md).

Vite writes the complete site to `web/dist/`, including the worker, Python archive, runtime/WASM files, dataset, checkpoint and backend validation page. The Pages artifact consists of that directory alone; `.venv`, source checkouts and `node_modules` are not deployed.

Preview over HTTP rather than opening HTML with `file://`:

```sh
npm --prefix web run preview -- --host 127.0.0.1 --port 4173
```

Open `http://127.0.0.1:4173/Forge/`. The production build defaults to `/Forge/`; the trailing slash matters for asset resolution. For a different repository path, set `FORGE_BASE=/YourRepository/` at **build time**. For a user/organization root site or a custom-domain root, set `FORGE_BASE=/`. Rebuild whenever this value changes.

To serve the exact directory at the repository subpath using the included small static server:

```sh
.venv/bin/python scripts/serve.py --port 4173
```

Development mode is also available:

```sh
npm --prefix web run dev
```

Use the URL printed by Vite, including `/Forge/` when configured. Runtime preparation is still needed before booting Python.

## Included workflows

[`.github/workflows/checks.yml`](../.github/workflows/checks.yml) runs on pushes, pull requests and manual dispatch. It has read-only repository permissions and three jobs:

1. Native numerical/compiler/training-artifact/serialization tests, plus Python package building.
2. Independent CPU PyTorch reference validation. It explicitly imports PyTorch before pytest so a missing dependency cannot be mistaken for a successful skipped reference run.
3. Static frontend typecheck/build at `/Forge/`, packaging current Python and the pinned browser runtime. Its downloadable artifact is retained for seven days; this job does not publish a site.

Pip and npm caches are keyed from their dependency lockfiles. The browser-runtime cache is keyed from the asset-preparation script and runtime manifest. Frontend dependencies use `npm ci`; the jobs do not run unconstrained package upgrades. The setup actions' documented cache inputs support dependency files outside the repository root. [Python setup and caching](https://github.com/actions/setup-python), [Node setup and caching](https://github.com/actions/setup-node).

[`.github/workflows/pages.yml`](../.github/workflows/pages.yml) has **only `workflow_dispatch`** as a trigger. Merely pushing it cannot publish. Its build job reruns native checks, regenerates the Python archive, prepares runtime/fixtures and builds static files. It uploads `web/dist` using `actions/upload-pages-artifact@v3`; the dependent deploy job uses `actions/deploy-pages@v4`.

Only the deploy job receives `pages: write` and `id-token: write`; checkout/build remain read-only. Deployment targets the `github-pages` environment and exposes the URL returned by the action. Environment protection rules can govern which branches or reviewers may deploy. The Pages job dependency ensures deployment consumes a completed build artifact, and its concurrency group avoids overlapping publications. These follow GitHub's documented custom-workflow requirements. [GitHub Pages workflow requirements](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages), [upload action v3](https://github.com/actions/upload-pages-artifact/tree/v3), [deploy action v4](https://github.com/actions/deploy-pages/tree/v4).

## Publishing a revision

To publish a reviewed revision:

1. Push the reviewed source and artifact files to that repository.
2. In repository **Settings → Pages → Build and deployment**, choose **GitHub Actions** as the source. This requires repository administrative or maintainer access. [GitHub publishing-source instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).
3. If the repository path differs from `/Forge/`, set repository Actions variable `FORGE_BASE` to the desired slash-delimited path before running the workflow. A root deployment uses `/`.
4. Run **Deploy Forge to GitHub Pages (manual)** from the Actions tab, choosing the reviewed revision. Satisfy any `github-pages` environment protection rules.
5. Verify the resulting public URL with a cold browser load: Python/WASM boot, CPU training, dataset/checkpoint loading, experiment export/import and repository-relative assets. Exercise WebGPU separately on a supported adapter.

The workflow does not create the repository or silently enable publishing settings. A successful local build cannot establish GitHub account permissions, deployment-environment policy or availability of a visitor's GPU adapter. Browser/backend validation and measured limitations are recorded separately in [validation](validation.md).
