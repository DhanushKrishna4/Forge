# Third-party notices

Forge's own implementation is MIT licensed (LICENSE). It is not affiliated with the following projects.

| Component | Role | License / source |
|---|---|---|
| Pyodide 0.27.7 | CPython/WebAssembly runtime in a worker | Mozilla Public License 2.0, https://github.com/pyodide/pyodide/tree/0.27.7 |
| CPython 3.12.7 in Pyodide | Browser Python interpreter | Python Software Foundation license, https://docs.python.org/3.12/license.html |
| NumPy 2.0.2 (browser), 2.2.6 (native) | Array storage and numerical primitives | BSD 3-Clause; full notices in wheel `numpy-*.dist-info/LICENSE.txt`, https://numpy.org/doc/stable/license.html |
| React / React DOM 19.1.0 | Interface rendering | MIT, https://github.com/facebook/react/tree/v19.1.0 |
| Vite 6.3.5 | Development and static build tooling | MIT, https://github.com/vitejs/vite/tree/v6.3.5 |
| TypeScript 5.8.3 | Frontend type checking | Apache 2.0, https://github.com/microsoft/TypeScript/tree/v5.8.3 |
| scikit-learn 1.7.2 | Development-only digits preparation | BSD 3-Clause, https://github.com/scikit-learn/scikit-learn/tree/1.7.2 |
| PyTorch 2.8.0 | Development-only independent numerical reference | BSD-style, https://github.com/pytorch/pytorch/tree/v2.8.0 |

Runtime dependencies are downloaded unmodified. Pyodide source corresponding to the distributed runtime is available at the versioned repository above; Forge's Python sources are distributed in `assets/forge.zip`. The deployed `licenses/` directory includes Pyodide and NumPy notices plus the frontend dependency licenses. Browser visitors receive only NumPy from Pyodide's scientific package set. Scikit-learn and PyTorch are not loaded into the browser and perform no Forge execution, differentiation, compilation or training.

## Digits dataset

E. Alpaydin and C. Kaynak (1998), **Optical Recognition of Handwritten Digits**, UCI Machine Learning Repository, DOI: https://doi.org/10.24432/C50P49.

The 1,797 images are the UCI test subset bundled by `sklearn.datasets.load_digits`. UCI licenses this dataset under **Creative Commons Attribution 4.0 International**: https://creativecommons.org/licenses/by/4.0/ (verified against https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits).

Forge modifications: lossless JSON serialization of 8×8 integer pixels and labels; added deterministic, stratified seed-2026 train/validation/test indices; execution divides pixels by16. The compact checkpoint was trained by Forge from random initialization. The original data are not presented as newly collected data. Asset provenance, split indices and SHA-256 are embedded in `web/public/assets/digits.json`.
