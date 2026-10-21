# Forge sharing card

The final card makes Forge's compiler the visual idea: three source operations become one copper WGSL kernel. Large DM Sans lettering establishes the name; IBM Plex Mono carries the mathematical expression. Graphite, cream, copper and blue come from the current Forge interface. The website itself is unchanged apart from sharing metadata.

- **Delivered image:** [`../../web/public/og-v1.png`](../../web/public/og-v1.png), exactly 1200 × 630 pixels, opaque 8-bit RGB PNG with an explicit sRGB declaration; 98,172 bytes (95.9 KiB).
- **Editable artwork:** [`card.svg`](card.svg), with editable text, shapes and embedded licensed fonts.
- **Authoritative export source:** [`render.py`](render.py), which produces the SVG and raster from the same geometry. Edit this source to keep regenerated files consistent.
- **Fonts:** locally vendored DM Sans and IBM Plex Mono; licenses, immutable source revision, URLs and checksums are in [`fonts/SOURCES.md`](fonts/SOURCES.md).

## Regenerate

From the repository root:

```sh
./tools/og/render.sh
```

The wrapper creates an isolated `tools/og/.venv` and installs the pinned **Pillow 12.3.0** dependency on first use. Once installed, rendering is entirely local and works offline. It loads the bundled font files directly, fixes DM Sans optical size and weight explicitly, renders at four times the final resolution, downsamples with Lanczos, and writes an optimized RGB PNG. No live simulation, browser session, animation, random frame, or cloud renderer is involved in export. The static image is copied into the production output by Vite.

The same command recreates the 600 × 315 and 300 × 158 previews and center-crop stress tests under [`proofs/`](proofs/). Use `./tools/og/render.sh --studies` to regenerate both initial directions, or `./tools/og/render.sh --polish` for the first polished direction. Run the plain command last to restore the final proofs.

## Exploration and refinement

1. [`studies/01-learned-surface.png`](studies/01-learned-surface.png) pairs a large title with an actual trained moons probability field. Its imagery emphasizes learning and retains the interface's copper/blue classes.
2. [`studies/02-compilation.png`](studies/02-compilation.png) uses a wide, diagram-led composition: operations merge into generated shader code. This direction was chosen because the compiler transformation distinguishes Forge immediately and remains understandable at small sizes.
3. [`studies/03-first-polished.png`](studies/03-first-polished.png) simplifies the diagram to mathematical symbols and makes the name dominant. The final refinement optically centers the operation symbols, enlarges and splits the descriptor for small-preview readability, strengthens the bottom sentence, and removes redundant numerical copy.

The final 1200 px export and both sharing-size previews were visually inspected. The name, operation chain and copper destination retain a clear silhouette at 300 px. The 16:9 center crop preserves essential content; the 4:3 and square stress crops clip important content. These crops document that limitation rather than implying a universal platform safe zone. The primary artwork remains designed for the requested 1200 × 630 format.

## Computational provenance

[`fixtures/moons.json`](fixtures/moons.json) records the actual Forge expression `tanh((x + bias) * scale)` for a 256-element float32 input with scalar bias and scale. Compiling with fusion disabled produces three kernels; enabling fusion produces one. The fixture includes source nodes, emitted WGSL, compiler settings, source checksums and elementwise NumPy schedule-replay verification against the unfused interpreter (zero difference). This example was **not executed on a GPU** for the card and makes **no speed claim**. The artwork abbreviates `bias` to `b` and `scale` to `s`.

The alternate learned-surface study uses the same fixture's real seed-42 moons experiment, trained for exactly 400 CPU steps, with a frozen 160 × 160 probability grid and source/data/state checksums. Recreate the computational fixture separately using the existing Forge development environment:

```sh
PYTHONPATH=src .venv/bin/python tools/og/generate_fixture.py
```

Ordinary artwork export reads the committed fixture and does not retrain a model.

## Metadata and delivery checks

[`../../web/index.html`](../../web/index.html) supplies the canonical URL and complete Open Graph and Twitter large-image metadata directly in its initial static head. Both systems reference the same absolute HTTPS image URL:

`https://dhanushkrishna4.github.io/Forge/og-v1.png`

Run the dependency-free local verifier:

```sh
python3 tools/og/verify.py
```

It checks PNG dimensions, chunk integrity, decoded scanline size, truecolor depth, opacity, sRGB, static encoding and file-size budget. It also checks required static metadata, duplicate tags, canonical URL, exact `/Forge/` casing, title/description agreement and matching image/alt fields.

After a production build, check its actual files:

```sh
npm --prefix web run build
python3 tools/og/verify.py --html web/dist/index.html --image web/dist/og-v1.png
```

After deployment, verify the public page and image:

```sh
python3 tools/og/verify.py --html web/dist/index.html --image web/dist/og-v1.png \
  --url https://dhanushkrishna4.github.io/Forge/
```

The hosted check requires HTTP 200, static HTML metadata agreement, `image/png` delivery and an exact SHA-256 match to the local PNG. On 2026-10-21, the production build passed; source HTML, built HTML and the locally served `/Forge/` page agreed; and the served image had the correct MIME type and matched the source and production PNG byte for byte. Two consecutive local renders were also byte-identical. Run the hosted command after each release to verify the public delivery separately. This documentation does not claim that a social platform has refreshed its cached preview.

Website sharing metadata and GitHub's **repository social-preview setting** are separate. This change configures the website. Upload the PNG in the repository's social-preview settings separately if that preview is desired; the file is not automatically assigned there by these tags.
