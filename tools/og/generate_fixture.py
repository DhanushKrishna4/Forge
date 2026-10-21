"""Freeze reproducible Forge computation for the Open Graph composition studies.

Run from the repository root:
    PYTHONPATH=src .venv/bin/python tools/og/generate_fixture.py

No timing or benchmark results are captured. The decision grid is the actual
class-1 probability of the trained float32 model, ordered from ymin to ymax
(row-major, with x increasing within each row). Only training samples are drawn.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from forge_ml.compiler import compile_graph, execute_plan
from forge_ml.graph import Graph
from forge_ml.interpreter import evaluate
from forge_ml.model import DEFAULT_CONFIG, Experiment


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent / "fixtures" / "moons.json"
GRID_SIZE = 160


def source_checksums():
    files = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / "src" / "forge_ml").glob("*.py"))
    }
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"sha256": digest, "files": files}


def bindings_checksum(bindings):
    """Hash named float32 values in a fixed byte order, independent of timing."""
    digest = hashlib.sha256()
    for name, value in sorted(bindings.items()):
        data = np.asarray(value, dtype="<f4")
        digest.update(name.encode() + b"\0")
        digest.update(json.dumps(list(data.shape), separators=(",", ":")).encode())
        digest.update(data.tobytes(order="C"))
    return digest.hexdigest()


def rounded(value):
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), 8)
    if isinstance(value, np.integer):
        return int(value)
    return value


def fusion_example():
    """A real three-operation DAG compiled into one pointwise WGSL kernel."""
    graph = Graph()
    x = graph.input("x", (256,))
    bias = graph.parameter("bias", ())
    scale = graph.parameter("scale", ())
    output = ((x + bias) * scale).tanh()
    fused = compile_graph(graph, {"result": output}, {"fusion": True})
    separate = compile_graph(graph, {"result": output}, {"fusion": False})
    bindings = {
        "x": np.linspace(-2.5, 2.5, 256, dtype=np.float32),
        "bias": np.array(0.25, dtype=np.float32),
        "scale": np.array(1.5, dtype=np.float32),
    }
    actual = execute_plan(fused, bindings, simulate_memory=True)["result"]
    expected = evaluate([output], bindings)[output.id]
    np.testing.assert_array_equal(actual, expected)
    assert len(separate["kernels"]) == 3 and len(fused["kernels"]) == 1
    return {
        "expression": "tanh((x + bias) * scale)",
        "dtype": "float32",
        "shape": [256],
        "sourceNodes": graph.to_dict()["nodes"],
        "sourceOperations": [
            {"id": node.id, "op": node.op, "inputs": list(node.inputs)}
            for node in graph.nodes
            if node.op not in {"input", "parameter"}
        ],
        "unfusedKernelCount": len(separate["kernels"]),
        "fusedKernelCount": len(fused["kernels"]),
        "kernel": fused["kernels"][0],
        "verification": {
            "method": "NumPy schedule replay equals unfused interpreter, elementwise",
            "maxAbsoluteDifference": float(np.max(np.abs(actual - expected))),
            "gpuExecuted": False,
        },
        "note": "Compiler output for this expression; no performance claim.",
    }


def main():
    experiment = Experiment(DEFAULT_CONFIG)
    for _ in range(4):
        experiment.cpu_step(100)
    assert experiment.step == 400

    snapshot = experiment.snapshot()
    coordinates = np.linspace(-2.5, 2.5, GRID_SIZE)
    xx, yy = np.meshgrid(coordinates, coordinates)
    grid = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    probabilities = experiment.evaluate(grid)["probabilities"][:, 1]
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()

    fixture = {
        "format": "forge-og-art-fixture",
        "version": 1,
        "provenance": {
            "generator": "tools/og/generate_fixture.py",
            "command": "PYTHONPATH=src .venv/bin/python tools/og/generate_fixture.py",
            "numpyVersion": np.__version__,
            "source": source_checksums(),
            "datasetSha256": experiment.data["checksum"],
            "parameterAndOptimizerStateSha256": bindings_checksum(
                experiment.bindings
            ),
            "training": "Exactly four calls to Experiment.cpu_step(100), NumPy CPU.",
            "displayPrecision": "All non-integer numeric values rounded to 8 decimals.",
            "note": "A fixed illustrative trained model, not a benchmark.",
        },
        "config": experiment.config,
        "step": experiment.step,
        "metrics": {
            key: snapshot["metrics"][key]
            for key in (
                "trainLoss",
                "validationLoss",
                "trainAccuracy",
                "validationAccuracy",
            )
        },
        "grid": {
            "width": GRID_SIZE,
            "height": GRID_SIZE,
            "xmin": -2.5,
            "xmax": 2.5,
            "ymin": -2.5,
            "ymax": 2.5,
            "ordering": "row-major; x ascending within rows; y ascending across rows",
            "class": 1,
            "probabilities": probabilities.tolist(),
        },
        "samples": snapshot["samples"],
        "sampleSplit": "train",
        "splitCounts": {
            key: len(ids) for key, ids in experiment.data["splits"].items()
        },
        "fusionExample": fusion_example(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(rounded(fixture), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(f"Wrote {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size:,} bytes)")
    print(json.dumps(rounded(fixture["metrics"]), sort_keys=True))
    print(f"SHA-256 {hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
