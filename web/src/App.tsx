import { useEffect, useRef, useState } from "react";
import { ForgeClient } from "./client";
import { DEFAULT_CONFIG } from "./types";
import type {
  BenchmarkReport,
  Capabilities,
  Dataset,
  DigitExample,
  ExperimentConfig,
  Sample,
  Snapshot,
  TensorInspection,
} from "./types";
import {
  bytes,
  ConfusionMatrix,
  DecisionPlot,
  DigitExamples,
  DrawingPad,
  EmptyState,
  formatNumber,
  Icon,
  LossChart,
  ModelTrace,
  ProbabilityBars,
} from "./components";
import {
  BenchmarkView,
  CompilerControls,
  GraphList,
  GraphView,
  MemoryInspector,
  MemoryView,
  TensorInspector,
} from "./InspectorViews";
import type { GraphMode } from "./InspectorViews";

type View = "playground" | "graph" | "memory" | "benchmarks";
const SETTINGS_KEY = "forge:settings:v1";
const CHECKPOINT_KEY = "forge:checkpoint:v1";
const MAX_STEPS = 5000;
const datasets: Array<{ id: Dataset; name: string; mark: string }> = [
  { id: "xor", name: "XOR", mark: "⌘" },
  { id: "moons", name: "Moons", mark: "◒" },
  { id: "spiral", name: "Spiral", mark: "◎" },
  { id: "custom", name: "Custom", mark: "+" },
  { id: "digits", name: "Digits", mark: "8" },
];
const titles: Record<View, [string, string]> = {
  playground: ["Playground", "Small networks. Real computation."],
  graph: ["Computation graph", "Follow the path from tensors to kernels."],
  memory: ["Memory planner", "Understand what stays alive, and why."],
  benchmarks: ["Benchmarks", "Measure what optimizations actually change."],
};
function storedConfig(): ExperimentConfig {
  try {
    const value = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "null");
    if (value && typeof value === "object" && Array.isArray(value.hidden))
      return {
        ...DEFAULT_CONFIG,
        ...value,
        compiler: { ...DEFAULT_CONFIG.compiler, ...value.compiler },
      };
  } catch {
    /* An unavailable browser store does not block local training. */
  }
  return structuredClone(DEFAULT_CONFIG);
}
function readCheckpoint(): unknown {
  try {
    return JSON.parse(localStorage.getItem(CHECKPOINT_KEY) || "null");
  } catch {
    return null;
  }
}
function downloadJson(value: unknown, name: string) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export default function App() {
  const [config, setConfig] = useState<ExperimentConfig>(storedConfig);
  const configRef = useRef(config);
  configRef.current = config;
  const client = useRef<ForgeClient | null>(null);
  const runId = useRef(0),
    runningRef = useRef(false),
    checkpointRef = useRef<unknown>(readCheckpoint());
  const lastCheckpointStep = useRef(-1);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const snapshotRef = useRef<Snapshot | null>(null);
  snapshotRef.current = snapshot;
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [ready, setReady] = useState(false),
    [running, setRunning] = useState(false),
    [working, setWorking] = useState(false);
  const [busy, setBusy] = useState<string | null>("Loading Python runtime");
  const [error, setError] = useState<string | null>(null),
    [notice, setNotice] = useState<string | null>(null);
  const [view, setView] = useState<View>("playground");
  const settingsRef = useRef<HTMLDialogElement>(null);
  const headingRef = useRef<HTMLHeadingElement>(null);
  const openSettings = () => settingsRef.current?.showModal();
  const closeSettings = () => settingsRef.current?.close();
  const navigate = (next: View) => {
    setView(next);
    requestAnimationFrame(() => {
      headingRef.current?.focus({ preventScroll: true });
      window.scrollTo({ top: 0, behavior: "instant" });
    });
  };
  const openNode = (id: number) => {
    setGraphMode("forward");
    setSelectedNode(id);
    navigate("graph");
  };
  const [graphMode, setGraphMode] = useState<GraphMode>("forward"),
    [selectedNode, setSelectedNode] = useState<number | null>(null),
    [kernelIndex, setKernelIndex] = useState(0);
  const [inspection, setInspection] = useState<TensorInspection | null>(null),
    [inspecting, setInspecting] = useState(false);
  const [benchmark, setBenchmark] = useState<BenchmarkReport | null>(null),
    [benchmarkRunning, setBenchmarkRunning] = useState(false);
  const [pointClass, setPointClass] = useState(0),
    [probabilities, setProbabilities] = useState<number[] | undefined>();
  const [hasCheckpoint, setHasCheckpoint] = useState(!!checkpointRef.current);
  const [showValidation, setShowValidation] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const dirty = snapshot
    ? JSON.stringify(config) !== JSON.stringify(snapshot.config)
    : false;
  const unavailable = !ready || !!busy || working;
  const currentConfig = snapshot?.config || config;
  const isDigits = currentConfig.dataset === "digits";
  const selected = snapshot?.graph.nodes.find((n) => n.id === selectedNode);

  const acceptSnapshot = (next: Snapshot) => {
    setSnapshot(next);
    snapshotRef.current = next;
    setSelectedNode((old) =>
      old !== null && next.graph.nodes.some((n) => n.id === old)
        ? old
        : (next.graph.parameterIds?.[0] ?? next.graph.nodes[0]?.id ?? null),
    );
  };
  const pause = () => {
    runningRef.current = false;
    setRunning(false);
  };
  const checkpoint = async (token: number) => {
    if (!client.current) return;
    const value = await client.current.request("export", {}, token);
    if (token !== runId.current) return;
    checkpointRef.current = value;
    setHasCheckpoint(true);
    try {
      localStorage.setItem(CHECKPOINT_KEY, JSON.stringify(value));
    } catch {
      /* The in-memory completed checkpoint remains available. */
    }
  };
  const initialize = async (next: ExperimentConfig, pretrained = false) => {
    pause();
    const token = ++runId.current;
    setBusy(
      pretrained
        ? "Loading Forge-trained checkpoint"
        : "Building the computation graph",
    );
    setError(null);
    setInspection(null);
    setProbabilities(undefined);
    try {
      const result = await client.current!.request<Snapshot>(
        "init",
        { config: next, ...(pretrained ? { checkpoint: "digits" } : {}) },
        token,
      );
      if (token !== runId.current) return;
      acceptSnapshot(result);
      setConfig(result.config);
      lastCheckpointStep.current = -1;
      setBenchmark(null);
      checkpointRef.current = null;
      setHasCheckpoint(false);
      await checkpoint(token);
    } catch (e) {
      if (token === runId.current)
        setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (token === runId.current) setBusy(null);
    }
  };
  const boot = async (restore = false) => {
    pause();
    setWorking(false);
    setReady(false);
    setError(null);
    setBusy("Loading Python runtime");
    const token = ++runId.current;
    try {
      const caps = await client.current!.request<Capabilities>(
        "boot",
        {},
        token,
      );
      if (token !== runId.current) return;
      setCapabilities(caps);
      setReady(true);
      setBusy(
        restore
          ? "Restoring completed checkpoint"
          : "Building the computation graph",
      );
      const next = await client.current!.request<Snapshot>(
        restore && checkpointRef.current ? "import" : "init",
        restore && checkpointRef.current
          ? { experiment: checkpointRef.current }
          : { config: configRef.current },
        token,
      );
      if (token !== runId.current) return;
      acceptSnapshot(next);
      setConfig(next.config);
      setInspection(null);
      setProbabilities(undefined);
      await checkpoint(token);
      if (restore) setNotice("Restored the last completed checkpoint.");
    } catch (e) {
      if (token === runId.current)
        setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (token === runId.current) setBusy(null);
    }
  };
  useEffect(() => {
    client.current = new ForgeClient();
    void boot();
    return () => {
      runningRef.current = false;
      runId.current++;
      client.current?.destroy();
    };
    // The worker is created once. Configuration is read through a ref on recovery.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(config));
    } catch {
      /* Storage is optional. */
    }
  }, [config]);
  const recover = (restore = false) => {
    client.current?.restart();
    void boot(restore);
  };
  const recoverCpu = async () => {
    pause();
    const token = ++runId.current;
    setBusy("Restoring on NumPy CPU");
    setError(null);
    setWorking(false);
    const saved = checkpointRef.current as {
      payload?: { config?: ExperimentConfig };
    } | null;
    const sameExperiment = (a: ExperimentConfig, b: ExperimentConfig) => {
      const identity = (c: ExperimentConfig) => [
        c.dataset,
        c.seed,
        c.samples,
        c.noise,
        c.hidden,
        c.activation,
        c.optimizer,
        c.learningRate,
        c.batchSize,
        c.customPoints?.map((p) => [p.x, p.y, p.label]) || [],
      ];
      return JSON.stringify(identity(a)) === JSON.stringify(identity(b));
    };
    const matchingCheckpoint =
      saved?.payload?.config &&
      snapshotRef.current &&
      sameExperiment(saved.payload.config, snapshotRef.current.config);
    try {
      const result = await client.current!.request<Snapshot>(
        matchingCheckpoint ? "import" : "init",
        matchingCheckpoint
          ? { experiment: saved, backend: "cpu" }
          : { config: { ...configRef.current, backend: "cpu" } },
        token,
      );
      if (token !== runId.current) return;
      acceptSnapshot(result);
      setConfig(result.config);
      setInspection(null);
      setReady(true);
      await checkpoint(token);
      setNotice(
        matchingCheckpoint
          ? `Recovered on NumPy CPU from completed checkpoint at step ${result.metrics.step}.`
          : "NumPy CPU is ready from the initial state; no matching completed checkpoint was available.",
      );
    } catch (e) {
      if (token === runId.current)
        setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (token === runId.current) setBusy(null);
    }
  };
  const step = async () => {
    if (!client.current || unavailable || running) return;
    const token = runId.current;
    setWorking(true);
    setError(null);
    try {
      const next = await client.current.request<Snapshot>(
        "step",
        { steps: 1 },
        token,
      );
      if (token === runId.current) {
        acceptSnapshot(next);
        await checkpoint(token);
      }
    } catch (e) {
      if (token === runId.current)
        setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (token === runId.current) setWorking(false);
    }
  };
  const train = async () => {
    if (unavailable || running || dirty) return;
    const token = runId.current;
    runningRef.current = true;
    setRunning(true);
    setError(null);
    try {
      while (runningRef.current && token === runId.current) {
        if ((snapshotRef.current?.metrics.step || 0) >= MAX_STEPS) {
          setNotice(
            `The ${MAX_STEPS.toLocaleString()}-step interactive limit was reached. Export or reset this experiment to continue.`,
          );
          break;
        }
        setWorking(true);
        const next = await client.current!.request<Snapshot>(
          "step",
          {
            steps: Math.min(
              3,
              MAX_STEPS - (snapshotRef.current?.metrics.step || 0),
            ),
          },
          token,
        );
        if (token !== runId.current) break;
        acceptSnapshot(next);
        if (
          Math.floor(next.metrics.step / 25) >
          Math.floor(lastCheckpointStep.current / 25)
        ) {
          await checkpoint(token);
          lastCheckpointStep.current = next.metrics.step;
        }
        setWorking(false);
        await new Promise((resolve) => setTimeout(resolve, 16));
      }
      if (token === runId.current && snapshotRef.current)
        await checkpoint(token);
    } catch (e) {
      if (token === runId.current)
        setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (token === runId.current) {
        pause();
        setWorking(false);
      }
    }
  };
  const exportExperiment = async () => {
    setBusy("Exporting experiment");
    setError(null);
    try {
      const value = await client.current!.request("export", {}, runId.current);
      downloadJson(
        value,
        `forge-${currentConfig.dataset}-step-${snapshot?.metrics.step || 0}.json`,
      );
      checkpointRef.current = value;
      setHasCheckpoint(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };
  const importExperiment = async (file: File) => {
    if (file.size > 5 * 1024 * 1024) {
      setError("Experiment files must be no larger than 5 MiB.");
      return;
    }
    pause();
    setBusy("Validating experiment");
    setError(null);
    const token = ++runId.current;
    try {
      const experiment = JSON.parse(await file.text());
      const next = await client.current!.request<Snapshot>(
        "import",
        { experiment },
        token,
      );
      if (token !== runId.current) return;
      acceptSnapshot(next);
      setConfig(next.config);
      setInspection(null);
      setProbabilities(undefined);
      await checkpoint(token);
      setNotice(
        `Imported ${file.name}. Parameters and optimizer state are ready to resume.`,
      );
    } catch (e) {
      if (token === runId.current)
        setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (token === runId.current) setBusy(null);
      if (fileRef.current) fileRef.current.value = "";
    }
  };
  const inspect = async () => {
    if (selectedNode === null || unavailable || running) return;
    const token = runId.current;
    setInspecting(true);
    setError(null);
    try {
      const next = await client.current!.request<TensorInspection>(
        "inspect",
        { nodeId: selectedNode },
        token,
      );
      if (token === runId.current) setInspection(next);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setInspecting(false);
    }
  };
  const predict = async (pixels: number[]) => {
    if (unavailable || running) return;
    setBusy("Predicting digit");
    setError(null);
    const token = runId.current;
    try {
      const result = await client.current!.request<{ probabilities: number[] }>(
        "predict",
        { pixels },
        token,
      );
      if (token === runId.current) setProbabilities(result.probabilities);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };
  const runBenchmark = async () => {
    if (unavailable || running) return;
    setBenchmarkRunning(true);
    setError(null);
    const token = runId.current;
    try {
      const result = await client.current!.request<BenchmarkReport>(
        "benchmark",
        { repetitions: 7 },
        token,
      );
      if (token === runId.current) setBenchmark(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBenchmarkRunning(false);
    }
  };
  const lossTest = async () => {
    setBusy("Testing device-loss recovery");
    setError(null);
    try {
      await checkpoint(runId.current);
      await client.current!.request("loss-test", {}, runId.current);
      setNotice(
        "Device-loss path exercised. Use CPU recovery to restore the last completed checkpoint.",
      );
      setError("The GPU device was deliberately lost for the recovery test.");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };
  const changeDataset = (dataset: Dataset) => {
    setConfig((c) => ({
      ...c,
      dataset,
      hidden:
        dataset === "digits"
          ? [64, 32]
          : c.dataset === "digits"
            ? [16, 16]
            : c.hidden,
      ...(dataset === "custom"
        ? {
            customPoints: c.customPoints ||
              snapshot?.samples?.map((p) => ({ ...p })) || [
                { x: -1, y: -1, label: 0 },
                { x: 1, y: 1, label: 1 },
                { x: -1, y: 1, label: 1 },
                { x: 1, y: -1, label: 0 },
                { x: 0, y: -1, label: 0 },
                { x: 0, y: 1, label: 1 },
              ],
          }
        : {}),
    }));
  };
  const updatePoints = (customPoints: Sample[]) =>
    setConfig((c) => ({ ...c, customPoints, samples: customPoints.length }));
  const selectExample = (example: DigitExample) => {
    if (example.probabilities) setProbabilities(example.probabilities);
    else void predict(example.pixels);
  };
  const selectGraphMode = (mode: GraphMode) => {
    setGraphMode(mode);
    if (mode !== "kernels") {
      const ids =
        mode === "forward"
          ? snapshot?.graph.forwardIds
          : snapshot?.graph.backwardIds;
      if (ids?.length) setSelectedNode(ids[0]);
    }
  };
  const setHidden = (i: number, n: number) =>
    setConfig((c) => ({
      ...c,
      hidden: c.hidden.map((v, j) => (j === i ? n : v)),
    }));
  const status = error
    ? "Attention needed"
    : busy ||
      (running
        ? "Training"
        : working
          ? "Finishing step"
          : ready
            ? "Ready to explore"
            : "Runtime offline");
  const noActions = unavailable || running || benchmarkRunning || inspecting;

  const trainingControls = (
    <div className="training-buttons">
      <button
        className={`button primary train-button ${running ? "is-running" : ""}`}
        disabled={
          !ready ||
          (!!busy && !running) ||
          benchmarkRunning ||
          inspecting ||
          (dirty && !running) ||
          (!running && working)
        }
        onClick={running ? pause : train}
      >
        <Icon name={running ? "pause" : "play"} size={19} />
        {running ? "Pause training" : "Train model"}
      </button>
      <button
        className="button secondary"
        disabled={noActions || dirty}
        onClick={step}
      >
        <Icon name="step" size={18} />
        Step
      </button>
      <button
        className="button icon-button"
        title="Stop after the current bounded group"
        aria-label="Stop training"
        disabled={!running}
        onClick={() => {
          pause();
          setNotice(
            "Stopped after the current bounded group. Completed state is retained.",
          );
        }}
      >
        <Icon name="stop" size={17} />
      </button>
      <button
        className="button icon-button"
        title="Reset to initial parameters"
        aria-label="Reset experiment"
        disabled={noActions}
        onClick={() => initialize(config)}
      >
        <Icon name="reset" size={18} />
      </button>
    </div>
  );
  const runtimeStatus = (
    <span
      className={`runtime-status ${running || busy ? "working" : error ? "error" : "ready"}`}
      role="status"
    >
      <i />
      {status}
    </span>
  );

  const learningCurve = (
    <div className="loss-section">
      <div className="section-heading">
        <h3>Learning curve</h3>
        <div className="legend line-legend">
          <span>
            <i className="class-zero" />
            Train
          </span>
          <span>
            <i className="class-one" />
            Validation
          </span>
        </div>
      </div>
      <LossChart history={snapshot?.history || []} />
    </div>
  );
  const predictedDigit = probabilities
    ? probabilities.indexOf(Math.max(...probabilities))
    : null;

  return (
    <div className="forge-app">
      <a className="skip-link" href="#main-content">
        Skip to workspace
      </a>
      <header className="masthead">
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            navigate("playground");
          }}
          aria-label="Forge playground"
        >
          <svg width="30" height="30" viewBox="0 0 30 30" aria-hidden="true">
            <path
              d="M5 4h21v6H11v5h12v6H11v7H5zM21 24h5v4h-5z"
              fill="currentColor"
            />
          </svg>
          <span>
            forge<span className="brand-dot">.</span>
          </span>
        </a>
        <span className="brand-description">
          THE INSPECTABLE
          <br />
          LEARNING MACHINE
        </span>
        <div className="masthead-context">
          <span>Python semantics.</span> Browser execution.
        </div>
        <div className="file-actions">
          <button
            className="button subtle"
            aria-label="Import experiment"
            onClick={() => fileRef.current?.click()}
            disabled={!ready || noActions}
          >
            <Icon name="upload" size={16} />
            <span>Import</span>
          </button>
          <button
            className="button outline"
            aria-label="Export experiment"
            onClick={exportExperiment}
            disabled={!snapshot || noActions}
          >
            <Icon name="download" size={16} />
            <span>Export</span>
          </button>
        </div>
        <input
          type="file"
          ref={fileRef}
          accept=".json,application/json"
          hidden
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void importExperiment(file);
          }}
        />
      </header>
      <div className="app-shell">
        <nav className="workspace-nav" aria-label="Workspace views">
          {(["playground", "graph", "memory", "benchmarks"] as const).map(
            (v, i) => (
              <button
                key={v}
                className={view === v ? "active" : ""}
                aria-current={view === v ? "page" : undefined}
                onClick={() => navigate(v)}
              >
                <span className="nav-number">0{i + 1}</span>
                <Icon name={["plot", "graph", "memory", "bolt"][i]} size={23} />
                <span>{v[0].toUpperCase() + v.slice(1)}</span>
              </button>
            ),
          )}
          <span className="nav-footnote">
            FLOAT32
            <br />
            STATIC SHAPES
          </span>
        </nav>
        <main className={`workspace-page view-${view}`} id="main-content">
          <div className="workspace-topline">
            <span className="eyebrow">
              {view === "playground"
                ? "01 / BUILD AN INTUITION"
                : `0${["playground", "graph", "memory", "benchmarks"].indexOf(view) + 1} / LOOK INSIDE THE MACHINE`}
            </span>
            {runtimeStatus}
          </div>
          <div className="workspace-heading">
            <h1 ref={headingRef} tabIndex={-1}>
              {view === "playground" ? (
                <>
                  See what
                  <br />
                  <em>learning does.</em>
                </>
              ) : (
                titles[view][0]
              )}
            </h1>
            <div className="workspace-intro">
              <p>
                {view === "playground"
                  ? "Train a neural network. Watch its decisions change. Follow every operation down to the GPU."
                  : titles[view][1]}
              </p>
              <span className="local-note">
                <Icon name="chip" size={15} />
                Computed here. On your device.
              </span>
            </div>
          </div>
          {error && (
            <div className="alert error" role="alert">
              <Icon name="info" />
              <span>{error}</span>
              <button onClick={recoverCpu} disabled={!!busy}>
                Use CPU
              </button>
              <button onClick={() => recover(hasCheckpoint)}>
                Restart worker{hasCheckpoint ? " & restore" : ""}
              </button>
              <button aria-label="Dismiss error" onClick={() => setError(null)}>
                <Icon name="close" size={18} />
              </button>
            </div>
          )}
          {notice && (
            <div className="alert notice" role="status">
              <Icon name="check" />
              <span>{notice}</span>
              <button
                aria-label="Dismiss notice"
                onClick={() => setNotice(null)}
              >
                <Icon name="close" size={18} />
              </button>
            </div>
          )}
          <div className="experiment-strip">
            <div className="experiment-identity">
              <span className="eyebrow">EXPERIMENT</span>
              <strong>
                {datasets.find((d) => d.id === currentConfig.dataset)?.name}
              </strong>
              <span className="experiment-meta">
                {currentConfig.hidden.join(" → ")} hidden ·{" "}
                {currentConfig.activation} ·{" "}
                {snapshot?.backend === "webgpu" ? "WebGPU" : "NumPy CPU"}
              </span>
            </div>
            <button className="button outline" onClick={openSettings}>
              <Icon name="chip" size={16} />
              Configure experiment
              <Icon name="plus" size={14} />
            </button>
          </div>
          {dirty && (
            <div className="pending-config" role="status">
              <div>
                <strong>Changes ready to apply.</strong>
                <span>
                  The current result still uses the previous configuration.
                </span>
              </div>
              <button
                className="button primary"
                disabled={!ready || noActions}
                onClick={() => initialize(config)}
              >
                Apply changes & reset
                <Icon name="arrow" size={16} />
              </button>
            </div>
          )}
          {view === "playground" ? (
            <>
              <section
                className={`learning-stage ${isDigits ? "digit-stage" : ""}`}
                aria-label="Live learning workspace"
              >
                <div className="mobile-train">{trainingControls}</div>
                <div className="decision-stage">
                  <div className="stage-heading">
                    <h2>
                      {isDigits
                        ? "Read the handwriting."
                        : "The decision surface."}
                    </h2>
                    <span className="tiny-code">
                      {isDigits
                        ? "64 INPUTS / 10 CLASSES"
                        : "2 INPUTS / 2 CLASSES"}
                    </span>
                  </div>
                  {!snapshot ? (
                    <div className="loading-plot">
                      <EmptyState
                        title={
                          busy
                            ? "Preparing your experiment"
                            : "Your runtime needs attention"
                        }
                        icon="chip"
                      >
                        {busy
                          ? "Loading Python, NumPy, and Forge. Training starts when you press Train model."
                          : "Restart the worker below to restore a completed checkpoint."}
                      </EmptyState>
                      {busy && <span className="loading-line" />}
                    </div>
                  ) : isDigits ? (
                    <div className="digit-interaction">
                      {probabilities && predictedDigit !== null && (
                        <div className="digit-feedback" role="status">
                          <span>
                            Prediction <strong>{predictedDigit}</strong>
                            <span>
                              {(probabilities[predictedDigit] * 100).toFixed(1)}
                              %
                            </span>
                          </span>
                          <a href="#digit-probabilities">
                            View probabilities
                            <Icon name="arrow" size={15} />
                          </a>
                        </div>
                      )}
                      <DigitExamples
                        examples={snapshot.digits?.examples || []}
                        onSelect={selectExample}
                      />
                      <DrawingPad onPredict={predict} disabled={noActions} />
                    </div>
                  ) : (
                    <>
                      <div className="plot-toolbar">
                        <div className="legend">
                          <span>
                            <i className="class-zero" />
                            Class 0
                          </span>
                          <span>
                            <i className="class-one" />
                            Class 1
                          </span>
                        </div>
                        <label className="check-label">
                          <input
                            type="checkbox"
                            checked={showValidation}
                            onChange={(e) =>
                              setShowValidation(e.target.checked)
                            }
                          />
                          Validation points
                        </label>
                      </div>
                      {config.dataset === "custom" && (
                        <div className="custom-tools">
                          <span>
                            Add or drag points. Changes apply on reset.
                          </span>
                          <div className="point-class">
                            <button
                              aria-pressed={pointClass === 0}
                              className={pointClass === 0 ? "selected" : ""}
                              onClick={() => setPointClass(0)}
                            >
                              <i className="class-zero" />
                              Class 0
                            </button>
                            <button
                              aria-pressed={pointClass === 1}
                              className={pointClass === 1 ? "selected" : ""}
                              onClick={() => setPointClass(1)}
                            >
                              <i className="class-one" />
                              Class 1
                            </button>
                          </div>
                        </div>
                      )}
                      <DecisionPlot
                        samples={
                          config.dataset === "custom"
                            ? config.customPoints || []
                            : snapshot.samples || []
                        }
                        validation={
                          showValidation ? snapshot.validationSamples : []
                        }
                        boundary={snapshot.boundary}
                        preview={dirty && config.dataset === "custom"}
                        editable={
                          config.dataset === "custom" && !running && !busy
                        }
                        pointClass={pointClass}
                        onChange={updatePoints}
                      />
                      <div className="plot-bottom">
                        <span>
                          <i className="mini-gradient" />
                          Model probability
                        </span>
                        <span>
                          {config.dataset === "custom"
                            ? `${config.customPoints?.length || 0} dataset points`
                            : `${snapshot.samples?.length || 0} training points`}
                        </span>
                      </div>
                    </>
                  )}
                </div>
                <div className="training-readout">
                  <div className="readout-heading">
                    <span className="eyebrow">VALIDATION ACCURACY</span>
                    <span className={`pill ${running ? "live" : ""}`}>
                      {running
                        ? "LIVE"
                        : snapshot?.metrics.step
                          ? "MEASURED"
                          : "INITIAL STATE"}
                    </span>
                  </div>
                  <div className="accuracy-value">
                    {snapshot
                      ? (snapshot.metrics.validationAccuracy * 100).toFixed(1)
                      : "—"}
                    <span>%</span>
                  </div>
                  <p className="accuracy-caption">
                    {isDigits
                      ? "Correct predictions on the validation split."
                      : "How often the model classifies unseen validation points correctly."}
                  </p>
                  <div className="loss-readout">
                    <div>
                      <span>Training loss</span>
                      <strong>
                        {formatNumber(snapshot?.metrics.trainLoss, 4)}
                      </strong>
                    </div>
                    <div>
                      <span>Validation loss</span>
                      <strong>
                        {formatNumber(snapshot?.metrics.validationLoss, 4)}
                      </strong>
                    </div>
                  </div>
                  <div className="train-command">
                    {trainingControls}
                    <div className="training-counters">
                      <span>
                        STEP{" "}
                        <b>
                          {(snapshot?.metrics.step || 0)
                            .toLocaleString()
                            .padStart(4, "0")}
                        </b>
                      </span>
                      <span>
                        ELAPSED{" "}
                        <b>
                          {formatNumber(
                            (snapshot?.metrics.elapsedMs || 0) / 1000,
                            1,
                          )}
                          s
                        </b>
                      </span>
                    </div>
                    <p className="training-hint">
                      {dirty
                        ? "Apply your changes above to start a fresh experiment."
                        : running
                          ? "Updating weights in bounded groups of three steps."
                          : snapshot?.metrics.step
                            ? "Continue training, or explore what changed inside the graph."
                            : isDigits
                              ? "Train from random weights, or load the saved checkpoint in Configure experiment."
                              : "Start training to watch the boundary form. Use Step for one update."}
                    </p>
                  </div>
                  {isDigits ? (
                    <section
                      className="prediction-section"
                      id="digit-probabilities"
                      aria-label="Prediction results"
                    >
                      <div className="section-heading">
                        <h3>Prediction probabilities</h3>
                        <span className="tiny-code">LIVE INFERENCE</span>
                      </div>
                      <ProbabilityBars values={probabilities} />
                      <p className="muted compact">
                        Select a validation example or draw a digit to inspect
                        its prediction.
                      </p>
                    </section>
                  ) : (
                    learningCurve
                  )}
                </div>
              </section>
              <section
                className="model-section"
                aria-label="Model architecture"
              >
                <div className="section-heading">
                  <div>
                    <span className="eyebrow">FROM INPUT TO INFERENCE</span>
                    <h2>The network behind the decision.</h2>
                  </div>
                  <button
                    className="text-button"
                    onClick={() => navigate("graph")}
                  >
                    Open computation graph
                    <Icon name="arrow" size={18} />
                  </button>
                </div>
                <ModelTrace snapshot={snapshot} onSelect={openNode} />
                <div className="model-facts">
                  <span>
                    <b>{snapshot?.parameterCount.toLocaleString() || "—"}</b>{" "}
                    trainable parameters
                  </span>
                  <span>
                    <b>
                      {currentConfig.optimizer === "adam"
                        ? "Adam"
                        : currentConfig.optimizer === "sgd"
                          ? "SGD"
                          : "Momentum"}
                    </b>{" "}
                    optimizer
                  </span>
                  <span>
                    <b>Cross-entropy</b> loss
                  </span>
                  <span>
                    <b>{currentConfig.learningRate}</b> learning rate
                  </span>
                </div>
              </section>
              <div className="evidence-grid">
                <section className="gradient-evidence">
                  {isDigits ? (
                    <>
                      {learningCurve}
                      {snapshot?.digits?.testAccuracy !== undefined && (
                        <div className="checkpoint-result">
                          <span className="eyebrow">
                            Reported checkpoint test
                          </span>
                          <strong>
                            {(snapshot.digits.testAccuracy * 100).toFixed(2)}%
                          </strong>
                          <p>
                            Saved evaluation metadata; not re-evaluated on
                            import.
                          </p>
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="gradient-section">
                      <div className="section-heading">
                        <h3>Gradient flow</h3>
                        <span className="tiny-code">L2 NORM</span>
                      </div>
                      <div className="gradient-total">
                        <strong>
                          {formatNumber(snapshot?.metrics.gradientNorm, 4)}
                        </strong>
                        <span>global norm</span>
                      </div>
                      {snapshot?.parameters.map((p, i) => (
                        <button
                          className="gradient-row"
                          key={p.name}
                          onClick={() => {
                            const n = snapshot.graph.nodes.find(
                              (n) => n.name === p.name,
                            );
                            if (n) {
                              openNode(n.id);
                            }
                          }}
                        >
                          <span>{p.name}</span>
                          <i>
                            <b
                              style={{
                                width: `${Math.min(100, ((p.gradientNorm ?? 0) / Math.max(0.000001, ...snapshot.parameters.map((p) => p.gradientNorm ?? 0))) * 100)}%`,
                                opacity: 0.65 + (i % 2) * 0.35,
                              }}
                            />
                          </i>
                          <code>{formatNumber(p.gradientNorm, 3)}</code>
                        </button>
                      ))}
                      <p className="muted compact">
                        Norms are measured during training. Restored checkpoints
                        show them after the next step. Select a parameter for
                        explicit gradient inspection.
                      </p>
                    </div>
                  )}
                </section>
                <section className="plan-evidence">
                  {isDigits && snapshot?.digits?.confusion ? (
                    <>
                      <div className="section-heading">
                        <h2>Confusion matrix</h2>
                        <span className="tiny-code">
                          {snapshot.digits.split}
                        </span>
                      </div>
                      <ConfusionMatrix matrix={snapshot.digits.confusion} />
                    </>
                  ) : (
                    <>
                      <span className="eyebrow">A MODEL BECOMES A PROGRAM</span>
                      <h2>
                        Inspect the work.
                        <br />
                        Account for every byte.
                      </h2>
                      <div className="plan-stat">
                        <b>{snapshot?.plan.kernels.length ?? "—"}</b>
                        <span>GPU dispatches in the compiled plan</span>
                        <button
                          className="circle-link"
                          onClick={() => {
                            setGraphMode("kernels");
                            navigate("graph");
                          }}
                          aria-label="Inspect GPU kernels"
                        >
                          <Icon name="arrow" />
                        </button>
                      </div>
                      <div className="plan-stat">
                        <b>
                          {snapshot
                            ? bytes(
                                Number(
                                  snapshot.plan.stats.allocated_bytes || 0,
                                ),
                              )
                            : "—"}
                        </b>
                        <span>
                          Planned buffer capacity · reuse{" "}
                          {currentConfig.compiler.reuse ? "on" : "off"}
                        </span>
                        <button
                          className="circle-link"
                          onClick={() => navigate("memory")}
                          aria-label="Inspect memory plan"
                        >
                          <Icon name="arrow" />
                        </button>
                      </div>
                      <p className="muted compact">
                        These are generated GPU plan figures. NumPy executes the
                        Python graph interpreter.
                      </p>
                    </>
                  )}
                </section>
              </div>
            </>
          ) : (
            <>
              <div className="secondary-session">
                <span className="tiny-code">
                  CURRENT MODEL / STEP {snapshot?.metrics.step ?? 0}
                </span>
                {trainingControls}
              </div>
              <div className={`diagnostic-workspace ${view}-workspace`}>
                {view === "graph" && (
                  <aside className="operation-index">
                    <GraphList
                      snapshot={snapshot}
                      mode={graphMode}
                      selected={selectedNode}
                      onSelect={setSelectedNode}
                      kernelIndex={kernelIndex}
                      onKernel={setKernelIndex}
                    />
                  </aside>
                )}
                <section
                  className="diagnostic-main"
                  aria-label={titles[view][0]}
                >
                  {view === "graph" ? (
                    <GraphView
                      snapshot={snapshot}
                      mode={graphMode}
                      setMode={selectGraphMode}
                      selected={selectedNode}
                      onSelect={setSelectedNode}
                      kernelIndex={kernelIndex}
                    />
                  ) : view === "memory" ? (
                    <MemoryView
                      snapshot={snapshot}
                      selected={selectedNode}
                      onSelect={setSelectedNode}
                    />
                  ) : (
                    <BenchmarkView
                      report={benchmark}
                      running={benchmarkRunning}
                      ready={ready && !noActions}
                      onRun={runBenchmark}
                      backend={currentConfig.backend}
                    />
                  )}
                </section>
                <aside className="context-panel">
                  {view === "graph" ? (
                    <TensorInspector
                      node={selected}
                      inspection={inspection}
                      loading={inspecting}
                      disabled={noActions}
                      onInspect={inspect}
                      mode={graphMode}
                      kernel={snapshot?.plan.kernels[kernelIndex]}
                    />
                  ) : view === "memory" ? (
                    <MemoryInspector
                      snapshot={snapshot}
                      selected={selectedNode}
                    />
                  ) : (
                    <>
                      <span className="eyebrow">MEASUREMENT PROTOCOL</span>
                      <h3>Context matters.</h3>
                      {[
                        [
                          "Same graph, same backend",
                          "Optimized and baseline configurations execute the same work.",
                        ],
                        [
                          "Wait for completion",
                          "GPU timings synchronize execution. Submission time alone is not a result.",
                        ],
                        [
                          "Repeat, then summarize",
                          "Warm up, measure repeatedly, and report median with variability.",
                        ],
                        [
                          "Account for boundaries",
                          "Compilation, upload, execution, and readback remain visible.",
                        ],
                      ].map(([title, body], i) => (
                        <div className="protocol-item" key={title}>
                          <span>0{i + 1}</span>
                          <div>
                            <h4>{title}</h4>
                            <p>{body}</p>
                          </div>
                        </div>
                      ))}
                      <div className="info-note">
                        Inspection instrumentation is excluded. Optimizations
                        can reduce dispatches or memory without improving
                        latency.
                      </div>
                    </>
                  )}
                </aside>
              </div>
            </>
          )}
          <div className="runtime-footer">
            <div>
              <span className="eyebrow">LOCAL BY DESIGN</span>
              <p>
                The same Forge Python package, running in your browser.
                <br />
                No account. No remote training server.
              </p>
            </div>{" "}
            <details className="runtime-details">
              <summary>Runtime & recovery</summary>
              <dl className="detail-list">
                <div>
                  <dt>Python</dt>
                  <dd>{capabilities?.python || "loading"}</dd>
                </div>
                <div>
                  <dt>NumPy</dt>
                  <dd>{capabilities?.numpy || "loading"}</dd>
                </div>
                <div>
                  <dt>Pyodide</dt>
                  <dd>{capabilities?.pyodide || "loading"}</dd>
                </div>
              </dl>
              {capabilities?.adapter && (
                <p className="muted compact">{capabilities.adapter}</p>
              )}
              <button className="text-button" onClick={() => recover(false)}>
                Restart Python worker
              </button>
              <button
                className="text-button"
                disabled={!hasCheckpoint}
                onClick={() => recover(true)}
              >
                Restore last completed checkpoint
              </button>
              {capabilities?.webgpu && (
                <button
                  className="text-button"
                  disabled={noActions}
                  onClick={lossTest}
                >
                  Test GPU device-loss path
                </button>
              )}
              <p className="muted compact">
                Restart replaces the worker. Already submitted GPU commands may
                still finish. Completed checkpoints are retained outside it.
              </p>
            </details>
          </div>
          <footer className="page-footer">
            <span>FORGE / AN INSPECTABLE ML SYSTEM</span>
            <a
              href="https://github.com/DhanushKrishna4/Forge"
              target="_blank"
              rel="noreferrer"
            >
              Explore the source
              <Icon name="arrow" size={15} />
            </a>
          </footer>
        </main>
      </div>
      <dialog
        className="settings-dialog"
        ref={settingsRef}
        aria-labelledby="settings-title"
        onClick={(e) => {
          if (e.target === e.currentTarget) closeSettings();
        }}
      >
        <div className="settings-shell">
          <div className="settings-header">
            <div>
              <span className="eyebrow">EXPERIMENT SETUP</span>
              <h2 id="settings-title">Shape the problem.</h2>
            </div>
            <button
              className="button icon-button"
              aria-label="Close experiment settings"
              onClick={closeSettings}
            >
              <Icon name="close" size={23} />
            </button>
          </div>
          <p className="settings-intro">
            Choose the data, build a network, then apply your changes to start
            from reproducible initial weights.
          </p>
          <fieldset disabled={noActions} className="settings-fields">
            {" "}
            <div className="config-section">
              <div className="section-heading">
                <h3>Dataset</h3>
                <span className="section-step">01</span>
              </div>
              <div className="dataset-grid">
                {datasets.map((d) => (
                  <button
                    key={d.id}
                    className={config.dataset === d.id ? "selected" : ""}
                    onClick={() => changeDataset(d.id)}
                    aria-pressed={config.dataset === d.id}
                  >
                    <span className="dataset-mark">{d.mark}</span>
                    <span>{d.name}</span>
                    {config.dataset === d.id && <i />}
                  </button>
                ))}
              </div>
              {config.dataset === "digits" ? (
                <div className="dataset-note">
                  <span className="tiny-code">REAL HANDWRITING · 8 × 8</span>
                  <p>
                    1,797 handwritten digits. Fixed stratified train,
                    validation, and test splits.
                  </p>
                  <button
                    className="text-button"
                    disabled={noActions}
                    onClick={() => {
                      void initialize(config, true);
                      closeSettings();
                    }}
                  >
                    Load Forge-trained checkpoint{" "}
                    <Icon name="arrow" size={13} />
                  </button>
                </div>
              ) : (
                <>
                  <label className="field-row">
                    <span>Samples</span>
                    <select
                      value={
                        config.dataset === "custom" ? "custom" : config.samples
                      }
                      disabled={config.dataset === "custom"}
                      onChange={(e) =>
                        setConfig((c) => ({
                          ...c,
                          samples: Number(e.target.value),
                        }))
                      }
                    >
                      {config.dataset === "custom" ? (
                        <option value="custom">
                          {config.customPoints?.length || 0} points
                        </option>
                      ) : (
                        [120, 240, 480, 960].map((n) => (
                          <option key={n} value={n}>
                            {n}
                          </option>
                        ))
                      )}
                    </select>
                  </label>
                  <label className="range-field">
                    <span>
                      Noise <b>{config.noise.toFixed(2)}</b>
                    </span>
                    <input
                      type="range"
                      min="0"
                      max="0.5"
                      step="0.01"
                      value={config.noise}
                      onChange={(e) =>
                        setConfig((c) => ({
                          ...c,
                          noise: Number(e.target.value),
                        }))
                      }
                      disabled={config.dataset === "custom"}
                    />
                  </label>
                  {config.dataset === "custom" && (
                    <>
                      <p className="muted compact">
                        Click the plot to add a point. Drag an existing point to
                        move it. Choose the class for new points.
                      </p>
                      <div className="point-class">
                        <button
                          className={pointClass === 0 ? "selected" : ""}
                          onClick={() => setPointClass(0)}
                        >
                          <i className="class-zero" />
                          Class 0
                        </button>
                        <button
                          className={pointClass === 1 ? "selected" : ""}
                          onClick={() => setPointClass(1)}
                        >
                          <i className="class-one" />
                          Class 1
                        </button>
                      </div>
                      <details className="point-editor">
                        <summary>Keyboard point editor</summary>
                        <button
                          className="text-button"
                          disabled={(config.customPoints?.length || 0) >= 1000}
                          onClick={() =>
                            updatePoints([
                              ...(config.customPoints || []),
                              { x: 0, y: 0, label: pointClass },
                            ])
                          }
                        >
                          Add point at origin
                        </button>
                        {config.customPoints?.map((p, i) => (
                          <div className="point-row" key={i}>
                            <span>{i}</span>
                            <input
                              aria-label={`Point ${i} x`}
                              type="number"
                              step=".1"
                              min="-2.4"
                              max="2.4"
                              value={p.x}
                              onChange={(e) =>
                                updatePoints(
                                  config.customPoints!.map((v, j) =>
                                    i === j
                                      ? { ...v, x: Number(e.target.value) }
                                      : v,
                                  ),
                                )
                              }
                            />
                            <input
                              aria-label={`Point ${i} y`}
                              type="number"
                              step=".1"
                              min="-2.4"
                              max="2.4"
                              value={p.y}
                              onChange={(e) =>
                                updatePoints(
                                  config.customPoints!.map((v, j) =>
                                    i === j
                                      ? { ...v, y: Number(e.target.value) }
                                      : v,
                                  ),
                                )
                              }
                            />
                            <button
                              aria-label={`Delete point ${i}`}
                              onClick={() =>
                                updatePoints(
                                  config.customPoints!.filter(
                                    (_, j) => i !== j,
                                  ),
                                )
                              }
                            >
                              ×
                            </button>
                          </div>
                        ))}
                      </details>
                    </>
                  )}
                </>
              )}
              <label className="field-row">
                <span>Random seed</span>
                <input
                  aria-label="Random seed"
                  type="number"
                  min="0"
                  max="2147483647"
                  value={config.seed}
                  onChange={(e) =>
                    setConfig((c) => ({ ...c, seed: Number(e.target.value) }))
                  }
                />
              </label>
            </div>
            <div className="config-section">
              <div className="section-heading">
                <h3>Network</h3>
                <span className="section-step">02</span>
              </div>
              <label className="field-row">
                <span>Hidden layers</span>
                <select
                  value={config.hidden.length}
                  onChange={(e) =>
                    setConfig((c) => ({
                      ...c,
                      hidden: Array.from(
                        { length: Number(e.target.value) },
                        (_, i) => c.hidden[i] || 16,
                      ),
                    }))
                  }
                >
                  {[1, 2, 3].map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
                </select>
              </label>
              <div className="layer-controls">
                {config.hidden.map((n, i) => (
                  <label key={i}>
                    <span>Layer {i + 1}</span>
                    <select
                      value={n}
                      onChange={(e) => setHidden(i, Number(e.target.value))}
                    >
                      {[4, 8, 16, 32, 64].map((size) => (
                        <option value={size} key={size}>
                          {size}
                        </option>
                      ))}
                    </select>
                  </label>
                ))}
              </div>
              <label className="field-row">
                <span>Activation</span>
                <select
                  value={config.activation}
                  onChange={(e) =>
                    setConfig((c) => ({
                      ...c,
                      activation: e.target
                        .value as ExperimentConfig["activation"],
                    }))
                  }
                >
                  <option value="tanh">tanh</option>
                  <option value="relu">ReLU</option>
                  <option value="sigmoid">sigmoid</option>
                </select>
              </label>
            </div>
            <div className="config-section">
              <div className="section-heading">
                <h3>Training</h3>
                <span className="section-step">03</span>
              </div>
              <label className="field-row">
                <span>Optimizer</span>
                <select
                  value={config.optimizer}
                  onChange={(e) =>
                    setConfig((c) => ({
                      ...c,
                      optimizer: e.target
                        .value as ExperimentConfig["optimizer"],
                    }))
                  }
                >
                  <option value="adam">Adam</option>
                  <option value="momentum">Momentum</option>
                  <option value="sgd">SGD</option>
                </select>
              </label>
              <label className="field-row">
                <span>Learning rate</span>
                <input
                  aria-label="Learning rate"
                  type="number"
                  min="0.00001"
                  max="1"
                  step="0.001"
                  value={config.learningRate}
                  onChange={(e) =>
                    setConfig((c) => ({
                      ...c,
                      learningRate: Number(e.target.value),
                    }))
                  }
                />
              </label>
              <label className="field-row">
                <span>Batch size</span>
                <select
                  value={config.batchSize}
                  onChange={(e) =>
                    setConfig((c) => ({
                      ...c,
                      batchSize: Number(e.target.value),
                    }))
                  }
                >
                  {[8, 16, 32, 64, 128].map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div className="config-section">
              <div className="section-heading">
                <h3>Execution</h3>
                <span className="section-step">04</span>
              </div>
              <div className="backend-selector">
                <button
                  className={config.backend === "cpu" ? "selected" : ""}
                  onClick={() => setConfig((c) => ({ ...c, backend: "cpu" }))}
                >
                  <span>NumPy</span>
                  <small>CPU</small>
                </button>
                <button
                  title={
                    capabilities?.reason ||
                    "Execute the complete training graph on WebGPU"
                  }
                  className={config.backend === "webgpu" ? "selected" : ""}
                  disabled={!capabilities?.webgpu}
                  onClick={() =>
                    setConfig((c) => ({ ...c, backend: "webgpu" }))
                  }
                >
                  <span>WebGPU</span>
                  <small>GPU</small>
                </button>
              </div>
              {capabilities && !capabilities.webgpu && (
                <p className="muted compact">
                  {capabilities.reason ||
                    "WebGPU is unavailable in this browser. NumPy is ready."}
                </p>
              )}
              <details className="compiler-disclosure" open={view === "memory"}>
                <summary>
                  Compiler passes{" "}
                  <span>
                    {Object.values(config.compiler).filter(Boolean).length}{" "}
                    enabled
                  </span>
                </summary>
                <CompilerControls
                  options={config.compiler}
                  onChange={(compiler) =>
                    setConfig((c) => ({ ...c, compiler }))
                  }
                  disabled={running || benchmarkRunning}
                />
              </details>
            </div>
          </fieldset>
          <div className="settings-apply">
            <span>
              {dirty
                ? "Unapplied configuration changes"
                : "Same seed. Same initial weights."}
            </span>
            <button
              className="button primary full-width"
              disabled={!ready || noActions}
              onClick={() => {
                void initialize(config);
                closeSettings();
              }}
            >
              <Icon name="reset" size={17} />
              {dirty ? "Apply changes & reset" : "Restore initial state"}
            </button>
          </div>
        </div>
      </dialog>
    </div>
  );
}
