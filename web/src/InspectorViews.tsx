import { useMemo, useState } from "react";
import type {
  BenchmarkReport,
  CompilerOptions,
  ExecutionPlan,
  GraphNode,
  Kernel,
  Lifetime,
  Snapshot,
  TensorInspection,
} from "./types";
import { bytes, EmptyState, formatNumber, Icon } from "./components";
import "./diagnostics.css";

export type GraphMode = "forward" | "backward" | "kernels";
export function kernelNodes(kernel: Kernel) {
  return (kernel.node_ids ||
    kernel.nodes ||
    kernel.source_nodes ||
    []) as number[];
}
export function graphNodes(snapshot: Snapshot | null, mode: GraphMode) {
  if (!snapshot) return [];
  const ids =
    mode === "forward" ? snapshot.graph.forwardIds : snapshot.graph.backwardIds;
  return ids
    ? snapshot.graph.nodes.filter((n) => ids.includes(n.id))
    : snapshot.graph.nodes;
}
export function CompilerControls({
  options,
  onChange,
  disabled,
}: {
  options: CompilerOptions;
  onChange: (options: CompilerOptions) => void;
  disabled: boolean;
}) {
  const descriptions: Record<keyof CompilerOptions, [string, string]> = {
    dce: [
      "Dead-code elimination",
      "Remove nodes not needed by requested outputs.",
    ],
    fold: ["Constant folding", "Evaluate immutable constant-only expressions."],
    cse: [
      "Common expressions",
      "Share identical expressions with ordered inputs.",
    ],
    fusion: [
      "Pointwise fusion",
      "Combine bounded pointwise regions into GPU kernels.",
    ],
    reuse: [
      "Buffer reuse",
      "Recycle scratch capacity after its final consumer.",
    ],
    tiled: [
      "Tiled matmul",
      "Use bounded workgroup tiles for GPU matrix products.",
    ],
  };
  return (
    <div className="compiler-controls">
      {(Object.keys(descriptions) as Array<keyof CompilerOptions>).map(
        (key) => (
          <label className="toggle-row" key={key} title={descriptions[key][1]}>
            <span>
              {descriptions[key][0]}
              {(key === "fusion" || key === "tiled") && <small>GPU</small>}
            </span>
            <input
              type="checkbox"
              checked={options[key]}
              disabled={disabled}
              onChange={(e) =>
                onChange({ ...options, [key]: e.target.checked })
              }
            />
            <i aria-hidden="true" />
          </label>
        ),
      )}
      <p className="muted compact">
        Fusion and tiled matmul affect generated GPU code. NumPy interprets the
        graph.
      </p>
    </div>
  );
}

export function GraphList({
  snapshot,
  mode,
  selected,
  onSelect,
  kernelIndex,
  onKernel,
}: {
  snapshot: Snapshot | null;
  mode: GraphMode;
  selected: number | null;
  onSelect: (id: number) => void;
  kernelIndex: number;
  onKernel: (index: number) => void;
}) {
  const [search, setSearch] = useState("");
  const query = search.trim().toLowerCase();
  const allNodes = graphNodes(snapshot, mode);
  const nodes = allNodes.filter((n) =>
    `${n.id} ${n.name || ""} ${n.op}`.toLowerCase().includes(query),
  );
  const kernels = snapshot?.plan.kernels || [];
  const filteredKernels = kernels
    .map((k, i) => ({ k, i }))
    .filter(({ k, i }) =>
      `k${i} ${k.id} ${k.op || ""} ${k.name || ""}`
        .toLowerCase()
        .includes(query),
    );
  const count = mode === "kernels" ? filteredKernels.length : nodes.length;
  const total = mode === "kernels" ? kernels.length : allNodes.length;
  return (
    <div className="node-browser">
      <div className="node-browser-title">
        <span className="eyebrow">
          {mode === "kernels" ? "Dispatch index" : "Operation index"}
        </span>
        <span className="tiny-code">{total}</span>
      </div>
      <label className="search-label">
        <Icon name="graph" size={15} />
        <input
          aria-label="Find an operation"
          placeholder={mode === "kernels" ? "Find kernel…" : "Find operation…"}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </label>
      <ul
        className="node-list"
        aria-label={mode === "kernels" ? "Compiled kernels" : "Graph nodes"}
      >
        {mode === "kernels"
          ? filteredKernels.map(({ k, i }) => (
              <li key={i}>
                <button
                  aria-pressed={kernelIndex === i}
                  className={`node-list-item ${kernelIndex === i ? "selected" : ""}`}
                  onClick={() => onKernel(i)}
                >
                  <span className="node-number">k{i}</span>
                  <span>
                    <b>{k.op || k.name || "kernel"}</b>
                    <small>{kernelNodes(k).length} source operations</small>
                  </span>
                  <Icon name="chevron" size={12} />
                </button>
              </li>
            ))
          : nodes.map((n) => (
              <li key={n.id}>
                <button
                  aria-pressed={selected === n.id}
                  className={`node-list-item ${selected === n.id ? "selected" : ""}`}
                  onClick={() => onSelect(n.id)}
                >
                  <span className={`node-kind ${n.kind}`}>%{n.id}</span>
                  <span>
                    <b>{n.name || n.op}</b>
                    <small>
                      {n.op} · [{n.shape.join(", ") || "scalar"}]
                    </small>
                  </span>
                  <Icon name="chevron" size={12} />
                </button>
              </li>
            ))}
      </ul>
      {!count && (
        <div className="index-empty">
          <strong>
            {query ? "No matching operations" : "No operations yet"}
          </strong>
          <p>
            {query
              ? "Try an operation name, source ID, or kernel ID."
              : "Initialize a model to build its graph."}
          </p>
          {query && (
            <button className="button subtle" onClick={() => setSearch("")}>
              Clear search
            </button>
          )}
        </div>
      )}
      <div className="list-footer tiny-code" role="status">
        {count}
        {count !== total ? ` / ${total}` : ""}{" "}
        {mode === "kernels" ? "dispatches" : "operations"}
      </div>
    </div>
  );
}

function NodeTile({
  node,
  center,
  onSelect,
}: {
  node: GraphNode;
  center?: boolean;
  onSelect: (id: number) => void;
}) {
  return (
    <button
      className={`graph-node ${center ? "focused" : ""} ${node.kind}`}
      onClick={() => onSelect(node.id)}
      aria-label={`Select operation ${node.id}, ${node.name || node.op}`}
    >
      <span className="node-identity">
        <span>%{node.id}</span>
        <small>{node.kind}</small>
      </span>
      <strong>{node.name || node.op}</strong>
      <span className="node-shape">
        {node.op} <i>·</i>{" "}
        {node.shape.length ? `[${node.shape.join(" × ")}]` : "scalar"}
      </span>
    </button>
  );
}

function PassComparison({
  plan,
  passIndex,
  selected,
  onSelect,
}: {
  plan: ExecutionPlan;
  passIndex: number;
  selected: number | null;
  onSelect: (id: number) => void;
}) {
  const pass = plan.passes[passIndex];
  if (
    !pass ||
    !Array.isArray(pass.before_nodes) ||
    !Array.isArray(pass.after_nodes)
  )
    return null;
  const source = (plan.source_nodes || plan.nodes) as GraphNode[];
  const before = pass.before_nodes as number[],
    after = pass.after_nodes as number[];
  const removed = new Set(
    Array.isArray(pass.removed) ? (pass.removed as number[]) : [],
  );
  const changed = new Set(
    Array.isArray(pass.changed) ? (pass.changed as number[]) : [],
  );
  const foldIndex = plan.passes.findIndex((p) => p.name === "fold");
  const folded = new Set((plan.passes[foldIndex]?.changed || []) as number[]);
  const title =
    String(pass.name) === "fold"
      ? "Replaced with constants"
      : String(pass.name) === "fusion"
        ? "Materialized inside fused kernels"
        : "Removed or combined";
  return (
    <div className="pass-comparison">
      <div className="pass-comparison-summary">
        <span>
          {title}: <b>{changed.size}</b>
        </span>
        <span>Measurement: {String(pass.unit || "nodes")}</span>
      </div>
      <div className="pass-state-columns">
        {[
          ["Before", before],
          ["After", after],
        ].map(([label, ids]) => (
          <div key={String(label)}>
            <div className="pass-state-title">
              <strong>{String(label)}</strong>
              <span>{(ids as number[]).length} node IDs</span>
            </div>
            <div className="pass-node-set">
              {(ids as number[]).map((id) => {
                const node = source.find((n) => n.id === id);
                const isFolded =
                  folded.has(id) &&
                  (label === "Before"
                    ? passIndex > foldIndex
                    : passIndex >= foldIndex);
                return (
                  <button
                    className={`${removed.has(id) ? "removed" : changed.has(id) ? "changed" : ""} ${selected === id ? "selected" : ""}`}
                    key={id}
                    onClick={() => onSelect(id)}
                    aria-pressed={selected === id}
                  >
                    <code>%{id}</code>
                    <span>
                      {isFolded ? "constant" : node?.op || "operation"}
                    </span>
                    {changed.has(id) && (
                      <small>{removed.has(id) ? "−" : "↻"}</small>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>
      <p className="muted compact">
        Membership at this exact pass. Highlighted operations changed; select a
        source ID to follow its original semantics and dependencies.
      </p>
    </div>
  );
}

export function GraphView({
  snapshot,
  mode,
  setMode,
  selected,
  onSelect,
  kernelIndex,
}: {
  snapshot: Snapshot | null;
  mode: GraphMode;
  setMode: (mode: GraphMode) => void;
  selected: number | null;
  onSelect: (id: number) => void;
  kernelIndex: number;
}) {
  const [selectedPass, setSelectedPass] = useState(0);
  if (!snapshot)
    return (
      <EmptyState title="The graph starts with a model">
        Initialize the Python runtime to inspect the actual forward, backward,
        and update computation.
      </EmptyState>
    );
  const nodes = snapshot.graph.nodes;
  const node =
    nodes.find((n) => n.id === selected) || graphNodes(snapshot, mode)[0];
  const inputs = node
    ? node.inputs
        .map((id) => nodes.find((n) => n.id === id))
        .filter((n): n is GraphNode => !!n)
    : [];
  const consumers = node ? nodes.filter((n) => n.inputs.includes(node.id)) : [];
  const kernel = snapshot.plan.kernels[kernelIndex];
  const pass = snapshot.plan.passes[selectedPass];
  const sourceMap = snapshot.plan.source_map as
    | Record<string, { node?: number; kernel?: number; status?: string }>
    | undefined;
  return (
    <div className="graph-view diagnostic-view">
      <div className="diagnostic-heading">
        <div>
          <span className="eyebrow">02 / Computation</span>
          <h2>Follow the signal.</h2>
          <p>
            From an operation to its dependencies, then into the code that runs
            it.
          </p>
        </div>
        <div className="diagnostic-heading-count">
          <strong>{nodes.length}</strong>
          <span>source operations</span>
        </div>
      </div>
      <div className="view-toolbar">
        <div className="segmented" aria-label="Graph direction">
          {(["forward", "backward", "kernels"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              aria-pressed={mode === m}
              className={mode === m ? "active" : ""}
            >
              {m === "kernels"
                ? "GPU kernels"
                : `${m[0].toUpperCase()}${m.slice(1)}`}
            </button>
          ))}
        </div>
        <span className="tiny-code">
          {mode === "kernels" ? "PYTHON → WGSL" : "TYPED STATIC-SHAPE IR"}
        </span>
      </div>
      {mode === "kernels" ? (
        kernel ? (
          <div className="kernel-stage">
            <div className="kernel-banner">
              <span className="kernel-id">k{kernelIndex}</span>
              <div>
                <span className="eyebrow">Compiled compute kernel</span>
                <strong>{kernel.op || kernel.name || "compute"}</strong>
                <span>
                  {kernelNodes(kernel).length} contributing source operations ·
                  dispatch [{kernel.dispatch?.join(", ") || "—"}]
                </span>
              </div>
            </div>
            <div
              className="kernel-source-nodes"
              aria-label="Kernel source operations"
            >
              {kernelNodes(kernel).map((id) => (
                <button
                  className="code-tag"
                  key={id}
                  onClick={() => {
                    setMode("forward");
                    onSelect(id);
                  }}
                >
                  %{id} {nodes.find((n) => n.id === id)?.op || ""}
                </button>
              ))}
            </div>
            <div className="code-heading">
              <span>Generated WGSL</span>
              <span className="tiny-code">float32 · compute</span>
            </div>
            <pre className="wgsl">
              <code>
                {kernel.wgsl ||
                  kernel.source ||
                  "This plan has no shader source for the selected kernel."}
              </code>
            </pre>
          </div>
        ) : (
          <EmptyState title="No kernels to display">
            Compile a model to generate GPU shaders.
          </EmptyState>
        )
      ) : node ? (
        <div className="graph-stage">
          <div className="graph-stage-label">
            <span className="tiny-code">LOCAL DEPENDENCIES</span>
            <span className="graph-direction-note">
              Input → operation → consumer
            </span>
          </div>
          <div className="dependency-graph">
            <div className="graph-column">
              <span className="graph-column-label">
                INPUTS <b>{inputs.length}</b>
              </span>
              {inputs.map((n) => (
                <NodeTile key={n.id} node={n} onSelect={onSelect} />
              ))}
              {!inputs.length && (
                <div className="graph-origin">
                  <span className="origin-dot" />
                  {node.kind === "parameter"
                    ? "Named trainable binding"
                    : node.kind === "constant"
                      ? "Immutable constant"
                      : "External binding"}
                </div>
              )}
            </div>
            <div className="graph-connector">
              <Icon name="arrow" />
            </div>
            <div className="graph-column center">
              <span className="graph-column-label">SELECTED</span>
              <NodeTile node={node} center onSelect={onSelect} />
              {consumers.length > 1 && (
                <span className="shared-label">
                  Shared by {consumers.length} consumers
                </span>
              )}
            </div>
            <div className="graph-connector">
              <Icon name="arrow" />
            </div>
            <div className="graph-column">
              <span className="graph-column-label">
                CONSUMERS <b>{consumers.length}</b>
              </span>
              {consumers.slice(0, 6).map((n) => (
                <NodeTile key={n.id} node={n} onSelect={onSelect} />
              ))}
              {consumers.length > 6 && (
                <details className="remaining-consumers">
                  <summary>{consumers.length - 6} more consumers</summary>
                  {consumers.slice(6).map((n) => (
                    <NodeTile key={n.id} node={n} onSelect={onSelect} />
                  ))}
                </details>
              )}
              {!consumers.length && (
                <div className="graph-origin">
                  <span className="origin-dot" />
                  Requested output
                </div>
              )}
            </div>
          </div>
          <div className="graph-caption">
            <Icon name="info" size={15} />
            <span>
              Select any operation to trace its inputs and consumers. This is a
              local dependency view of the source graph.
            </span>
          </div>
        </div>
      ) : null}
      <section className="pass-section" aria-label="Compiler transformations">
        <div className="diagnostic-section-heading">
          <div>
            <span className="eyebrow">From source to schedule</span>
            <h3>Compiler transformations</h3>
          </div>
          <span className="tiny-code">BEFORE → AFTER</span>
        </div>
        <div className="pass-strip">
          {snapshot.plan.passes.map((p, i) => (
            <button
              className={selectedPass === i ? "selected" : ""}
              aria-pressed={selectedPass === i}
              key={i}
              onClick={() => setSelectedPass(i)}
            >
              <span>
                <i>{String(i + 1).padStart(2, "0")}</i>
                {String(p.name).replaceAll("_", " ")}
              </span>
              <b>
                {typeof p.before === "number" ? p.before : "—"} <i>→</i>{" "}
                {typeof p.after === "number" ? p.after : "—"}
              </b>
              <small>
                {String(p.unit || "nodes")} ·{" "}
                {p.enabled === false ? "off" : "on"}
              </small>
            </button>
          ))}
        </div>
        {pass && (
          <div className="pass-detail">
            <div className="pass-detail-heading">
              <h4>{String(pass.name).replaceAll("_", " ")}</h4>
              <p>
                {String(
                  pass.detail ||
                    "This pass preserves the requested outputs and mutable parameter bindings.",
                )}
              </p>
            </div>
            {Array.isArray(pass.removed) && pass.removed.length > 0 && (
              <div className="kernel-source-nodes">
                {pass.removed.map((id) => (
                  <button
                    key={String(id)}
                    className="code-tag"
                    onClick={() => onSelect(Number(id))}
                  >
                    %{String(id)} removed
                  </button>
                ))}
              </div>
            )}
            <PassComparison
              plan={snapshot.plan}
              passIndex={selectedPass}
              selected={selected}
              onSelect={onSelect}
            />
          </div>
        )}
        {node && sourceMap?.[String(node.id)] && (
          <div className="source-status">
            <span>Source %{node.id}</span>
            <strong>{sourceMap[String(node.id)].status}</strong>
            <span>
              {sourceMap[String(node.id)].kernel != null
                ? `→ kernel ${sourceMap[String(node.id)].kernel}`
                : ""}
            </span>
          </div>
        )}
      </section>
    </div>
  );
}

export function TensorInspector({
  node,
  inspection,
  loading,
  disabled = false,
  onInspect,
  mode,
  kernel,
}: {
  node?: GraphNode;
  inspection: TensorInspection | null;
  loading: boolean;
  disabled?: boolean;
  onInspect: () => void;
  mode: GraphMode;
  kernel?: Kernel;
}) {
  if (mode === "kernels")
    return (
      <div className="diagnostic-inspector">
        <div className="inspector-header">
          <span className="eyebrow">Compiled execution</span>
          <h3>Kernel details</h3>
        </div>
        {kernel ? (
          <>
            <dl className="detail-list">
              <div>
                <dt>Kernel</dt>
                <dd>{String(kernel.id)}</dd>
              </div>
              <div>
                <dt>Operation</dt>
                <dd>{kernel.op || "compute"}</dd>
              </div>
              <div>
                <dt>Source nodes</dt>
                <dd>{kernelNodes(kernel).length}</dd>
              </div>
              <div>
                <dt>Dispatch</dt>
                <dd>[{kernel.dispatch?.join(", ")}]</dd>
              </div>
            </dl>
            <div className="info-note">
              WGSL comes from the Python compiler. JavaScript creates buffers
              and pipelines, then dispatches this plan.
            </div>
            <h4>Fusion boundaries</h4>
            <p className="muted compact">
              Matrix products and reductions remain explicit dispatches.
              Compatible pointwise operations can share one bounded kernel.
            </p>
          </>
        ) : (
          <p className="muted">Select a kernel.</p>
        )}
      </div>
    );
  return (
    <div className="diagnostic-inspector">
      <div className="inspector-header">
        <span className="eyebrow">Tensor inspector</span>
        <h3>
          {node
            ? node.name || `%${node.id} · ${node.op}`
            : "Select an operation"}
        </h3>
      </div>
      {node && (
        <>
          <dl className="detail-list">
            <div>
              <dt>Identity</dt>
              <dd>%{node.id}</dd>
            </div>
            <div>
              <dt>Operation</dt>
              <dd>{node.op}</dd>
            </div>
            <div>
              <dt>Shape</dt>
              <dd>
                {node.shape.length ? `[${node.shape.join(", ")}]` : "scalar"}
              </dd>
            </div>
            <div>
              <dt>Type</dt>
              <dd>{node.dtype}</dd>
            </div>
            <div>
              <dt>Kind</dt>
              <dd>{node.kind}</dd>
            </div>
            <div>
              <dt>Inputs</dt>
              <dd>{node.inputs.map((n) => `%${n}`).join(", ") || "none"}</dd>
            </div>
          </dl>
          <button
            className="button secondary full-width"
            disabled={loading || disabled}
            onClick={onInspect}
          >
            <Icon name="graph" size={15} />
            {loading ? "Inspecting…" : "Capture tensor & gradient"}
          </button>
          <p className="muted compact">
            An explicit instrumented execution. Capture may allocate extra
            tensors and is excluded from benchmark timing.
          </p>
          {loading && (
            <p className="capture-status" role="status">
              Executing instrumented graph…
            </p>
          )}
          {inspection && inspection.nodeId === node.id && (
            <div className="tensor-values">
              {inspection.unavailable && (
                <div className="info-note">{inspection.unavailable}</div>
              )}
              {inspection.instrumentation && (
                <p className="muted compact">{inspection.instrumentation}</p>
              )}
              {inspection.values && (
                <>
                  <h4>
                    Values{" "}
                    <span>
                      {inspection.totalElements &&
                      inspection.totalElements > inspection.values.length
                        ? `first ${inspection.values.length} / ${inspection.totalElements}`
                        : inspection.values.length}
                    </span>
                  </h4>
                  <pre>
                    {inspection.values
                      .map((v) => formatNumber(v, 5))
                      .join("  ")}
                  </pre>
                </>
              )}
              {inspection.gradient && (
                <>
                  <h4>Gradient</h4>
                  <pre>
                    {inspection.gradient
                      .map((v) => formatNumber(v, 5))
                      .join("  ")}
                  </pre>
                </>
              )}
              {!inspection.gradient && (
                <p className="muted compact">
                  No gradient was captured for this tensor. Integer labels and
                  constants are nondifferentiable bindings.
                </p>
              )}
            </div>
          )}
          {node.attrs && Object.keys(node.attrs).length > 0 && (
            <>
              <h4>Attributes</h4>
              <pre className="attribute-values">
                {JSON.stringify(node.attrs, null, 2).slice(0, 1500)}
              </pre>
            </>
          )}
        </>
      )}
    </div>
  );
}

function normalizeLifetime(lifetime: Lifetime) {
  return {
    ...lifetime,
    node: Number(lifetime.node ?? lifetime.tensor ?? lifetime.tensor_id),
    start: Number(lifetime.start ?? lifetime.first ?? 0),
    end: Number(lifetime.end ?? lifetime.last ?? 0),
    buffer: lifetime.buffer ?? lifetime.buffer_id,
    kind: lifetime.kind || "scratch",
  };
}
export function MemoryView({
  snapshot,
  selected,
  onSelect,
}: {
  snapshot: Snapshot | null;
  selected: number | null;
  onSelect: (id: number) => void;
}) {
  const [kindFilter, setKindFilter] = useState("all");
  const [search, setSearch] = useState("");
  const all = useMemo(() => {
    const lifetimes = snapshot?.plan.lifetimes || [];
    return (
      Array.isArray(lifetimes) ? lifetimes : Object.values(lifetimes)
    ).map(normalizeLifetime);
  }, [snapshot]);
  if (!snapshot)
    return (
      <EmptyState title="See where every byte goes" icon="memory">
        Initialize a model to inspect its complete forward, backward, and
        optimizer memory schedule.
      </EmptyState>
    );
  const stats = snapshot.plan.stats;
  const maxEnd = Math.max(1, ...all.map((l) => l.end));
  const kinds = [...new Set(all.map((l) => l.kind))];
  const query = search.trim().toLowerCase();
  const rows = all.filter(
    (l) =>
      (kindFilter === "all" || l.kind === kindFilter) &&
      `${l.node} ${snapshot.graph.nodes.find((n) => n.id === l.node)?.name || ""}`
        .toLowerCase()
        .includes(query),
  );
  const allocated = Number(stats.allocated_bytes || 0),
    logical = Number(stats.logical_bytes || 0),
    saved = Number(stats.saved_scratch_bytes || 0);
  return (
    <div className="memory-view diagnostic-view">
      <div className="diagnostic-heading">
        <div>
          <span className="eyebrow">03 / Memory</span>
          <h2>Every byte has a lifetime.</h2>
          <p>
            The full schedule, from forward execution through gradients and
            updates.
          </p>
        </div>
        <span className="pill">
          {snapshot.config.compiler.reuse ? "Reuse enabled" : "Reuse disabled"}
        </span>
      </div>
      <div className="allocation-summary">
        <div className="allocation-primary">
          <span>Owned allocation</span>
          <strong>{bytes(allocated)}</strong>
          <small>Planned buffer capacity</small>
        </div>
        <div>
          <span>Logical tensors</span>
          <strong>{bytes(logical)}</strong>
          <small>Sum of logical tensor bytes</small>
        </div>
        <div className="allocation-saved">
          <span>Scratch saved</span>
          <strong>{bytes(saved)}</strong>
          <small>Compared with no reuse</small>
        </div>
      </div>
      <div className="memory-explainer">
        <Icon name="info" size={17} />
        <p>
          Bars extend through each tensor’s final consumer, including backward
          execution. Matching buffer IDs share capacity only when lifetimes do
          not overlap. These are compiled plan figures, not total browser or
          physical GPU memory.
        </p>
      </div>
      <div className="diagnostic-section-heading timeline-title">
        <div>
          <span className="eyebrow">Execution schedule</span>
          <h3>Where storage is shared</h3>
        </div>
        <span className="tiny-code">{all.length} TOTAL LIFETIMES</span>
      </div>
      <div className="timeline-toolbar">
        <select
          value={kindFilter}
          onChange={(e) => setKindFilter(e.target.value)}
          aria-label="Filter tensor lifetimes"
        >
          <option value="all">All buffer kinds</option>
          {kinds.map((k) => (
            <option value={k} key={k}>
              {k}
            </option>
          ))}
        </select>
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Tensor ID or name"
          aria-label="Search tensor lifetimes"
        />
        <span className="tiny-code" role="status">
          {rows.length} tensors
        </span>
      </div>
      <div
        className="timeline-scroll"
        tabIndex={0}
        role="region"
        aria-label="Tensor lifetime schedule"
      >
        <div className="timeline">
          <div className="timeline-header">
            <span>Tensor / buffer</span>
            <div>
              {[0, 0.25, 0.5, 0.75, 1].map((t) => (
                <span style={{ left: `${t * 100}%` }} key={t}>
                  {Math.round(maxEnd * t)}
                </span>
              ))}
            </div>
            <span>Capacity</span>
          </div>
          <div className="timeline-rows">
            {rows.map((l, i) => (
              <button
                key={i}
                className={`timeline-row ${selected === l.node ? "selected" : ""}`}
                onClick={() => onSelect(l.node)}
                aria-pressed={selected === l.node}
                aria-label={`Tensor ${l.node}, buffer ${l.buffer}, lifetime ${l.start} through ${l.end}`}
              >
                <span>
                  <b>%{l.node}</b>
                  <small>
                    b{l.buffer} · {l.kind}
                  </small>
                </span>
                <div className="timeline-track">
                  <i
                    className={`lifetime-bar kind-${l.kind}`}
                    style={{
                      left: `${(l.start / (maxEnd + 1)) * 100}%`,
                      width: `${Math.max(0.5, ((l.end - l.start + 1) / (maxEnd + 1)) * 100)}%`,
                      filter: `hue-rotate(${l.kind === "scratch" ? (Number(l.buffer) % 7) * 5 : 0}deg)`,
                    }}
                  />
                </div>
                <span className="tiny-code">
                  {bytes(Number(l.capacity || l.bytes || 0))}
                </span>
              </button>
            ))}
          </div>
          <div className="timeline-bottom">Scheduled dispatch index →</div>
        </div>
      </div>
      {!rows.length && (
        <div className="index-empty">
          <strong>No matching lifetimes</strong>
          <p>Choose another buffer kind or clear your search.</p>
          <button
            className="button subtle"
            onClick={() => {
              setKindFilter("all");
              setSearch("");
            }}
          >
            Show all lifetimes
          </button>
        </div>
      )}
    </div>
  );
}
export function MemoryInspector({
  snapshot,
  selected,
}: {
  snapshot: Snapshot | null;
  selected: number | null;
}) {
  const life = snapshot
    ? (Array.isArray(snapshot.plan.lifetimes)
        ? snapshot.plan.lifetimes
        : Object.values(snapshot.plan.lifetimes)
      )
        .map(normalizeLifetime)
        .find((l) => l.node === selected)
    : undefined;
  const buffer = snapshot?.plan.buffers.find((b) => b.id === life?.buffer);
  return (
    <div className="diagnostic-inspector">
      <div className="inspector-header">
        <span className="eyebrow">Allocation inspector</span>
        <h3>{life ? `Tensor %${life.node}` : "Select a lifetime"}</h3>
      </div>
      {life && (
        <dl className="detail-list">
          <div>
            <dt>Buffer</dt>
            <dd>b{String(life.buffer)}</dd>
          </div>
          <div>
            <dt>Kind</dt>
            <dd>{String(life.kind)}</dd>
          </div>
          <div>
            <dt>First / last use</dt>
            <dd>
              {life.start} / {life.end}
            </dd>
          </div>
          <div>
            <dt>Tensor bytes</dt>
            <dd>{bytes(Number(life.bytes))}</dd>
          </div>
          <div>
            <dt>Capacity</dt>
            <dd>{bytes(Number(life.capacity || life.bytes))}</dd>
          </div>
        </dl>
      )}
      {buffer && (
        <>
          <h4>Shares this buffer</h4>
          <div className="kernel-source-nodes">
            {(Array.isArray(buffer.nodes) ? buffer.nodes : []).map((id) => (
              <span className="code-tag" key={String(id)}>
                %{String(id)}
              </span>
            ))}
          </div>
        </>
      )}
      <div className="info-note">
        Inputs, parameters, optimizer state, and requested outputs remain
        protected. A kernel never overlaps its input and output storage.
      </div>
      <h4>Other owned resources</h4>
      <p className="muted compact">
        Readback staging is allocated for explicit host reads and then released.
        Instrumented snapshots add temporary storage outside this plan.
      </p>
    </div>
  );
}

export function BenchmarkView({
  report,
  running,
  onRun,
  ready,
  backend,
}: {
  report: BenchmarkReport | null;
  running: boolean;
  onRun: () => void;
  ready: boolean;
  backend: string;
}) {
  const save = () => {
    if (!report) return;
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = "forge-benchmark.json";
    a.click();
    URL.revokeObjectURL(url);
  };
  const maxMedian = Math.max(
    0,
    ...(report?.rows.map((row) => row.medianMs) || []),
  );
  return (
    <div className="benchmark-view diagnostic-view" aria-busy={running}>
      <div className="diagnostic-heading">
        <div>
          <span className="eyebrow">04 / Benchmarks</span>
          <h2>Make the tradeoffs measurable.</h2>
          <p>The same work. Different plans. Measurements from this machine.</p>
        </div>
        <span className="pill">
          {backend === "webgpu" ? "WebGPU" : "NumPy CPU"}
        </span>
      </div>
      <div className="benchmark-setup">
        <p>
          Compare optimizations enabled and disabled. Small workloads can be
          slower on a GPU; dispatch, transfer, and compilation costs all count.
        </p>
        <div className="benchmark-actions">
          <button
            className="button primary"
            disabled={!ready || running}
            onClick={onRun}
          >
            <Icon name="bolt" size={17} />
            {running ? "Measuring…" : "Run benchmark suite"}
          </button>
          {report && (
            <button className="button subtle" onClick={save}>
              <Icon name="download" size={16} />
              Export JSON
            </button>
          )}
        </div>
        <span className="benchmark-protocol">
          Warm-up + repeated synchronized measurements
        </span>
      </div>
      {running && (
        <div className="benchmark-running" role="status">
          <span className="measurement-pulse" />
          <div>
            <strong>Measuring the current backend</strong>
            <span>
              Running warm-up and synchronized repetitions. This may take a
              moment.{report ? " Previous results remain visible below." : ""}
            </span>
          </div>
        </div>
      )}
      {!report ? (
        <div className="benchmark-fixtures">
          {[
            ["Elementwise chain", "Pointwise fusion & dispatch count"],
            ["Shared expressions", "Common-subexpression elimination"],
            ["Scratch lifetimes", "Safe buffer reuse across consumers"],
            ["Matrix multiplication", "Baseline & bounded tiled kernels"],
            ["MLP training step", "Forward, backward & optimizer update"],
          ].map(([title, detail], i) => (
            <div key={title}>
              <span className="fixture-number">0{i + 1}</span>
              <div>
                <h3>{title}</h3>
                <p>{detail}</p>
              </div>
              <span className="fixture-status">
                {running ? "Suite running" : "Awaiting measurement"}
              </span>
            </div>
          ))}
        </div>
      ) : (
        <>
          <div className="measurement-note">
            <Icon name="check" size={16} />
            <span>{report.timingMethod}</span>
          </div>
          <div className="diagnostic-section-heading">
            <div>
              <span className="eyebrow">Synchronized execution</span>
              <h3>Time per fixture</h3>
            </div>
            <span className="tiny-code">MEDIAN / MS</span>
          </div>
          <div
            className="timing-chart"
            aria-label="Measured median execution times"
          >
            {report.rows.map((row, i) => (
              <div className="timing-row" key={i}>
                <div>
                  <strong>{row.name}</strong>
                  <span>{row.configuration}</span>
                </div>
                <div className="timing-track">
                  <i
                    style={{
                      width: `${maxMedian > 0 ? (row.medianMs / maxMedian) * 100 : 0}%`,
                    }}
                  />
                </div>
                <span>
                  {formatNumber(row.medianMs)} <small>ms</small>
                </span>
              </div>
            ))}
          </div>
          <div className="diagnostic-section-heading">
            <h3>Measurement details</h3>
            <span className="tiny-code">REPEATED SAMPLES</span>
          </div>
          <p className="muted compact">
            Dispatch and scratch columns describe generated GPU plans. NumPy
            runs the Python graph interpreter; it does not dispatch GPU kernels
            or use the GPU buffer plan.
          </p>
          <div
            className="table-scroll"
            tabIndex={0}
            role="region"
            aria-label="Benchmark measurement details"
          >
            <table className="benchmark-table">
              <thead>
                <tr>
                  <th>Fixture / configuration</th>
                  <th>Median</th>
                  <th>Range / deviation</th>
                  <th>GPU plan dispatches</th>
                  <th>Planned GPU scratch</th>
                </tr>
              </thead>
              <tbody>
                {report.rows.map((row, i) => (
                  <tr key={i}>
                    <td>
                      <strong>{row.name}</strong>
                      <span>
                        {row.configuration} · {row.backend} · n=
                        {row.repetitions}
                      </span>
                    </td>
                    <td className="measured">
                      {formatNumber(row.medianMs)} <small>ms</small>
                    </td>
                    <td>
                      {row.minMs !== undefined && row.maxMs !== undefined
                        ? `${formatNumber(row.minMs)}–${formatNumber(row.maxMs)}`
                        : "—"}
                      {row.stdMs !== undefined && (
                        <span>σ {formatNumber(row.stdMs)}</span>
                      )}
                    </td>
                    <td>{row.kernels ?? "—"}</td>
                    <td>
                      {row.scratchBytes === undefined
                        ? "—"
                        : bytes(row.scratchBytes)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="diagnostic-section-heading">
            <h3>Measured phases</h3>
            <span className="tiny-code">MILLISECONDS</span>
          </div>
          <div
            className="table-scroll"
            tabIndex={0}
            role="region"
            aria-label="Benchmark phase timings"
          >
            <table className="benchmark-table phase-table">
              <thead>
                <tr>
                  <th>Configuration</th>
                  <th>Compile</th>
                  <th>Upload</th>
                  <th>Execute</th>
                  <th>Readback</th>
                </tr>
              </thead>
              <tbody>
                {report.rows.map((row, i) => (
                  <tr key={i}>
                    <td>
                      {row.name}
                      <span>{row.configuration}</span>
                    </td>
                    <td>{formatNumber(row.compilationMs)}</td>
                    <td>{formatNumber(row.uploadMs)}</td>
                    <td>{formatNumber(row.executionMs)}</td>
                    <td>{formatNumber(row.readbackMs)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <details className="environment-details">
            <summary>Recorded environment</summary>
            <pre>{JSON.stringify(report.environment, null, 2)}</pre>
          </details>
          {report.notes?.map((note, i) => (
            <p className="muted compact" key={i}>
              {note}
            </p>
          ))}
        </>
      )}
    </div>
  );
}
