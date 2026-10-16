import { useId, useRef, useState } from "react";
import type { CSSProperties, PointerEvent as ReactPointerEvent } from "react";
import type {
  DecisionBoundary,
  DigitExample,
  HistoryPoint,
  Sample,
  Snapshot,
} from "./types";
import "./visualizations.css";

export function Icon({ name, size = 18 }: { name: string; size?: number }) {
  const paths: Record<string, string> = {
    play: "M8 5l11 7-11 7V5Z",
    pause: "M8 5v14M16 5v14",
    step: "M6 5l10 7-10 7V5ZM19 5v14",
    reset: "M3 11a9 9 0 1 1 2 7M3 4v7h7",
    download: "M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5",
    upload: "M12 16V4m-5 5 5-5 5 5M4 16v5h16v-5",
    check: "m5 12 4 4L19 6",
    close: "m6 6 12 12M6 18 18 6",
    chevron: "m9 5 7 7-7 7",
    plus: "M12 5v14M5 12h14",
    graph: "M5 5h5v5H5zM14 14h5v5h-5zM14 3h5v5h-5zM8 10v7h6M10 7h4",
    plot: "M4 3v17h17M7 14l4-6 4 3 5-6",
    chip: "M7 7h10v10H7zM9 2v5m6-5v5M9 17v5m6-5v5M2 9h5m-5 6h5m10-6h5m-5 6h5",
    memory: "M3 6h18v12H3zM7 10v4m5-4v4m5-4v4M7 18v3m5-3v3m5-3v3",
    info: "M12 11v6m0-10v1M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z",
    bolt: "m13 2-9 12h7l-1 8 10-13h-7l0-7Z",
    arrow: "M4 12h16m-5-5 5 5-5 5",
    folder: "M3 6h7l2 3h9v11H3V6Z",
    stop: "M6 6h12v12H6z",
  };
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={paths[name] || paths.info} />
    </svg>
  );
}

export function formatNumber(value?: number | null, digits = 3) {
  return value == null || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}
export function bytes(value: number) {
  return value >= 1048576
    ? `${(value / 1048576).toFixed(2)} MiB`
    : value >= 1024
      ? `${(value / 1024).toFixed(1)} KiB`
      : `${value} B`;
}
export function EmptyState({
  title,
  children,
  icon = "graph",
}: {
  title: string;
  children: React.ReactNode;
  icon?: string;
}) {
  return (
    <div className="empty-state">
      <span className="empty-icon">
        <Icon name={icon} size={27} />
      </span>
      <h3>{title}</h3>
      <p>{children}</p>
    </div>
  );
}

export function DecisionPlot({
  samples,
  validation = [],
  boundary,
  editable,
  pointClass,
  onChange,
  preview,
}: {
  samples: Sample[];
  validation?: Sample[];
  boundary?: DecisionBoundary;
  editable?: boolean;
  pointClass: number;
  onChange?: (samples: Sample[]) => void;
  preview?: boolean;
}) {
  const ref = useRef<SVGSVGElement>(null);
  const plotId = useId().replace(/:/g, "");
  const drag = useRef<number | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const bounds = boundary || { xmin: -2.4, xmax: 2.4, ymin: -2.4, ymax: 2.4 };
  const inset = 36,
    size = 520,
    plotSize = size - 2 * inset;
  const tx = (x: number) =>
    inset + ((x - bounds.xmin) / (bounds.xmax - bounds.xmin)) * plotSize;
  const ty = (y: number) =>
    size - inset - ((y - bounds.ymin) / (bounds.ymax - bounds.ymin)) * plotSize;
  const position = (event: ReactPointerEvent<SVGSVGElement>) => {
    const rect = ref.current!.getBoundingClientRect();
    const px = ((event.clientX - rect.left) / rect.width) * size;
    const py = ((event.clientY - rect.top) / rect.height) * size;
    return {
      x: Math.max(
        bounds.xmin,
        Math.min(
          bounds.xmax,
          bounds.xmin + ((px - inset) / plotSize) * (bounds.xmax - bounds.xmin),
        ),
      ),
      y: Math.max(
        bounds.ymin,
        Math.min(
          bounds.ymax,
          bounds.ymax - ((py - inset) / plotSize) * (bounds.ymax - bounds.ymin),
        ),
      ),
    };
  };
  const pointerDown = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (!editable || !onChange) return;
    const pos = position(event);
    const nearest = samples.findIndex(
      (p) => Math.hypot(tx(p.x) - tx(pos.x), ty(p.y) - ty(pos.y)) < 11,
    );
    if (nearest >= 0) {
      drag.current = nearest;
      setSelected(nearest);
    } else if (samples.length < 1000) {
      drag.current = samples.length;
      setSelected(samples.length);
      onChange([...samples, { ...pos, label: pointClass }]);
    }
    event.currentTarget.setPointerCapture(event.pointerId);
  };
  const ticks = Array.from({ length: 5 }, (_, i) => i / 4);
  return (
    <div
      className={`decision-plot ${editable ? "editable" : ""} ${preview ? "is-preview" : ""}`}
    >
      <div className="decision-surface">
        <svg
          ref={ref}
          viewBox={`0 0 ${size} ${size}`}
          role="img"
          aria-label={`${preview ? "Draft dataset" : "Decision boundary"} with ${samples.length} data points and ${validation.length} visible validation points. Copper is class zero and blue is class one. Horizontal axis x one, vertical axis x two.`}
          onPointerDown={pointerDown}
          onPointerMove={(event) => {
            if (drag.current === null || !onChange) return;
            const index = drag.current,
              pos = position(event);
            onChange(
              samples.map((p, i) => (i === index ? { ...p, ...pos } : p)),
            );
          }}
          onPointerUp={() => {
            drag.current = null;
          }}
          onPointerCancel={() => {
            drag.current = null;
          }}
        >
          <defs>
            <clipPath id={`${plotId}-clip`}>
              <rect
                x={inset}
                y={inset}
                width={plotSize}
                height={plotSize}
                rx="2"
              />
            </clipPath>
            <pattern
              id={`${plotId}-grid`}
              width={plotSize / 16}
              height={plotSize / 16}
              patternUnits="userSpaceOnUse"
              x={inset}
              y={inset}
            >
              <path
                d={`M ${plotSize / 16} 0 H 0 V ${plotSize / 16}`}
                fill="none"
                stroke="#d6d8d0"
                strokeOpacity="0.055"
                strokeWidth="1"
              />
            </pattern>
          </defs>
          <rect
            x={inset}
            y={inset}
            width={plotSize}
            height={plotSize}
            fill="#202524"
            stroke="#a0a69f"
            strokeOpacity=".12"
          />
          {boundary && !preview && (
            <g clipPath={`url(#${plotId}-clip)`}>
              {boundary.probabilities.map((p, i) => {
                const mix = Math.max(0, Math.min(1, p));
                const rgb = [
                  Math.round(177 * (1 - mix) + 83 * mix),
                  Math.round(103 * (1 - mix) + 142 * mix),
                  Math.round(71 * (1 - mix) + 166 * mix),
                ];
                return (
                  <rect
                    key={i}
                    x={
                      inset + ((i % boundary.width) * plotSize) / boundary.width
                    }
                    y={
                      size -
                      inset -
                      ((Math.floor(i / boundary.width) + 1) * plotSize) /
                        boundary.height
                    }
                    width={plotSize / boundary.width + 0.3}
                    height={plotSize / boundary.height + 0.3}
                    fill={`rgb(${rgb.join(",")})`}
                    opacity={0.26 + Math.abs(mix - 0.5) * 0.36}
                  />
                );
              })}
            </g>
          )}
          <rect
            x={inset}
            y={inset}
            width={plotSize}
            height={plotSize}
            fill={`url(#${plotId}-grid)`}
          />
          {ticks.map((t, i) => (
            <g key={i}>
              <line
                x1={inset + t * plotSize}
                x2={inset + t * plotSize}
                y1={inset}
                y2={size - inset}
                stroke="#b5c5b6"
                strokeOpacity=".10"
              />
              <line
                x1={inset}
                x2={size - inset}
                y1={inset + t * plotSize}
                y2={inset + t * plotSize}
                stroke="#b5c5b6"
                strokeOpacity=".10"
              />
            </g>
          ))}
          <g clipPath={`url(#${plotId}-clip)`}>
            {validation.map((p, i) => (
              <circle
                key={`v${i}`}
                cx={tx(p.x)}
                cy={ty(p.y)}
                r="3.4"
                fill="none"
                stroke={p.label === 0 ? "#eaaa83" : "#95bac9"}
                strokeWidth="1.3"
                opacity=".6"
              />
            ))}
            {samples.map((p, i) => (
              <circle
                key={i}
                cx={tx(p.x)}
                cy={ty(p.y)}
                r={editable && selected === i ? 5.6 : 3.8}
                fill={p.label === 0 ? "#e9ac84" : "#9dbfcd"}
                stroke={editable && selected === i ? "#fff" : "#242828"}
                strokeWidth={editable && selected === i ? 2 : 1.4}
              />
            ))}
          </g>
        </svg>
        <div className="decision-axes" aria-hidden="true">
          {ticks.map((t, i) => (
            <span key={i}>
              <span
                className="decision-x-tick"
                style={{ left: `${((inset + t * plotSize) / size) * 100}%` }}
              >
                {(bounds.xmin + t * (bounds.xmax - bounds.xmin)).toFixed(1)}
              </span>
              <span
                className="decision-y-tick"
                style={{
                  top: `${((size - inset - t * plotSize) / size) * 100}%`,
                }}
              >
                {(bounds.ymin + t * (bounds.ymax - bounds.ymin)).toFixed(1)}
              </span>
            </span>
          ))}
          <span className="decision-axis-name x">x₁</span>
          <span className="decision-axis-name y">x₂</span>
        </div>
      </div>
      {editable && (
        <div className="plot-edit-note">
          <span>
            <Icon name="plus" size={13} /> Click to add · drag to move
          </span>
          {selected !== null && (
            <button
              className="text-button"
              onClick={() => {
                onChange?.(samples.filter((_, i) => i !== selected));
                setSelected(null);
              }}
            >
              Delete selected
            </button>
          )}
        </div>
      )}
    </div>
  );
}

export function LossChart({ history }: { history: HistoryPoint[] }) {
  const valid = history.filter(
    (p) => Number.isFinite(p.trainLoss) && Number.isFinite(p.validationLoss),
  );
  const width = 640,
    height = 160,
    left = 5,
    right = 5,
    top = 12,
    bottom = 12;
  const maxStep = Math.max(1, ...valid.map((p) => p.step));
  const maxValue =
    Math.max(0.01, ...valid.flatMap((p) => [p.trainLoss, p.validationLoss])) *
    1.1;
  const x = (step: number) => left + (step / maxStep) * (width - left - right);
  const y = (value: number) =>
    top + (1 - value / maxValue) * (height - top - bottom);
  const line = (key: "trainLoss" | "validationLoss") =>
    valid
      .map(
        (p, i) =>
          `${i ? "L" : "M"}${x(p.step).toFixed(2)} ${y(p[key]).toFixed(2)}`,
      )
      .join(" ");
  const last = valid.at(-1);
  return (
    <div
      className="loss-visual"
      role="img"
      aria-label={`Training and validation loss by completed training step.${last ? ` Step ${last.step}: training loss ${formatNumber(last.trainLoss, 4)}, validation loss ${formatNumber(last.validationLoss, 4)}.` : " No training history yet."}`}
    >
      <div className="loss-y-axis" aria-hidden="true">
        {[0, 0.5, 1].map((t) => (
          <span key={t} style={{ top: `${(y(t * maxValue) / height) * 100}%` }}>
            {(t * maxValue).toFixed(2)}
          </span>
        ))}
      </div>
      <div className="loss-plot">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          preserveAspectRatio="none"
          aria-hidden="true"
          className="loss-chart"
        >
          {[0, 0.5, 1].map((t) => (
            <line
              key={t}
              x1={left}
              x2={width - right}
              y1={y(t * maxValue)}
              y2={y(t * maxValue)}
              stroke="currentColor"
              strokeOpacity=".10"
              vectorEffect="non-scaling-stroke"
            />
          ))}
          {last && (
            <>
              <path
                d={line("trainLoss")}
                fill="none"
                stroke="#e6a27a"
                strokeWidth="2"
                vectorEffect="non-scaling-stroke"
              />
              <path
                d={line("validationLoss")}
                fill="none"
                stroke="#8fb6c8"
                strokeDasharray="5 4"
                strokeWidth="1.6"
                vectorEffect="non-scaling-stroke"
              />
              <circle
                cx={x(last.step)}
                cy={y(last.trainLoss)}
                r="3"
                fill="#e6a27a"
              />
              <circle
                cx={x(last.step)}
                cy={y(last.validationLoss)}
                r="2.5"
                fill="#8fb6c8"
              />
            </>
          )}
        </svg>
        {valid.length < 2 && (
          <p className="loss-empty">
            {last
              ? last.step > 0
                ? "Continue training to trace new steps."
                : "Take your first step to trace the learning curve."
              : "Your learning curve starts with the first step."}
          </p>
        )}
        <div className="loss-x-axis" aria-hidden="true">
          <span>0</span>
          <span>
            {maxStep.toLocaleString()} {maxStep === 1 ? "step" : "steps"}
          </span>
        </div>
      </div>
    </div>
  );
}

export function ModelTrace({
  snapshot,
  onSelect,
}: {
  snapshot: Snapshot | null;
  onSelect: (id: number) => void;
}) {
  if (!snapshot)
    return (
      <div className="model-trace-pending">
        <Icon name="graph" />
        <span>The model architecture appears when the runtime is ready.</span>
      </div>
    );
  const { config, parameters, graph } = snapshot;
  const inputSize = config.dataset === "digits" ? 64 : 2;
  const outputSize = config.dataset === "digits" ? 10 : 2;
  const widths = [inputSize, ...config.hidden, outputSize];
  return (
    <div
      className="model-trace"
      tabIndex={0}
      aria-label="Model architecture. Select a layer to inspect its weight tensor. Scroll horizontally to see all layers on a small screen."
    >
      <div className="model-trace-input">
        <span className="trace-index">INPUT</span>
        <strong>
          {inputSize}
          <small>features</small>
        </strong>
        <span className="trace-detail">
          {config.dataset === "digits" ? "8 × 8 pixels" : "x₁ · x₂"}
        </span>
      </div>
      {widths.slice(1).map((width, i) => {
        const name = `layer${i}.weight`;
        const node = graph.nodes.find((n) => n.name === name);
        const count = parameters
          .filter((p) => p.name === name || p.name === `layer${i}.bias`)
          .reduce((sum, p) => sum + p.shape.reduce((a, b) => a * b, 1), 0);
        const isOutput = i === widths.length - 2;
        return (
          <div
            className={`trace-step ${isOutput ? "trace-output" : ""}`}
            key={name}
          >
            <span className="trace-connector" aria-hidden="true">
              <Icon name="arrow" size={17} />
            </span>
            <button
              className="model-trace-layer"
              disabled={!node}
              onClick={() => {
                if (node) onSelect(node.id);
              }}
              aria-label={`Inspect ${name}, shape ${widths[i]} by ${width}. ${count.toLocaleString()} layer parameters.`}
            >
              <span className="trace-index">
                {isOutput
                  ? "OUTPUT"
                  : `DENSE ${String(i + 1).padStart(2, "0")}`}
                <Icon name="arrow" size={14} />
              </span>
              <strong>
                {width}
                <small>{isOutput ? "classes" : "neurons"}</small>
              </strong>
              <span className="trace-detail">
                {isOutput ? "linear" : config.activation}
                <span>{count.toLocaleString()} params</span>
              </span>
            </button>
          </div>
        );
      })}
    </div>
  );
}

export function DigitImage({
  pixels,
  label,
  size = 58,
}: {
  pixels: number[];
  label?: string;
  size?: number;
}) {
  const max = Math.max(1, ...pixels);
  return (
    <svg
      viewBox="0 0 8 8"
      width={size}
      height={size}
      role="img"
      aria-label={label || "Handwritten digit"}
      className="digit-image"
      shapeRendering="crispEdges"
    >
      <rect width="8" height="8" fill="#111615" />
      {pixels.map((p, i) => (
        <rect
          key={i}
          x={i % 8}
          y={Math.floor(i / 8)}
          width="1"
          height="1"
          fill={`rgb(${Math.round((234 * p) / max)},${Math.round((207 * p) / max)},${Math.round((177 * p) / max)})`}
        />
      ))}
    </svg>
  );
}

export function DigitExamples({
  examples,
  onSelect,
}: {
  examples: DigitExample[];
  onSelect: (example: DigitExample) => void;
}) {
  const [filter, setFilter] = useState<"all" | "correct" | "incorrect">("all");
  const shown = examples.filter(
    (e) =>
      filter === "all" ||
      (filter === "correct"
        ? e.label === e.prediction
        : e.label !== e.prediction),
  );
  return (
    <div className="digit-gallery">
      <div className="section-heading">
        <h3>Validation examples</h3>
        <div
          className="segmented small"
          aria-label="Filter validation examples"
        >
          {(["all", "correct", "incorrect"] as const).map((f) => (
            <button
              key={f}
              className={filter === f ? "active" : ""}
              aria-pressed={filter === f}
              onClick={() => setFilter(f)}
            >
              {f}
            </button>
          ))}
        </div>
      </div>
      <p className="digit-gallery-caption">
        Read the prediction. Select a digit to see its class probabilities.
      </p>
      <div className="digit-examples">
        {shown.slice(0, 24).map((e, i) => (
          <button
            key={i}
            className={`digit-example ${e.label === e.prediction ? "correct" : "incorrect"}`}
            onClick={() => onSelect(e)}
            aria-label={`Actual digit ${e.label}, predicted ${e.prediction}. Show prediction.`}
          >
            <DigitImage pixels={e.pixels} />
            <span>
              <span>{e.label}</span>
              <Icon name="arrow" size={12} />
              <span>{e.prediction}</span>
            </span>
          </button>
        ))}
      </div>
      <p className="digit-filter-status" role="status">
        {!shown.length
          ? `No ${filter} examples in this bounded snapshot.`
          : `${Math.min(24, shown.length)} ${filter === "all" ? "" : `${filter} `}examples shown${shown.length > 24 ? ` of ${shown.length}` : ""}. Actual → predicted.`}
      </p>
    </div>
  );
}

export function DrawingPad({
  onPredict,
  disabled,
}: {
  onPredict: (pixels: number[]) => void;
  disabled: boolean;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const down = useRef(false);
  const [hasInk, setHasInk] = useState(false);
  const draw = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const canvas = ref.current!,
      ctx = canvas.getContext("2d")!,
      rect = canvas.getBoundingClientRect();
    const x = ((event.clientX - rect.left) * canvas.width) / rect.width,
      y = ((event.clientY - rect.top) * canvas.height) / rect.height;
    if (event.type === "pointerdown") {
      down.current = true;
      ctx.beginPath();
      ctx.moveTo(x, y);
      canvas.setPointerCapture(event.pointerId);
    }
    if (!down.current) return;
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 21;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineTo(x + 0.01, y + 0.01);
    ctx.stroke();
    setHasInk(true);
  };
  const predict = () => {
    const canvas = ref.current!,
      ctx = canvas.getContext("2d")!;
    const data = ctx.getImageData(0, 0, 192, 192).data;
    const pixels = Array.from({ length: 64 }, (_, i) => {
      let sum = 0;
      const bx = (i % 8) * 24,
        by = Math.floor(i / 8) * 24;
      for (let y = by; y < by + 24; y++)
        for (let x = bx; x < bx + 24; x++)
          sum += data[(y * 192 + x) * 4 + 3] / 255;
      return (16 * sum) / (24 * 24);
    });
    onPredict(pixels);
  };
  return (
    <div className="drawing">
      <div className="section-heading">
        <h3>Try your handwriting</h3>
        <span className="tiny-code">8 × 8 INPUT</span>
      </div>
      <div className="drawing-row">
        <div className="drawing-surface">
          <canvas
            ref={ref}
            width="192"
            height="192"
            onPointerDown={draw}
            onPointerMove={draw}
            onPointerUp={() => {
              down.current = false;
            }}
            onPointerCancel={() => {
              down.current = false;
            }}
            aria-label="Draw one digit using mouse or touch. Use validation examples as a keyboard accessible alternative."
          />
          {!hasInk && (
            <span className="drawing-placeholder" aria-hidden="true">
              <Icon name="plus" size={24} />
              <span>Draw a digit here</span>
            </span>
          )}
        </div>
        <div className="drawing-instructions">
          <p>Make it your own.</p>
          <p className="muted">
            Draw a large, centered digit. Strokes are averaged into 64 pixels in
            [0, 16]. Your handwriting may differ from the training data.
          </p>
          <div className="drawing-actions">
            <button
              className="button secondary"
              onClick={predict}
              disabled={disabled || !hasInk}
            >
              Predict <Icon name="arrow" size={14} />
            </button>
            <button
              className="text-button"
              disabled={!hasInk}
              onClick={() => {
                ref.current!.getContext("2d")!.clearRect(0, 0, 192, 192);
                setHasInk(false);
              }}
            >
              Clear canvas
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export function ProbabilityBars({ values }: { values?: number[] }) {
  const winner = values?.length ? values.indexOf(Math.max(...values)) : -1;
  return (
    <div
      className="probability-bars"
      aria-label="Prediction probabilities by digit"
    >
      {Array.from({ length: 10 }, (_, i) => (
        <div
          key={i}
          className={`probability-row ${winner === i ? "probability-winner" : ""}`}
        >
          <span className="tiny-code">{i}</span>
          <div className="probability-track">
            <i
              style={
                { "--amount": `${(values?.[i] || 0) * 100}%` } as CSSProperties
              }
            />
          </div>
          <span className="tiny-code">
            {values ? `${(values[i] * 100).toFixed(1)}%` : "—"}
          </span>
        </div>
      ))}
    </div>
  );
}

export function ConfusionMatrix({ matrix }: { matrix: number[][] }) {
  const max = Math.max(1, ...matrix.flat());
  return (
    <div className="confusion">
      <div className="matrix-caption">Predicted class →</div>
      <table
        className="confusion-table"
        aria-label="Validation confusion matrix. Rows are actual classes; columns are predicted classes."
      >
        <thead>
          <tr>
            <th scope="col">
              <span className="matrix-corner" aria-label="Actual class">
                ↓
              </span>
            </th>
            {Array.from({ length: 10 }, (_, i) => (
              <th scope="col" className="matrix-label" key={i}>
                {i}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <th scope="row" className="matrix-label">
                {i}
              </th>
              {row.map((n, j) => (
                <td
                  title={`Actual ${i}, predicted ${j}: ${n}`}
                  key={j}
                  className="matrix-cell"
                  style={{
                    backgroundColor: `rgba(${i === j ? "222, 153, 109" : "128, 172, 195"}, ${0.03 + (n / max) * 0.66})`,
                  }}
                >
                  {n || <span aria-label="0">·</span>}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted compact">
        Rows: actual class. Counts from the validation split.
      </p>
    </div>
  );
}
