import { useRef, useState } from 'react';
import type { CSSProperties, PointerEvent as ReactPointerEvent } from 'react';
import type { DecisionBoundary, DigitExample, HistoryPoint, Sample } from './types';

export function Icon({ name, size = 18 }: { name: string; size?: number }) {
  const paths: Record<string, string> = {
    play: 'M8 5l11 7-11 7V5Z', pause: 'M8 5v14M16 5v14', step: 'M6 5l10 7-10 7V5ZM19 5v14',
    reset: 'M3 11a9 9 0 1 1 2 7M3 4v7h7', download: 'M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5',
    upload: 'M12 16V4m-5 5 5-5 5 5M4 16v5h16v-5', check: 'm5 12 4 4L19 6',
    close: 'm6 6 12 12M6 18 18 6', chevron: 'm9 5 7 7-7 7', plus: 'M12 5v14M5 12h14',
    graph: 'M5 5h5v5H5zM14 14h5v5h-5zM14 3h5v5h-5zM8 10v7h6M10 7h4',
    plot: 'M4 3v17h17M7 14l4-6 4 3 5-6', chip: 'M7 7h10v10H7zM9 2v5m6-5v5M9 17v5m6-5v5M2 9h5m-5 6h5m10-6h5m-5 6h5',
    memory: 'M3 6h18v12H3zM7 10v4m5-4v4m5-4v4M7 18v3m5-3v3m5-3v3',
    info: 'M12 11v6m0-10v1M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z',
    bolt: 'm13 2-9 12h7l-1 8 10-13h-7l0-7Z', arrow: 'M4 12h16m-5-5 5 5-5 5',
    folder: 'M3 6h7l2 3h9v11H3V6Z', stop: 'M6 6h12v12H6z',
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name] || paths.info} /></svg>;
}

export function formatNumber(value?: number | null, digits = 3) {
  return value == null || !Number.isFinite(value) ? '—' : value.toFixed(digits);
}
export function bytes(value: number) { return value >= 1048576 ? `${(value / 1048576).toFixed(2)} MiB` : value >= 1024 ? `${(value / 1024).toFixed(1)} KiB` : `${value} B`; }
export function Stat({ label, value, detail, accent }: { label: string; value: string; detail?: string; accent?: boolean }) {
  return <div className={`stat ${accent ? 'accent-stat' : ''}`}><span className="eyebrow">{label}</span><strong>{value}</strong>{detail && <span className="stat-detail">{detail}</span>}</div>;
}
export function EmptyState({ title, children, icon = 'graph' }: { title: string; children: React.ReactNode; icon?: string }) {
  return <div className="empty-state"><span className="empty-icon"><Icon name={icon} size={27} /></span><h3>{title}</h3><p>{children}</p></div>;
}

export function DecisionPlot({ samples, validation = [], boundary, editable, pointClass, onChange, preview }: {
  samples: Sample[]; validation?: Sample[]; boundary?: DecisionBoundary; editable?: boolean;
  pointClass: number; onChange?: (samples: Sample[]) => void; preview?: boolean;
}) {
  const ref = useRef<SVGSVGElement>(null);
  const drag = useRef<number | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const bounds = boundary || { xmin: -2.4, xmax: 2.4, ymin: -2.4, ymax: 2.4 };
  const inset = 36, size = 520, plotSize = size - 2 * inset;
  const tx = (x: number) => inset + (x - bounds.xmin) / (bounds.xmax - bounds.xmin) * plotSize;
  const ty = (y: number) => size - inset - (y - bounds.ymin) / (bounds.ymax - bounds.ymin) * plotSize;
  const position = (event: ReactPointerEvent<SVGSVGElement>) => {
    const rect = ref.current!.getBoundingClientRect();
    const px = (event.clientX - rect.left) / rect.width * size;
    const py = (event.clientY - rect.top) / rect.height * size;
    return {
      x: Math.max(bounds.xmin, Math.min(bounds.xmax, bounds.xmin + (px - inset) / plotSize * (bounds.xmax - bounds.xmin))),
      y: Math.max(bounds.ymin, Math.min(bounds.ymax, bounds.ymax - (py - inset) / plotSize * (bounds.ymax - bounds.ymin))),
    };
  };
  const pointerDown = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (!editable || !onChange) return;
    const pos = position(event);
    const nearest = samples.findIndex(p => Math.hypot(tx(p.x) - tx(pos.x), ty(p.y) - ty(pos.y)) < 11);
    if (nearest >= 0) { drag.current = nearest; setSelected(nearest); }
    else if (samples.length < 1000) { drag.current = samples.length; setSelected(samples.length); onChange([...samples, { ...pos, label: pointClass }]); }
    event.currentTarget.setPointerCapture(event.pointerId);
  };
  const ticks = Array.from({ length: 5 }, (_, i) => i / 4);
  return <div className={`decision-plot ${editable ? 'editable' : ''}`}>
    <svg ref={ref} viewBox={`0 0 ${size} ${size}`} role="img" aria-label={`Decision boundary with ${samples.length} training points. Copper is class zero and blue is class one.`}
      onPointerDown={pointerDown} onPointerMove={event => {
        if (drag.current === null || !onChange) return;
        const index = drag.current, pos = position(event);
        onChange(samples.map((p, i) => i === index ? { ...p, ...pos } : p));
      }} onPointerUp={() => { drag.current = null; }} onPointerCancel={() => { drag.current = null; }}>
      <defs><clipPath id="plot-clip"><rect x={inset} y={inset} width={plotSize} height={plotSize} rx="2" /></clipPath>
      <pattern id="fine-grid" width={plotSize / 16} height={plotSize / 16} patternUnits="userSpaceOnUse" x={inset} y={inset}><path d={`M ${plotSize / 16} 0 H 0 V ${plotSize / 16}`} fill="none" stroke="#d6d8d0" strokeOpacity="0.055" strokeWidth="1" /></pattern></defs>
      <rect x={inset} y={inset} width={plotSize} height={plotSize} fill="#202524" />
      {boundary && !preview && <g clipPath="url(#plot-clip)">{boundary.probabilities.map((p, i) => {
        const mix = Math.max(0, Math.min(1, p));
        const rgb = [Math.round(177 * (1 - mix) + 83 * mix), Math.round(103 * (1 - mix) + 142 * mix), Math.round(71 * (1 - mix) + 166 * mix)];
        return <rect key={i} x={inset + (i % boundary.width) * plotSize / boundary.width} y={size - inset - (Math.floor(i / boundary.width) + 1) * plotSize / boundary.height}
          width={plotSize / boundary.width + .3} height={plotSize / boundary.height + .3} fill={`rgb(${rgb.join(',')})`} opacity={0.26 + Math.abs(mix - .5) * .36} />;
      })}</g>}
      <rect x={inset} y={inset} width={plotSize} height={plotSize} fill="url(#fine-grid)" />
      {ticks.map((t, i) => <g key={i}>
        <line x1={inset + t * plotSize} x2={inset + t * plotSize} y1={inset} y2={size - inset} stroke="#b5c5b6" strokeOpacity=".10" />
        <line x1={inset} x2={size - inset} y1={inset + t * plotSize} y2={inset + t * plotSize} stroke="#b5c5b6" strokeOpacity=".10" />
        <text x={inset + t * plotSize} y={size - inset + 19} className="axis-label" textAnchor="middle">{(bounds.xmin + t * (bounds.xmax - bounds.xmin)).toFixed(1)}</text>
        <text x={inset - 10} y={size - inset - t * plotSize + 4} className="axis-label" textAnchor="end">{(bounds.ymin + t * (bounds.ymax - bounds.ymin)).toFixed(1)}</text>
      </g>)}
      <g clipPath="url(#plot-clip)">{validation.map((p, i) => <circle key={`v${i}`} cx={tx(p.x)} cy={ty(p.y)} r="3.4" fill="none" stroke={p.label === 0 ? '#eaaa83' : '#95bac9'} strokeWidth="1.3" opacity=".6" />)}
        {samples.map((p, i) => <circle key={i} cx={tx(p.x)} cy={ty(p.y)} r={editable && selected === i ? 5.6 : 3.8} fill={p.label === 0 ? '#e9ac84' : '#9dbfcd'} stroke={editable && selected === i ? '#fff' : '#242828'} strokeWidth={editable && selected === i ? 2 : 1.4} />)}
      </g>
      <text x={size - inset - 5} y={size - 3} className="axis-label">x₁</text><text x="5" y={inset - 9} className="axis-label">x₂</text>
    </svg>
    {editable && <div className="plot-edit-note"><Icon name="plus" size={13} /> Click to add · drag to move{selected !== null && <button onClick={() => { onChange?.(samples.filter((_, i) => i !== selected)); setSelected(null); }}>Delete selected</button>}</div>}
  </div>;
}

export function LossChart({ history }: { history: HistoryPoint[] }) {
  const valid = history.filter(p => Number.isFinite(p.trainLoss) && Number.isFinite(p.validationLoss));
  const width = 640, height = 138, left = 39, right = 13, top = 13, bottom = 26;
  const maxStep = Math.max(1, ...valid.map(p => p.step));
  const maxValue = Math.max(.01, ...valid.flatMap(p => [p.trainLoss, p.validationLoss])) * 1.1;
  const x = (step: number) => left + step / maxStep * (width - left - right);
  const y = (value: number) => top + (1 - value / maxValue) * (height - top - bottom);
  const line = (key: 'trainLoss' | 'validationLoss') => valid.map((p, i) => `${i ? 'L' : 'M'}${x(p.step).toFixed(2)} ${y(p[key]).toFixed(2)}`).join(' ');
  return <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Training and validation loss by completed training step" className="loss-chart">
    {[0, .5, 1].map(t => <g key={t}><line x1={left} x2={width - right} y1={y(t * maxValue)} y2={y(t * maxValue)} stroke="#ffffff" strokeOpacity=".055" /><text className="axis-label" x={left - 8} y={y(t * maxValue) + 3} textAnchor="end">{(t * maxValue).toFixed(2)}</text></g>)}
    {valid.length > 0 && <><path d={line('trainLoss')} fill="none" stroke="#e9a477" strokeWidth="2" /><path d={line('validationLoss')} fill="none" stroke="#8cb4c7" strokeDasharray="4 3" strokeWidth="1.8" />
      <circle cx={x(valid.at(-1)!.step)} cy={y(valid.at(-1)!.trainLoss)} r="3" fill="#e9a477" /></>}
    <text x={left} y={height - 5} className="axis-label">0</text><text x={width - right} y={height - 5} className="axis-label" textAnchor="end">{maxStep} steps</text>
    {valid.length < 2 && <text x={width / 2} y={height / 2 - 4} textAnchor="middle" fill="#989d96" fontSize="11">Training history appears after your first step</text>}
  </svg>;
}

export function NetworkDiagram({ layers }: { layers: number[] }) {
  const width = 236, height = 134;
  const xs = layers.map((_, i) => 22 + i * (width - 44) / (layers.length - 1));
  const ys = layers.map(n => Array.from({ length: Math.min(n, 5) }, (_, i) => height / 2 + (i - (Math.min(n, 5) - 1) / 2) * 18));
  return <svg className="network-diagram" viewBox={`0 0 ${width} ${height + 28}`} aria-label={`Network layers: ${layers.join(' to ')}`} role="img">
    {layers.slice(1).flatMap((_, j) => ys[j].flatMap((a, i) => ys[j + 1].map((b, k) => <line key={`${j}-${i}-${k}`} x1={xs[j]} x2={xs[j + 1]} y1={a} y2={b} stroke="#b2bcb2" strokeOpacity=".13" />)))}
    {layers.flatMap((_, j) => ys[j].map((y, i) => <circle key={`${j}-${i}`} cx={xs[j]} cy={y} r="4" fill={j === layers.length - 1 ? '#94b8c9' : j === 0 ? '#e3a077' : '#343b37'} stroke={j === layers.length - 1 ? '#94b8c9' : '#d2926c'} strokeWidth="1.3" />))}
    {layers.map((n, i) => <text key={i} x={xs[i]} y={height + 17} className="axis-label" textAnchor="middle">{n}</text>)}
  </svg>;
}

export function DigitImage({ pixels, label, size = 58 }: { pixels: number[]; label?: string; size?: number }) {
  const max = Math.max(1, ...pixels);
  return <svg viewBox="0 0 8 8" width={size} height={size} role="img" aria-label={label || 'Handwritten digit'} className="digit-image" shapeRendering="crispEdges"><rect width="8" height="8" fill="#111615" />{pixels.map((p, i) => <rect key={i} x={i % 8} y={Math.floor(i / 8)} width="1" height="1" fill={`rgb(${Math.round(234 * p / max)},${Math.round(207 * p / max)},${Math.round(177 * p / max)})`} />)}</svg>;
}

export function DigitExamples({ examples, onSelect }: { examples: DigitExample[]; onSelect: (example: DigitExample) => void }) {
  const [filter, setFilter] = useState<'all' | 'correct' | 'incorrect'>('all');
  const shown = examples.filter(e => filter === 'all' || (filter === 'correct' ? e.label === e.prediction : e.label !== e.prediction));
  return <div><div className="section-heading"><h3>Validation examples</h3><div className="segmented small">{(['all', 'correct', 'incorrect'] as const).map(f => <button key={f} className={filter === f ? 'active' : ''} onClick={() => setFilter(f)}>{f}</button>)}</div></div><div className="digit-examples">{shown.slice(0, 24).map((e, i) => <button key={i} className={`digit-example ${e.label === e.prediction ? 'correct' : 'incorrect'}`} onClick={() => onSelect(e)} aria-label={`Actual digit ${e.label}, predicted ${e.prediction}. Show prediction.`}><DigitImage pixels={e.pixels} /><span>{e.label} <span>→</span> {e.prediction}</span></button>)}</div>{!shown.length && <p className="muted compact">No {filter} examples in this bounded snapshot.</p>}</div>;
}

export function DrawingPad({ onPredict, disabled }: { onPredict: (pixels: number[]) => void; disabled: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const down = useRef(false);
  const [hasInk, setHasInk] = useState(false);
  const draw = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const canvas = ref.current!, ctx = canvas.getContext('2d')!, rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) * canvas.width / rect.width, y = (event.clientY - rect.top) * canvas.height / rect.height;
    if (event.type === 'pointerdown') { down.current = true; ctx.beginPath(); ctx.moveTo(x, y); canvas.setPointerCapture(event.pointerId); }
    if (!down.current) return;
    ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 21; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.lineTo(x + .01, y + .01); ctx.stroke(); setHasInk(true);
  };
  const predict = () => {
    const canvas = ref.current!, ctx = canvas.getContext('2d')!;
    const data = ctx.getImageData(0, 0, 192, 192).data;
    const pixels = Array.from({ length: 64 }, (_, i) => {
      let sum = 0;
      const bx = (i % 8) * 24, by = Math.floor(i / 8) * 24;
      for (let y = by; y < by + 24; y++) for (let x = bx; x < bx + 24; x++) sum += data[(y * 192 + x) * 4 + 3] / 255;
      return 16 * sum / (24 * 24);
    });
    onPredict(pixels);
  };
  return <div className="drawing"><div className="section-heading"><h3>Try your handwriting</h3><span className="tiny-code">8 × 8</span></div><div className="drawing-row"><canvas ref={ref} width="192" height="192" onPointerDown={draw} onPointerMove={draw} onPointerUp={() => { down.current = false; }} onPointerCancel={() => { down.current = false; }} aria-label="Draw one digit using mouse or touch. Use validation examples as a keyboard accessible alternative." /><div><p>Draw a large, centered digit.</p><p className="muted">Strokes are averaged into 64 pixels in [0, 16]. Your handwriting may differ from the training data.</p><button className="button secondary" onClick={predict} disabled={disabled || !hasInk}>Predict <Icon name="arrow" size={14} /></button><button className="text-button" onClick={() => { ref.current!.getContext('2d')!.clearRect(0, 0, 192, 192); setHasInk(false); }}>Clear canvas</button></div></div></div>;
}

export function ProbabilityBars({ values }: { values?: number[] }) {
  return <div className="probability-bars">{Array.from({ length: 10 }, (_, i) => <div key={i}><span className="tiny-code">{i}</span><div><i style={{ '--amount': `${(values?.[i] || 0) * 100}%` } as CSSProperties} /></div><span className="tiny-code">{values ? `${(values[i] * 100).toFixed(1)}%` : '—'}</span></div>)}</div>;
}

export function ConfusionMatrix({ matrix }: { matrix: number[][] }) {
  const max = Math.max(1, ...matrix.flat());
  return <div className="confusion"><div className="matrix-caption">Predicted class →</div><div className="confusion-grid"><span />{Array.from({ length: 10 }, (_, i) => <span className="matrix-label" key={`h${i}`}>{i}</span>)}{matrix.flatMap((row, i) => [<span className="matrix-label" key={`r${i}`}>{i}</span>, ...row.map((n, j) => <span title={`Actual ${i}, predicted ${j}: ${n}`} key={`${i}-${j}`} className="matrix-cell" style={{ backgroundColor: `rgba(${i === j ? '222, 153, 109' : '128, 172, 195'}, ${.03 + n / max * .66})` }}>{n || '·'}</span>)])}</div><p className="muted compact">Rows: actual class. Counts from the validation split.</p></div>;
}
