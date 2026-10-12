/** The worker boundary is coarse grained: graphs, training batches and bounded snapshots. */
export type Backend = 'cpu' | 'webgpu';
export type Dataset = 'xor' | 'moons' | 'spiral' | 'custom' | 'digits';
export type Command = 'boot' | 'init' | 'step' | 'snapshot' | 'inspect' | 'export' | 'import' | 'predict' | 'benchmark' | 'loss-test' | 'validate' | 'resources';
export type CompilerOptions = { dce: boolean; fold: boolean; cse: boolean; fusion: boolean; reuse: boolean; tiled: boolean };
export interface Sample { x: number; y: number; label: number }
export interface ExperimentConfig {
  dataset: Dataset;
  seed: number;
  samples: number;
  noise: number;
  hidden: number[];
  activation: 'relu' | 'tanh' | 'sigmoid';
  optimizer: 'sgd' | 'momentum' | 'adam';
  learningRate: number;
  batchSize: number;
  backend: Backend;
  compiler: CompilerOptions;
  customPoints?: Sample[];
}
export interface Capabilities { webgpu: boolean; adapter?: string; reason?: string; python?: string; numpy?: string; pyodide?: string }
export interface Metrics {
  step: number;
  trainLoss: number;
  validationLoss: number;
  trainAccuracy: number;
  validationAccuracy: number;
  gradientNorm: number | null;
  elapsedMs: number;
  stepMs?: number;
}
export interface HistoryPoint { step: number; trainLoss: number; validationLoss: number }
export interface GraphNode {
  id: number; op: string; inputs: number[]; shape: number[]; dtype: string;
  attrs?: Record<string, unknown>; name?: string | null; kind: string; provenance?: number[];
}
export interface GraphDescription {
  nodes: GraphNode[];
  forwardIds?: number[];
  backwardIds?: number[];
  parameterIds?: number[];
  gradientIds?: Record<string, number>;
}
export interface Kernel {
  id: number | string; name?: string; op?: string; source?: string; wgsl?: string;
  nodes?: number[]; source_nodes?: number[]; inputs?: number[]; outputs?: number[];
  dispatch?: number[]; [key: string]: unknown;
}
export interface Lifetime {
  node?: number; tensor?: number; tensor_id?: number; name?: string; start?: number; end?: number;
  first?: number; last?: number; buffer?: number | string; buffer_id?: number | string;
  bytes?: number; capacity?: number; kind?: string; [key: string]: unknown;
}
export interface ExecutionPlan {
  nodes: GraphNode[]; kernels: Kernel[];
  buffers: Array<Record<string, unknown>>;
  lifetimes: Lifetime[] | Record<string, Lifetime>;
  passes: Array<Record<string, unknown>>;
  stats: Record<string, number | string | boolean>;
  tensor_buffers?: Record<string, number | string>;
  outputs?: Record<string, number>;
  source_mapping?: Record<string, unknown>;
  cache_key?: string;
  [key: string]: unknown;
}
export interface ParameterSummary { name: string; shape: number[]; gradientNorm: number | null; valueNorm?: number }
export interface DecisionBoundary { width: number; height: number; xmin: number; xmax: number; ymin: number; ymax: number; probabilities: number[] }
export interface DigitExample { pixels: number[]; label: number; prediction: number; probabilities?: number[] }
export interface DigitsResult {
  examples: DigitExample[]; confusion: number[][]; accuracy: number; split: string;
  probabilities?: number[]; testAccuracy?: number; checkpointStep?: number;
}
export interface Snapshot {
  config: ExperimentConfig;
  backend: Backend;
  metrics: Metrics;
  history: HistoryPoint[];
  samples?: Sample[];
  validationSamples?: Sample[];
  boundary?: DecisionBoundary;
  graph: GraphDescription;
  plan: ExecutionPlan;
  parameters: ParameterSummary[];
  parameterCount: number;
  digits?: DigitsResult;
  notes?: string[];
}
export interface TensorInspection {
  nodeId: number; shape: number[]; dtype: string; values?: number[]; gradient?: number[];
  gradientShape?: number[]; unavailable?: string; totalElements?: number;
  instrumentation?: string;
}
export interface BenchmarkRow {
  name: string; backend: string; configuration: string;
  medianMs: number; minMs?: number; maxMs?: number; stdMs?: number;
  compilationMs?: number; uploadMs?: number; executionMs?: number; readbackMs?: number;
  kernels?: number; scratchBytes?: number; repetitions: number;
}
export interface BenchmarkReport {
  rows: BenchmarkRow[];
  timingMethod: string;
  environment: Record<string, unknown>;
  timestamp?: string;
  notes?: string[];
}
export interface WorkerRequest { id: number; command: Command; payload?: unknown; runId: number; generation: number }
export interface WorkerResponse { id: number; runId: number; generation: number; result?: unknown; error?: string }
export const DEFAULT_CONFIG: ExperimentConfig = {
  dataset: 'moons', seed: 42, samples: 240, noise: 0.12, hidden: [16, 16],
  activation: 'tanh', optimizer: 'adam', learningRate: 0.01, batchSize: 32, backend: 'cpu',
  compiler: { dce: true, fold: true, cse: true, fusion: true, reuse: true, tiled: true },
};
