export interface SftPrompt {
  id: string
  source: string
  prompt: string
  choices?: string[]
  ground_truth?: string
}

export interface RawTrace {
  id: string
  source?: string
  prompt?: string
  raw_thinking?: string
  raw_answer?: string
  raw_answer_correct?: boolean
}

export interface CompressedTrace {
  id: string
  compressed_thinking?: string
}

export interface ValidatedTrace {
  id: string
  compressed_thinking?: string
}

export interface SftSplit {
  text: string
  type: "train" | "valid"
}

export interface ValidationReport {
  total_checked?: number
  accepted?: number
  rejected?: number
  rejection_rate?: number
  rejection_reasons?: Record<string, number>
}

export interface EvalItem {
  id: number
  question: string
  ground_truth: string
  thinking_content?: string
  answer_content?: string
  output?: string
  predicted_answer?: string
  correct: boolean
  thinking_tokens: number
  answer_tokens: number
  total_tokens: number
  latency_seconds?: number
  tokens_per_second?: number
  format_compliance?: boolean
}

export interface EvalSummary {
  accuracy: number
  format_compliance_rate?: number
  mean_thinking_tokens: number
  mean_answer_tokens: number
  mean_total_tokens: number
  mean_latency?: number
  mean_tokens_per_second?: number
  sample_count: number
  correct_count?: number
  format_compliant_count?: number
}

export interface EvalRun {
  summary: EvalSummary
  results: EvalItem[]
  metadata: {
    model: string
    runType: string
    benchmark: string
    promptStyle: string
  }
}

export interface WorkspaceData {
  prompts: Record<string, SftPrompt>
  rawTraces: Record<string, RawTrace>
  compressedTraces: Record<string, CompressedTrace>
  validatedTraces: Record<string, ValidatedTrace>
  sftFormatted: SftSplit[]
  validationReport: ValidationReport | null
  results: Record<string, EvalRun>
  sources: string[]
  isDemo: boolean
}
