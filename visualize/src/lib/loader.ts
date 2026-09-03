import type {
  SftPrompt,
  RawTrace,
  CompressedTrace,
  ValidatedTrace,
  WorkspaceData,
} from "@/types"

export function createEmptyWorkspace(): WorkspaceData {
  return {
    prompts: {},
    rawTraces: {},
    compressedTraces: {},
    validatedTraces: {},
    sftFormatted: [],
    validationReport: null,
    results: {},
    sources: [],
    isDemo: false,
  }
}

export function parsePromptsJsonl(
  content: string,
  data: WorkspaceData
): void {
  const lines = content.split("\n")
  const sourcesSet = new Set<string>(data.sources)
  lines.forEach((line) => {
    if (!line.trim()) return
    try {
      const obj = JSON.parse(line) as SftPrompt
      if (obj.id) {
        data.prompts[obj.id] = obj
        if (obj.source) {
          sourcesSet.add(obj.source)
        }
      }
    } catch (e) {
      // Ignore parse errors for individual lines
    }
  })
  data.sources = Array.from(sourcesSet)
}

export function parseRawTracesJsonl(
  content: string,
  data: WorkspaceData
): void {
  const lines = content.split("\n")
  lines.forEach((line) => {
    if (!line.trim()) return
    try {
      const obj = JSON.parse(line) as RawTrace
      if (obj.id) {
        data.rawTraces[obj.id] = obj
      }
    } catch (e) {
      // Ignore
    }
  })
}

export function parseCompressedTracesJsonl(
  content: string,
  data: WorkspaceData
): void {
  const lines = content.split("\n")
  lines.forEach((line) => {
    if (!line.trim()) return
    try {
      const obj = JSON.parse(line) as CompressedTrace
      if (obj.id) {
        data.compressedTraces[obj.id] = obj
      }
    } catch (e) {
      // Ignore
    }
  })
}

export function parseValidatedTracesJsonl(
  content: string,
  data: WorkspaceData
): void {
  const lines = content.split("\n")
  lines.forEach((line) => {
    if (!line.trim()) return
    try {
      const obj = JSON.parse(line) as ValidatedTrace
      if (obj.id) {
        data.validatedTraces[obj.id] = obj
      }
    } catch (e) {
      // Ignore
    }
  })
}

export function parseSftFormattedJsonl(
  content: string,
  type: "train" | "valid",
  data: WorkspaceData
): void {
  const lines = content.split("\n")
  lines.forEach((line) => {
    if (!line.trim()) return
    try {
      const obj = JSON.parse(line) as { text: string }
      if (obj.text) {
        data.sftFormatted.push({
          text: obj.text,
          type,
        })
      }
    } catch (e) {
      // Ignore
    }
  })
}

export function parseResultsJson(
  content: string,
  path: string,
  data: WorkspaceData
): void {
  try {
    const parsed = JSON.parse(content)
    if (!parsed.summary || !parsed.results) {
      console.warn('Invalid results file structure at:', path)
      return
    }

    // Extract run metadata from path
    let model = 'deepseek-r1-1.5b'
    let runType = 'baseline'
    let benchmark = 'gsm8k'
    let promptStyle = 'normal'

    const normalizedPath = path.replace(/\\/g, '/')
    if (normalizedPath.includes('7b')) {
      model = 'deepseek-r1-7b'
    } else if (normalizedPath.includes('1.5b')) {
      model = 'deepseek-r1-1.5b'
    }

    if (normalizedPath.includes('dpo')) {
      runType = 'dpo'
    } else if (normalizedPath.includes('finetuned') || normalizedPath.includes('sft')) {
      runType = 'finetuned'
    } else if (normalizedPath.includes('baseline')) {
      runType = 'baseline'
    }

    if (normalizedPath.includes('grug_prompt')) {
      promptStyle = 'grug_prompt'
    } else if (normalizedPath.includes('normal')) {
      promptStyle = 'normal'
    }

    const normalizedResults = parsed.results.map((item: Record<string, unknown>, idx: number) => {
      const id = typeof item.id === 'number' ? item.id : (typeof item.index === 'number' ? item.index : idx + 1)
      const question = typeof item.question === 'string' ? item.question : ''
      const groundTruth = String(item.ground_truth ?? item.ground_truth_raw ?? item.ground_truth_numeric ?? '')
      const thinking = String(item.thinking_content ?? item.thinking ?? '')
      const answer = String(item.answer_content ?? item.answer ?? '')
      const output = String(item.output ?? item.raw_response ?? (thinking ? `${thinking}\n</think>\n${answer}` : answer))
      const predictedAnswer = item.predicted_answer !== undefined ? String(item.predicted_answer) : (item.prediction_numeric !== undefined ? String(item.prediction_numeric) : '')
      const correct = Boolean(item.correct ?? item.is_correct ?? false)
      const formatCompliance = item.format_compliance !== undefined ? Boolean(item.format_compliance) : (item.is_format_compliant !== undefined ? Boolean(item.is_format_compliant) : true)
      const thinkingTokens = typeof item.thinking_tokens === 'number' ? item.thinking_tokens : 0
      const answerTokens = typeof item.answer_tokens === 'number' ? item.answer_tokens : 0
      const totalTokens = typeof item.total_tokens === 'number' ? item.total_tokens : (thinkingTokens + answerTokens)
      const latencySeconds = typeof item.latency_seconds === 'number' ? item.latency_seconds : (typeof item.latency_sec === 'number' ? item.latency_sec : 0)
      const tokensPerSecond = typeof item.tokens_per_second === 'number' ? item.tokens_per_second : (latencySeconds > 0 ? totalTokens / latencySeconds : 0)

      return {
        id,
        question,
        ground_truth: groundTruth,
        thinking_content: thinking,
        answer_content: answer,
        output,
        predicted_answer: predictedAnswer,
        correct,
        thinking_tokens: thinkingTokens,
        answer_tokens: answerTokens,
        total_tokens: totalTokens,
        latency_seconds: latencySeconds,
        tokens_per_second: tokensPerSecond,
        format_compliance: formatCompliance,
      }
    })

    const runId = `${model}-${runType}-${benchmark}_${promptStyle}`
    data.results[runId] = {
      summary: parsed.summary,
      results: normalizedResults,
      metadata: { model, runType, benchmark, promptStyle },
    }
    console.log(`Successfully parsed run ID: ${runId}`)
  } catch (err) {
    console.error('Error parsing results JSON:', err)
  }
}

export function enrichWorkspaceFromResults(data: WorkspaceData): void {
  const prompts = data.prompts
  const rawTraces = data.rawTraces
  const compressedTraces = data.compressedTraces
  const validatedTraces = data.validatedTraces

  Object.values(data.results).forEach((run) => {
    const isFinetuned = run.metadata.runType === 'finetuned' || run.metadata.runType === 'dpo'
    const benchmark = run.metadata.benchmark

    run.results.forEach((item) => {
      const id = `${benchmark}-${item.id}`

      // Create prompts from results if missing
      if (!prompts[id]) {
        prompts[id] = {
          id,
          source: benchmark,
          prompt: item.question,
          ground_truth: item.ground_truth,
        }
      }

      if (!isFinetuned) {
        // Populate baseline raw trace
        if (!rawTraces[id]) {
          rawTraces[id] = {
            id,
            source: benchmark,
            prompt: item.question,
            raw_thinking: item.thinking_content || item.output || '',
            raw_answer: item.answer_content || '',
            raw_answer_correct: item.correct,
          }
        }
      } else {
        // Populate finetuned/dpo compressed trace
        if (!compressedTraces[id]) {
          compressedTraces[id] = {
            id,
            compressed_thinking: item.thinking_content || item.output || '',
          }
        }
        // Also populate validated trace as placeholder
        if (item.correct && !validatedTraces[id]) {
          validatedTraces[id] = { id }
        }
      }
    })
  })
}

