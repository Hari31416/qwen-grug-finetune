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
      console.warn("Invalid results file structure at:", path)
      return
    }

    // Extract run metadata from path
    let model = "deepseek-r1-1.5b"
    let runType = "baseline"
    let benchmark = "gsm8k"
    let promptStyle = "normal"

    const normalizedPath = path.replace(/\\/g, "/")
    const parts = normalizedPath.split("/")

    if (parts.length >= 4) {
      model = parts[parts.length - 3]
      runType = parts[parts.length - 2]
      const fileName = parts[parts.length - 1]
      const nameParts = fileName.replace(".json", "").split("_")
      benchmark = nameParts[0]
      if (nameParts.length > 1) {
        promptStyle = nameParts.slice(1).join("_")
      }
    } else {
      const fileName = parts[parts.length - 1]
      if (fileName.includes("grug_prompt")) {
        promptStyle = "grug_prompt"
      } else if (fileName.includes("normal")) {
        promptStyle = "normal"
      }
      if (normalizedPath.includes("finetuned")) {
        runType = "finetuned"
      }
      if (fileName.includes("arc")) {
        benchmark = "arc"
      }
    }

    const runId = `${model}-${runType}-${benchmark}_${promptStyle}`
    data.results[runId] = {
      summary: parsed.summary,
      results: parsed.results,
      metadata: { model, runType, benchmark, promptStyle },
    }
    console.log(`Successfully parsed run ID: ${runId}`)
  } catch (err) {
    console.error("Error parsing results JSON:", err)
  }
}
