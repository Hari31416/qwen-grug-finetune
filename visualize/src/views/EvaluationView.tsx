import { useState, useMemo, useEffect } from "react"
import type { WorkspaceData } from "@/types"
import { Bar } from "react-chartjs-2"
import { CodeBlock } from "@/components/ui/CodeBlock"
import { ThinkingBubble } from "@/components/ui/ThinkingBubble"
import { Search, HelpCircle, Check, X, GitCompare, LayoutGrid } from "lucide-react"

interface EvaluationViewProps {
  data: WorkspaceData
}

export function EvaluationView({ data }: EvaluationViewProps) {
  const [selectedRunId, setSelectedRunId] = useState<string>("none")
  const [explorerSearch, setExplorerSearch] = useState("")
  const [correctnessFilter, setCorrectnessFilter] = useState("all")
  const [selectedItemId, setSelectedItemId] = useState<number | null>(null)
  
  // Side by side comparator state
  const [sxsQuestionId, setSxsQuestionId] = useState<number | null>(null)
  const [sxsBenchmark, setSxsBenchmark] = useState<string>("gsm8k")
  const [sxsPromptStyle, setSxsPromptStyle] = useState<string>("normal")

  const runIds = useMemo(() => {
    return Object.keys(data.results)
  }, [data.results])

  // Select a default run when results load
  useEffect(() => {
    if (runIds.length > 0) {
      if (selectedRunId === "none" || !runIds.includes(selectedRunId)) {
        setSelectedRunId(runIds[0])
      }
    } else {
      setSelectedRunId("none")
    }
  }, [runIds, selectedRunId])



  // Get active run data
  const activeRun = useMemo(() => {
    if (selectedRunId === "none") return null
    return data.results[selectedRunId] || null
  }, [data.results, selectedRunId])

  // Render main summary comparison table row helper
  const renderTableRow = (
    model: string,
    benchmark: string,
    runType: string,
    style: string,
    name: string
  ) => {
    const runId = `${model}-${runType}-${benchmark}_${style}`
    const run = data.results[runId]

    const baselineId = `${model}-baseline-${benchmark}_normal`
    const baselineRun = data.results[baselineId]

    if (!run) {
      return (
        <tr key={runId} className="opacity-40">
          <td className="p-3 text-[13px] font-semibold text-gray-400">{name}</td>
          <td colSpan={7} className="p-3 text-center text-xs text-gray-500 italic">
            Not Loaded / Pending Pipeline Run
          </td>
        </tr>
      )
    }

    const s = run.summary
    const acc = `${(s.accuracy * 100).toFixed(1)}%`
    const compliance = s.format_compliance_rate !== undefined
      ? `${(s.format_compliance_rate * 100).toFixed(1)}%`
      : "N/A"

    let accDelta = null
    let tokDelta = null

    if (baselineRun && runId !== baselineId) {
      const accDiff = (s.accuracy - baselineRun.summary.accuracy) * 100
      const tokDiff = s.mean_thinking_tokens - baselineRun.summary.mean_thinking_tokens
      const tokPct = (tokDiff / baselineRun.summary.mean_thinking_tokens) * 100

      accDelta = (
        <span
          className={`ml-2 text-[10px] font-bold px-1 py-0.2 rounded font-mono ${
            accDiff >= 0
              ? "bg-emerald-500/10 text-emerald-400"
              : "bg-red-500/10 text-red-400"
          }`}
        >
          {accDiff >= 0 ? "+" : ""}
          {accDiff.toFixed(0)}%
        </span>
      )

      tokDelta = (
        <span
          className={`ml-2 text-[10px] font-bold px-1 py-0.2 rounded font-mono ${
            tokPct <= 0
              ? "bg-emerald-500/10 text-emerald-400"
              : "bg-red-500/10 text-red-400"
          }`}
        >
          {tokPct >= 0 ? "+" : ""}
          {tokPct.toFixed(0)}%
        </span>
      )
    }

    return (
      <tr key={runId} className="hover:bg-white/[0.01] border-b border-white/5">
        <td className="p-3 text-[13px] font-semibold text-white">{name}</td>
        <td className="p-3 text-center text-[13px] font-semibold text-white">
          {acc}
          {accDelta}
        </td>
        <td className="p-3 text-center text-[13px] text-gray-400">{compliance}</td>
        <td className="p-3 text-center text-[13px] font-mono text-amber-400">
          {Math.round(s.mean_thinking_tokens)}
          {tokDelta}
        </td>
        <td className="p-3 text-center text-[13px] font-mono text-gray-400">
          {Math.round(s.mean_answer_tokens)}
        </td>
        <td className="p-3 text-center text-[13px] text-gray-400">
          {s.mean_latency ? `${s.mean_latency.toFixed(2)}s` : "N/A"}
        </td>
        <td className="p-3 text-center text-[13px] font-mono text-gray-400">
          {s.mean_tokens_per_second ? Math.round(s.mean_tokens_per_second) : "N/A"}
        </td>
        <td className="p-3 text-center text-[13px] text-gray-500">{s.sample_count}</td>
      </tr>
    )
  }

  // Grouped charts config for react-chartjs-2
  const { accChartData, tokensChartData } = useMemo(() => {
    const runsConfig = [
      { runType: "baseline", style: "normal", name: "Base Normal" },
      { runType: "baseline", style: "grug_prompt", name: "Base Grug" },
      { runType: "finetuned", style: "normal", name: "FT Normal" },
      { runType: "finetuned", style: "grug_prompt", name: "FT Grug" },
    ]

    const labels = runsConfig.map((c) => c.name)
    const accuracyDatasets: any[] = []
    const tokensDatasets: any[] = []

    const benchmarks = ["gsm8k", "arc"]
    const colors: Record<string, { bg: string; border: string }> = {
      gsm8k: { bg: "rgba(59, 130, 246, 0.5)", border: "#3b82f6" },
      arc: { bg: "rgba(139, 92, 246, 0.5)", border: "#8b5cf6" },
    }

    benchmarks.forEach((bench) => {
      const accData: number[] = []
      const tokData: number[] = []
      let hasAny = false

      runsConfig.forEach((cfg) => {
        const runId = `deepseek-r1-1.5b-${cfg.runType}-${bench}_${cfg.style}`
        const run = data.results[runId]
        if (run) {
          accData.push(run.summary.accuracy * 100)
          tokData.push(run.summary.mean_thinking_tokens)
          hasAny = true
        } else {
          accData.push(0)
          tokData.push(0)
        }
      })

      if (hasAny) {
        accuracyDatasets.push({
          label: bench.toUpperCase(),
          data: accData,
          backgroundColor: colors[bench].bg,
          borderColor: colors[bench].border,
          borderWidth: 1.5,
          borderRadius: 4,
        })

        tokensDatasets.push({
          label: bench.toUpperCase(),
          data: tokData,
          backgroundColor: colors[bench].bg,
          borderColor: colors[bench].border,
          borderWidth: 1.5,
          borderRadius: 4,
        })
      }
    })

    return {
      accChartData: { labels, datasets: accuracyDatasets },
      tokensChartData: { labels, datasets: tokensDatasets },
    }
  }, [data.results])

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        grid: { color: "rgba(255, 255, 255, 0.05)" },
        ticks: { color: "#9ca3af" },
      },
      x: {
        grid: { display: false },
        ticks: { color: "#9ca3af" },
      },
    },
    plugins: {
      legend: { labels: { color: "#9ca3af" } },
    },
  }

  // Explorer sidebar questions filter
  const explorerItems = useMemo(() => {
    if (!activeRun) return []

    return activeRun.results
      .filter((item) => {
        if (
          explorerSearch.trim() &&
          !item.question.toLowerCase().includes(explorerSearch.toLowerCase()) &&
          !String(item.id).includes(explorerSearch)
        ) {
          return false
        }

        if (correctnessFilter === "correct" && !item.correct) return false
        if (correctnessFilter === "incorrect" && item.correct) return false

        return true
      })
      .sort((a, b) => a.id - b.id)
  }, [activeRun, explorerSearch, correctnessFilter])

  // Select default explorer question
  useEffect(() => {
    if (explorerItems.length > 0) {
      if (!selectedItemId || !explorerItems.some((item) => item.id === selectedItemId)) {
        setSelectedItemId(explorerItems[0].id)
      }
    } else {
      setSelectedItemId(null)
    }
  }, [explorerItems, selectedItemId])

  const selectedExplorerItem = useMemo(() => {
    if (!activeRun) return null
    return activeRun.results.find((r) => r.id === selectedItemId) || null
  }, [activeRun, selectedItemId])

  // SxS: Available question list (based on whatever gsm8k normal run is loaded)
  const sxsQuestions = useMemo(() => {
    const baselineKey = `deepseek-r1-1.5b-baseline-${sxsBenchmark}_${sxsPromptStyle}`
    const baseRun = data.results[baselineKey]
    if (!baseRun) return []
    return baseRun.results.map((r) => ({ id: r.id, text: r.question }))
  }, [data.results, sxsBenchmark, sxsPromptStyle])

  // Select default sxs question
  useEffect(() => {
    if (sxsQuestions.length > 0) {
      if (!sxsQuestionId || !sxsQuestions.some((q) => q.id === sxsQuestionId)) {
        setSxsQuestionId(sxsQuestions[0].id)
      }
    } else {
      setSxsQuestionId(null)
    }
  }, [sxsQuestions, sxsQuestionId])

  // Fetch the matched baseline & finetuned items for SxS
  const sxsItems = useMemo(() => {
    if (!sxsQuestionId) return { baseline: null, finetuned: null }

    const model = "deepseek-r1-1.5b"
    const baseId = `${model}-baseline-${sxsBenchmark}_${sxsPromptStyle}`
    const ftId = `${model}-finetuned-${sxsBenchmark}_${sxsPromptStyle}`

    const baseRun = data.results[baseId]
    const ftRun = data.results[ftId]

    const baselineItem = baseRun?.results.find((r) => r.id === sxsQuestionId) || null
    const finetunedItem = ftRun?.results.find((r) => r.id === sxsQuestionId) || null

    return {
      baseline: baselineItem,
      finetuned: finetunedItem,
    }
  }, [data.results, sxsQuestionId, sxsBenchmark, sxsPromptStyle])

  const noResults = Object.keys(data.results).length === 0

  return (
    <div className="flex flex-col gap-10 w-full">
      {noResults ? (
        <div className="rounded-xl border border-white/5 bg-white/[0.02] p-16 text-center flex flex-col items-center justify-center gap-3">
          <HelpCircle className="h-10 w-10 text-gray-600 animate-pulse" />
          <p className="text-gray-400 text-sm">
            Please load pipeline workspace files or demo data to display evaluation results.
          </p>
        </div>
      ) : (
        <>
          {/* Main summary benchmarks table */}
          <div className="rounded-xl border border-white/5 bg-white/[0.02] p-6 flex flex-col gap-4">
            <h3 className="font-heading font-semibold text-lg text-white flex items-center gap-2">
              <GitCompare className="h-5 w-5 text-blue-400" />
              SFT Benchmark Performance Comparison
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b-2 border-white/5 text-[11px] font-bold text-blue-400 uppercase tracking-wider">
                    <th className="p-3 text-left">Run Name / Config</th>
                    <th className="p-3 text-center">Accuracy</th>
                    <th className="p-3 text-center">Format Compliance</th>
                    <th className="p-3 text-center">Avg Thinking Tokens</th>
                    <th className="p-3 text-center">Avg Answer Tokens</th>
                    <th className="p-3 text-center">Avg Latency</th>
                    <th className="p-3 text-center">Avg Speed (tok/s)</th>
                    <th className="p-3 text-center">Samples</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="bg-white/[0.02]">
                    <td
                      colSpan={8}
                      className="p-2 text-xs font-bold text-blue-400 border-b border-white/5"
                    >
                      Benchmark: GSM8K (DEEPSEEK-R1-1.5B)
                    </td>
                  </tr>
                  {renderTableRow(
                    "deepseek-r1-1.5b",
                    "gsm8k",
                    "baseline",
                    "normal",
                    "Base / Normal Prompt"
                  )}
                  {renderTableRow(
                    "deepseek-r1-1.5b",
                    "gsm8k",
                    "baseline",
                    "grug_prompt",
                    "Base / Grug Prompt"
                  )}
                  {renderTableRow(
                    "deepseek-r1-1.5b",
                    "gsm8k",
                    "finetuned",
                    "normal",
                    "Fine-Tuned / Normal Prompt"
                  )}
                  {renderTableRow(
                    "deepseek-r1-1.5b",
                    "gsm8k",
                    "finetuned",
                    "grug_prompt",
                    "Fine-Tuned / Grug Prompt"
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* ACC and Tokens bar charts side-by-side */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="rounded-xl border border-white/5 bg-white/[0.02] p-6 flex flex-col gap-4 min-h-[350px]">
              <h3 className="font-heading font-semibold text-lg text-white">
                Accuracy Comparison (%)
              </h3>
              <div className="flex-1 relative min-h-[220px]">
                {accChartData.datasets.length > 0 && (
                  <Bar data={accChartData} options={chartOptions} />
                )}
              </div>
            </div>

            <div className="rounded-xl border border-white/5 bg-white/[0.02] p-6 flex flex-col gap-4 min-h-[350px]">
              <h3 className="font-heading font-semibold text-lg text-white">
                Avg Thinking Length (Tokens)
              </h3>
              <div className="flex-1 relative min-h-[220px]">
                {tokensChartData.datasets.length > 0 && (
                  <Bar data={tokensChartData} options={chartOptions} />
                )}
              </div>
            </div>
          </div>

          {/* USER REQUESTED: Side-by-Side Inference Comparator */}
          <div className="rounded-xl border border-white/5 bg-white/[0.02] p-6 flex flex-col gap-5">
            <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4 border-b border-white/5 pb-4">
              <div className="flex flex-col gap-1">
                <h3 className="font-heading font-semibold text-lg text-white flex items-center gap-2">
                  <LayoutGrid className="h-5 w-5 text-emerald-400" />
                  Side-by-Side Inference Comparator
                </h3>
                <p className="text-[12px] text-gray-400">
                  Directly compare how the baseline model and your fine-tuned model answer the exact same question.
                </p>
              </div>

              {/* SXS Controls */}
              <div className="flex items-center gap-2">
                <select
                  value={sxsBenchmark}
                  onChange={(e) => setSxsBenchmark(e.target.value)}
                  className="bg-white/[0.03] border border-white/5 rounded-lg p-2 text-xs text-white focus:outline-none focus:border-blue-500"
                >
                  <option value="gsm8k">GSM8K Benchmark</option>
                  <option value="arc">ARC Benchmark</option>
                </select>

                <select
                  value={sxsPromptStyle}
                  onChange={(e) => setSxsPromptStyle(e.target.value)}
                  className="bg-white/[0.03] border border-white/5 rounded-lg p-2 text-xs text-white focus:outline-none focus:border-blue-500"
                >
                  <option value="normal">Normal Prompt Style</option>
                  <option value="grug_prompt">Grug Prompt Style</option>
                </select>
              </div>
            </div>

            {sxsQuestions.length === 0 ? (
              <div className="text-center py-8 text-gray-500 text-xs italic">
                Selected baseline evaluation runs are not loaded yet to establish questions.
              </div>
            ) : (
              <div className="flex flex-col gap-4">
                {/* Question dropdown */}
                <div className="flex flex-col gap-1.5">
                  <label className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">
                    Compare Question:
                  </label>
                  <select
                    value={sxsQuestionId || ""}
                    onChange={(e) => setSxsQuestionId(Number(e.target.value))}
                    className="bg-white/[0.03] border border-white/5 rounded-lg p-2.5 text-xs text-white focus:outline-none focus:border-blue-500"
                  >
                    {sxsQuestions.map((q) => (
                      <option key={q.id} value={q.id}>
                        [ID #{q.id}] {q.text.length > 100 ? `${q.text.substring(0, 100)}...` : q.text}
                      </option>
                    ))}
                  </select>
                </div>

                {/* The comparative columns layout */}
                {sxsQuestionId && (
                  <div className="flex flex-col gap-4 mt-2">
                    {/* Display the full question */}
                    <div className="rounded-lg border border-white/5 bg-white/[0.01] p-4 flex flex-col gap-2">
                      <span className="text-[10px] font-bold text-gray-500 uppercase tracking-wider block">
                        Question Text
                      </span>
                      <p className="text-[13px] text-gray-200 leading-relaxed">
                        {sxsQuestions.find((q) => q.id === sxsQuestionId)?.text}
                      </p>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-2">
                      {/* Left Column: Baseline */}
                      <div className="rounded-lg border border-white/5 bg-white/[0.01] p-4 flex flex-col gap-4">
                        <div className="flex justify-between items-center border-b border-white/5 pb-2">
                          <span className="text-xs font-bold text-red-400 uppercase tracking-wider">
                            Baseline Output
                          </span>
                          {sxsItems.baseline && (
                            <span
                              className={`text-[10px] font-bold px-1.5 py-0.5 rounded uppercase ${
                                sxsItems.baseline.correct
                                  ? "bg-emerald-500/10 text-emerald-400"
                                  : "bg-red-500/10 text-red-400"
                              }`}
                            >
                              {sxsItems.baseline.correct ? "Correct" : "Incorrect"}
                            </span>
                          )}
                        </div>

                        {!sxsItems.baseline ? (
                          <div className="text-center py-10 text-gray-600 text-xs italic">
                            Baseline result not loaded
                          </div>
                        ) : (
                          <div className="flex flex-col gap-3">
                            {sxsItems.baseline.thinking_content && (
                              <ThinkingBubble
                                title="Baseline thinking Process"
                                content={sxsItems.baseline.thinking_content}
                              />
                            )}
                            <CodeBlock
                              content={
                                sxsItems.baseline.answer_content ||
                                sxsItems.baseline.output ||
                                ""
                              }
                            />
                            {/* Stats */}
                            <div className="grid grid-cols-3 gap-2 bg-[#070913]/30 border border-white/5 rounded-lg p-2.5 text-[11px] font-mono text-gray-400">
                              <div>
                                Latency:{" "}
                                <strong className="text-gray-300">
                                  {sxsItems.baseline.latency_seconds?.toFixed(2)}s
                                </strong>
                              </div>
                              <div>
                                thinking:{" "}
                                <strong className="text-gray-300">
                                  {sxsItems.baseline.thinking_tokens} tok
                                </strong>
                              </div>
                              <div>
                                Speed:{" "}
                                <strong className="text-gray-300">
                                  {Math.round(sxsItems.baseline.tokens_per_second || 0)} t/s
                                </strong>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>

                      {/* Right Column: Finetuned */}
                      <div className="rounded-lg border border-white/5 bg-white/[0.01] p-4 flex flex-col gap-4">
                        <div className="flex justify-between items-center border-b border-white/5 pb-2">
                          <span className="text-xs font-bold text-emerald-400 uppercase tracking-wider">
                            Fine-Tuned Output
                          </span>
                          {sxsItems.finetuned && (
                            <span
                              className={`text-[10px] font-bold px-1.5 py-0.5 rounded uppercase ${
                                sxsItems.finetuned.correct
                                  ? "bg-emerald-500/10 text-emerald-400"
                                  : "bg-red-500/10 text-red-400"
                              }`}
                            >
                              {sxsItems.finetuned.correct ? "Correct" : "Incorrect"}
                            </span>
                          )}
                        </div>

                        {!sxsItems.finetuned ? (
                          <div className="text-center py-10 text-gray-600 text-xs italic">
                            Finetuned result not loaded
                          </div>
                        ) : (
                          <div className="flex flex-col gap-3">
                            {sxsItems.finetuned.thinking_content && (
                              <ThinkingBubble
                                title="Finetuned thinking Process"
                                content={sxsItems.finetuned.thinking_content}
                              />
                            )}
                            <CodeBlock
                              content={
                                sxsItems.finetuned.answer_content ||
                                sxsItems.finetuned.output ||
                                ""
                              }
                            />
                            {/* Stats & Delta comparisons */}
                            {sxsItems.baseline && sxsItems.finetuned && (
                              <div className="grid grid-cols-3 gap-2 bg-[#070913]/30 border border-white/5 rounded-lg p-2.5 text-[11px] font-mono text-gray-400">
                                <div>
                                  Latency:{" "}
                                  <strong className="text-gray-300">
                                    {sxsItems.finetuned.latency_seconds?.toFixed(2)}s
                                  </strong>
                                </div>
                                <div>
                                  thinking:{" "}
                                  <strong className="text-gray-300">
                                    {sxsItems.finetuned.thinking_tokens} tok
                                  </strong>
                                  {(() => {
                                    const baseTok = sxsItems.baseline?.thinking_tokens || 0
                                    const ftTok = sxsItems.finetuned?.thinking_tokens || 0
                                    if (baseTok > 0) {
                                      const pct = Math.round(((ftTok - baseTok) / baseTok) * 100)
                                      return (
                                        <span
                                          className={`ml-1 text-[9px] font-bold ${
                                            pct <= 0 ? "text-emerald-400" : "text-red-400"
                                          }`}
                                        >
                                          {pct <= 0 ? "" : "+"}
                                          {pct}%
                                        </span>
                                      )
                                    }
                                    return null
                                  })()}
                                </div>
                                <div>
                                  Speed:{" "}
                                  <strong className="text-gray-300">
                                    {Math.round(sxsItems.finetuned.tokens_per_second || 0)} t/s
                                  </strong>
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Per-run evaluation result explorer */}
          <div className="rounded-xl border border-white/5 bg-white/[0.02] p-6 flex flex-col gap-4">
            <h3 className="font-heading font-semibold text-lg text-white">
              Per-Run Evaluation Detail Explorer
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 h-[500px] mt-2">
              {/* Sidebar filter list */}
              <div className="rounded-xl border border-white/5 bg-white/[0.02] p-4 flex flex-col gap-3 overflow-y-auto">
                <div className="flex flex-col gap-1">
                  <label className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">
                    Select Run Config:
                  </label>
                  <select
                    value={selectedRunId}
                    onChange={(e) => setSelectedRunId(e.target.value)}
                    className="bg-white/[0.03] border border-white/5 rounded-lg p-2 text-xs text-white focus:outline-none focus:border-blue-500"
                  >
                    {runIds.map((rId) => {
                      const run = data.results[rId]
                      if (!run) return null
                      const m = run.metadata
                      return (
                        <option key={rId} value={rId}>
                          {m.model.toUpperCase()} | {m.runType.toUpperCase()} |{" "}
                          {m.benchmark.toUpperCase()} ({m.promptStyle})
                        </option>
                      )
                    })}
                  </select>
                </div>

                <div className="relative">
                  <Search className="absolute left-2.5 top-2.2 h-3.5 w-3.5 text-gray-500" />
                  <input
                    type="text"
                    placeholder="Search questions..."
                    value={explorerSearch}
                    onChange={(e) => setExplorerSearch(e.target.value)}
                    className="w-full bg-white/[0.03] border border-white/5 rounded-lg py-1.5 pl-8 pr-3 text-xs text-white focus:outline-none focus:border-blue-500"
                  />
                </div>

                <div className="flex flex-col gap-1">
                  <label className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">
                    Correctness:
                  </label>
                  <select
                    value={correctnessFilter}
                    onChange={(e) => setCorrectnessFilter(e.target.value)}
                    className="bg-white/[0.03] border border-white/5 rounded-lg p-1.5 text-xs text-white focus:outline-none"
                  >
                    <option value="all">All Answers</option>
                    <option value="correct">Correct Only</option>
                    <option value="incorrect">Incorrect Only</option>
                  </select>
                </div>

                <div className="flex-1 overflow-y-auto flex flex-col gap-2 mt-2">
                  {explorerItems.map((item) => (
                    <button
                      key={item.id}
                      onClick={() => setSelectedItemId(item.id)}
                      className={`w-full text-left p-2.5 rounded-lg border transition-all duration-200 flex flex-col gap-1 cursor-pointer ${
                        selectedItemId === item.id
                          ? "bg-blue-500/10 border-blue-500/30"
                          : "bg-white/[0.01] border-transparent hover:bg-white/[0.03] hover:border-white/5"
                      }`}
                    >
                      <div className="flex justify-between items-center text-[10px]">
                        <span className="font-mono text-blue-400 font-semibold">
                          ID: {item.id}
                        </span>
                        <span className="text-amber-500 font-mono">
                          {item.thinking_tokens} tok
                        </span>
                      </div>
                      <div className="text-[11px] text-gray-300 truncate w-full">
                        {item.question}
                      </div>
                      <div className="mt-1 flex items-center gap-1.5">
                        {item.correct ? (
                          <span className="text-[8px] font-bold bg-emerald-500/15 text-emerald-400 px-1 py-0.2 rounded flex items-center gap-0.5">
                            <Check className="h-2 w-2" /> Correct
                          </span>
                        ) : (
                          <span className="text-[8px] font-bold bg-red-500/15 text-red-400 px-1 py-0.2 rounded flex items-center gap-0.5">
                            <X className="h-2 w-2" /> Fail
                          </span>
                        )}
                        {item.format_compliance === false && (
                          <span className="text-[8px] font-bold bg-yellow-500/10 text-yellow-400 px-1 py-0.2 rounded">
                            Format Fail
                          </span>
                        )}
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Detail view area */}
              <div className="col-span-1 md:col-span-2 rounded-xl border border-white/5 bg-white/[0.02] flex flex-col overflow-y-auto p-4 gap-4">
                {!selectedExplorerItem ? (
                  <div className="flex-1 flex flex-col items-center justify-center gap-2 text-gray-500 italic text-xs">
                    Select a question from the sidebar to inspect detailed output.
                  </div>
                ) : (
                  <div className="flex flex-col gap-4">
                    <div className="flex justify-between items-center border-b border-white/5 pb-2">
                      <div className="flex flex-col">
                        <span className="text-[12px] font-semibold text-white">
                          Sample Detail (ID #{selectedExplorerItem.id})
                        </span>
                        <span className="text-[10px] text-gray-500 font-mono mt-0.5">
                          Run Config: {selectedRunId}
                        </span>
                      </div>
                      <span
                        className={`text-[10px] font-bold px-1.5 py-0.5 rounded uppercase ${
                          selectedExplorerItem.correct
                            ? "bg-emerald-500/10 text-emerald-400"
                            : "bg-red-500/10 text-red-400"
                        }`}
                      >
                        {selectedExplorerItem.correct ? "Correct Answer" : "Incorrect Answer"}
                      </span>
                    </div>

                    <div className="rounded-lg border border-white/5 bg-white/[0.01] p-3 flex flex-col gap-2">
                      <span className="text-[10px] font-bold text-gray-500 uppercase tracking-wider block">
                        Question
                      </span>
                      <p className="text-[13px] text-gray-200 leading-relaxed font-sans">
                        {selectedExplorerItem.question}
                      </p>
                      <div className="flex gap-4 text-[11px] border-t border-white/5 pt-2 mt-1">
                        <div>
                          <span className="text-gray-500">Ground Truth: </span>
                          <strong className="text-emerald-400">
                            {selectedExplorerItem.ground_truth}
                          </strong>
                        </div>
                        <div>
                          <span className="text-gray-500">Predicted Answer: </span>
                          <strong className="text-blue-400">
                            {selectedExplorerItem.predicted_answer || "N/A"}
                          </strong>
                        </div>
                      </div>
                    </div>

                    {selectedExplorerItem.thinking_content && (
                      <ThinkingBubble content={selectedExplorerItem.thinking_content} />
                    )}

                    {(selectedExplorerItem.answer_content || selectedExplorerItem.output) && (
                      <CodeBlock
                        content={
                          selectedExplorerItem.answer_content ||
                          selectedExplorerItem.output ||
                          ""
                        }
                      />
                    )}

                    {/* Stats metrics block */}
                    <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 bg-[#070913]/30 border border-white/5 rounded-lg p-3 text-[11px] font-mono text-gray-400 mt-2">
                      <div>
                        Thinking:{" "}
                        <strong className="text-gray-300">
                          {selectedExplorerItem.thinking_tokens} tok
                        </strong>
                      </div>
                      <div>
                        Answer:{" "}
                        <strong className="text-gray-300">
                          {selectedExplorerItem.answer_tokens} tok
                        </strong>
                      </div>
                      <div>
                        Total:{" "}
                        <strong className="text-gray-300">
                          {selectedExplorerItem.total_tokens} tok
                        </strong>
                      </div>
                      <div>
                        Latency:{" "}
                        <strong className="text-gray-300">
                          {selectedExplorerItem.latency_seconds?.toFixed(2)}s
                        </strong>
                      </div>
                      <div>
                        Speed:{" "}
                        <strong className="text-gray-300">
                          {Math.round(selectedExplorerItem.tokens_per_second || 0)} tok/s
                        </strong>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
