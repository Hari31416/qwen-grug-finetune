import { useState, useMemo, useEffect } from "react"
import type { WorkspaceData } from "@/types"
import { Search, Info, Check, X, Circle } from "lucide-react"
import { CodeBlock } from "@/components/ui/CodeBlock"
import { ThinkingBubble } from "@/components/ui/ThinkingBubble"

interface ExplorerViewProps {
  data: WorkspaceData
}

export function ExplorerView({ data }: ExplorerViewProps) {
  const [search, setSearch] = useState("")
  const [sourceFilter, setSourceFilter] = useState("all")
  const [statusFilter, setStatusFilter] = useState("all")
  const [selectedId, setSelectedId] = useState<string | null>(null)

  // Get list of all IDs
  const allIds = useMemo(() => {
    const ids = new Set([
      ...Object.keys(data.prompts),
      ...Object.keys(data.rawTraces),
      ...Object.keys(data.compressedTraces),
      ...Object.keys(data.validatedTraces),
    ])
    return Array.from(ids).sort()
  }, [data])

  // Map IDs to their respective records and filter
  const items = useMemo(() => {
    return allIds
      .map((id) => {
        const prompt = data.prompts[id]
        const raw = data.rawTraces[id]
        const comp = data.compressedTraces[id]
        const val = data.validatedTraces[id]

        const source = prompt?.source || raw?.source || id.split("-")[0] || "unknown"
        const promptText = prompt?.prompt || raw?.prompt || ""
        const hasRaw = !!raw?.raw_thinking
        const hasComp = !!comp?.compressed_thinking
        const isValidated = !!val?.compressed_thinking || data.validatedTraces[id] !== undefined
        const isCorrect = raw?.raw_answer_correct === true
        const isIncorrect = raw?.raw_answer_correct === false

        // Check if matching SFT rows
        let inSft = false
        if (data.sftFormatted.length > 0) {
          inSft = data.sftFormatted.some(
            (row) => row.text && row.text.includes(promptText.substring(0, 30))
          )
        }

        return {
          id,
          source,
          promptText,
          prompt,
          raw,
          comp,
          val,
          hasRaw,
          hasComp,
          isValidated,
          isCorrect,
          isIncorrect,
          inSft,
        }
      })
      .filter((item) => {
        // Search filter
        if (
          search.trim() &&
          !item.id.toLowerCase().includes(search.toLowerCase()) &&
          !item.promptText.toLowerCase().includes(search.toLowerCase())
        ) {
          return false
        }

        // Source filter
        if (sourceFilter !== "all" && item.source !== sourceFilter) {
          return false
        }

        // Status filter
        if (statusFilter === "raw" && !item.hasRaw) return false
        if (statusFilter === "compressed" && !item.hasComp) return false
        if (statusFilter === "validated" && !item.isValidated) return false
        if (statusFilter === "incorrect" && !item.isIncorrect) return false
        if (statusFilter === "sft" && !item.inSft) return false

        return true
      })
  }, [allIds, data, search, sourceFilter, statusFilter])

  // Select first item if nothing is selected or if selected item isn't in list
  useEffect(() => {
    if (items.length > 0) {
      if (!selectedId || !items.some((item) => item.id === selectedId)) {
        setSelectedId(items[0].id)
      }
    } else {
      setSelectedId(null)
    }
  }, [items, selectedId])

  const selectedItem = useMemo(() => {
    return items.find((item) => item.id === selectedId) || null
  }, [items, selectedId])

  // SFT format highlighter
  const renderSftText = (text: string) => {
    // Regex splits by tag markers
    const parts = text.split(
      /(<｜beginofsentence｜>|<｜User｜>|<｜Assistant｜>|<think>|<\/think>|<｜endofsentence｜>)/g
    )

    return (
      <div className="font-mono text-[13px] leading-relaxed text-gray-300">
        {parts.map((part, idx) => {
          if (part === "<｜beginofsentence｜>" || part === "<｜endofsentence｜>") {
            return (
              <span
                key={idx}
                className="inline-block font-semibold px-1 py-0.5 rounded text-[11px] bg-purple-500/15 border border-purple-500/30 text-purple-400 mx-0.5 my-0.5"
              >
                {part}
              </span>
            )
          }
          if (part === "<｜User｜>") {
            return (
              <span
                key={idx}
                className="inline-block font-semibold px-1 py-0.5 rounded text-[11px] bg-blue-500/15 border border-blue-500/30 text-blue-400 mx-0.5 my-0.5"
              >
                {part}
              </span>
            )
          }
          if (part === "<｜Assistant｜>") {
            return (
              <span
                key={idx}
                className="inline-block font-semibold px-1 py-0.5 rounded text-[11px] bg-pink-500/15 border border-pink-500/30 text-pink-400 mx-0.5 my-0.5"
              >
                {part}
              </span>
            )
          }
          if (part === "<think>" || part === "</think>") {
            return (
              <span
                key={idx}
                className="inline-block font-semibold px-1 py-0.5 rounded text-[11px] bg-amber-500/15 border border-amber-500/30 text-amber-400 mx-0.5 my-0.5"
              >
                {part}
              </span>
            )
          }
          return <span key={idx}>{part}</span>
        })}
      </div>
    )
  }

  // Find matching SFT Row if any
  const matchingSftRow = useMemo(() => {
    if (!selectedItem || data.sftFormatted.length === 0) return null
    return (
      data.sftFormatted.find(
        (row) =>
          row.text && row.text.includes(selectedItem.promptText.substring(0, 30))
      ) || null
    )
  }, [selectedItem, data])

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 h-[calc(100vh-220px)] min-h-[500px]">
      {/* Sidebar List */}
      <div className="rounded-xl border border-white/5 bg-white/[0.02] p-4 flex flex-col gap-4 overflow-y-auto">
        <div className="flex flex-col gap-3">
          <div className="relative">
            <Search className="absolute left-3 top-2.5 h-4 w-4 text-gray-500" />
            <input
              type="text"
              placeholder="Search prompts..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full bg-white/[0.03] border border-white/5 rounded-lg py-2 pl-9 pr-4 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            />
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div className="flex flex-col gap-1">
              <label className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">
                Source
              </label>
              <select
                value={sourceFilter}
                onChange={(e) => setSourceFilter(e.target.value)}
                className="bg-white/[0.03] border border-white/5 rounded-lg p-2 text-xs text-white focus:outline-none focus:border-blue-500"
              >
                <option value="all">All Sources</option>
                {data.sources.map((src) => (
                  <option key={src} value={src}>
                    {src.toUpperCase()}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex flex-col gap-1">
              <label className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">
                Status
              </label>
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="bg-white/[0.03] border border-white/5 rounded-lg p-2 text-xs text-white focus:outline-none focus:border-blue-500"
              >
                <option value="all">All Statuses</option>
                <option value="raw">Has Raw Trace</option>
                <option value="compressed">Has Compressed</option>
                <option value="validated">Validated</option>
                <option value="sft">SFT Splits</option>
                <option value="incorrect">Incorrect Raw</option>
              </select>
            </div>
          </div>

          <div className="flex justify-between items-center text-[11px] text-gray-500 border-b border-white/5 pb-2">
            <span>{items.length} listed</span>
            <span>{allIds.length} total</span>
          </div>
        </div>

        {/* List items */}
        <div className="flex-1 overflow-y-auto flex flex-col gap-2 pr-1">
          {items.length === 0 ? (
            <div className="text-center py-8 text-gray-500 text-xs italic">
              No items match filters
            </div>
          ) : (
            items.map((item) => (
              <button
                key={item.id}
                onClick={() => setSelectedId(item.id)}
                className={`w-full text-left p-3 rounded-lg border transition-all duration-200 flex flex-col gap-1.5 cursor-pointer ${selectedId === item.id
                  ? "bg-blue-500/10 border-blue-500/30"
                  : "bg-white/[0.01] border-transparent hover:bg-white/[0.03] hover:border-white/5"
                  }`}
              >
                <div className="flex justify-between items-center">
                  <span className="font-mono font-medium text-[12px] text-blue-400">
                    {item.id}
                  </span>
                  <span className="text-[10px] font-bold text-gray-500 bg-white/5 px-1.5 py-0.5 rounded uppercase tracking-wider">
                    {item.source}
                  </span>
                </div>
                <div className="text-[12px] text-gray-300 truncate w-full">
                  {item.promptText}
                </div>
                <div className="flex gap-1.5 items-center mt-1">
                  {/* Badges */}
                  {item.isValidated ? (
                    <span className="text-[9px] font-bold bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 px-1 py-0.2 rounded uppercase">
                      Val
                    </span>
                  ) : item.hasComp ? (
                    <span className="text-[9px] font-bold bg-amber-500/15 border border-amber-500/30 text-amber-400 px-1 py-0.2 rounded uppercase">
                      Comp
                    </span>
                  ) : item.hasRaw ? (
                    <span className="text-[9px] font-bold bg-blue-500/15 border border-blue-500/30 text-blue-400 px-1 py-0.2 rounded uppercase">
                      Raw
                    </span>
                  ) : (
                    <span className="text-[9px] font-bold bg-gray-500/15 border border-gray-500/30 text-gray-400 px-1 py-0.2 rounded uppercase">
                      Prm
                    </span>
                  )}

                  {item.hasRaw && (
                    item.isCorrect ? (
                      <span className="text-[9px] font-bold bg-emerald-500/15 text-emerald-400 px-1 py-0.2 rounded flex items-center gap-0.5">
                        <Check className="h-2 w-2" /> Correct
                      </span>
                    ) : (
                      <span className="text-[9px] font-bold bg-red-500/15 text-red-400 px-1 py-0.2 rounded flex items-center gap-0.5">
                        <X className="h-2 w-2" /> Fail
                      </span>
                    )
                  )}
                </div>
              </button>
            ))
          )}
        </div>
      </div>

      {/* Detail Area */}
      <div className="col-span-1 md:col-span-2 rounded-xl border border-white/5 bg-white/[0.02] flex flex-col overflow-hidden h-full">
        {!selectedItem ? (
          <div className="flex-1 flex flex-col items-center justify-center gap-3 text-gray-500">
            <Info className="h-10 w-10 text-gray-600 animate-pulse" />
            <span className="text-sm">Select a sample from the list to view its pipeline history</span>
          </div>
        ) : (
          <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-6">
            {/* Header */}
            <div className="flex justify-between items-start border-b border-white/5 pb-4">
              <div className="flex flex-col gap-1">
                <h3 className="font-heading font-semibold text-lg text-white">
                  Sample Trace History
                </h3>
                <div className="flex gap-2 items-center">
                  <span className="font-mono text-blue-400 text-sm">{selectedItem.id}</span>
                  <span className="text-[10px] font-bold text-gray-400 bg-white/5 px-1.5 py-0.5 rounded uppercase">
                    {selectedItem.source}
                  </span>
                  {selectedItem.hasRaw && (
                    <span
                      className={`text-[10px] font-bold px-1.5 py-0.5 rounded uppercase ${selectedItem.isCorrect
                        ? "bg-emerald-500/10 text-emerald-400"
                        : "bg-red-500/10 text-red-400"
                        }`}
                    >
                      {selectedItem.isCorrect ? "Correct answer" : "Incorrect answer"}
                    </span>
                  )}
                </div>
              </div>
            </div>

            {/* Stepper progress indicator */}
            <div className="grid grid-cols-5 gap-2 bg-white/[0.01] border border-white/5 rounded-lg p-3 text-[11px] select-none font-medium">
              {[
                { label: "Prompt", active: true },
                { label: "Raw Trace", active: selectedItem.hasRaw },
                { label: "Compressed", active: selectedItem.hasComp },
                { label: "Validated", active: selectedItem.isValidated },
                { label: "SFT Row", active: !!matchingSftRow },
              ].map((step, idx) => (
                <div
                  key={idx}
                  className={`flex items-center justify-center gap-1 py-1 rounded text-center ${step.active ? "text-blue-400" : "text-gray-600"
                    }`}
                >
                  {step.active ? (
                    <Check className="h-3.5 w-3.5 text-emerald-400" />
                  ) : (
                    <Circle className="h-3.5 w-3.5" />
                  )}
                  <span className="hidden sm:inline">{step.label}</span>
                </div>
              ))}
            </div>

            {/* Card 1: Prompt details */}
            <div className="rounded-xl border border-white/5 bg-white/[0.01] p-4 flex flex-col gap-3">
              <span className="text-[11px] font-bold text-gray-500 uppercase tracking-wider">
                Stage 1: Prompt Input
              </span>
              <CodeBlock content={selectedItem.promptText} />
              <div className="flex gap-6 text-[12px] border-t border-white/5 pt-3">
                {selectedItem.prompt?.choices && selectedItem.prompt.choices.length > 0 && (
                  <div>
                    <span className="text-gray-500 font-medium">Choices: </span>
                    <strong className="text-gray-300">
                      {selectedItem.prompt.choices.join(", ")}
                    </strong>
                  </div>
                )}
                {selectedItem.prompt?.ground_truth && (
                  <div>
                    <span className="text-gray-500 font-medium">Ground Truth: </span>
                    <strong className="text-emerald-400">{selectedItem.prompt.ground_truth}</strong>
                  </div>
                )}
              </div>
            </div>

            {/* Card 2: Raw Reasoning */}
            {selectedItem.hasRaw && selectedItem.raw?.raw_thinking && (
              <div className="rounded-xl border border-white/5 bg-white/[0.01] p-4 flex flex-col gap-3">
                <div className="flex justify-between items-center">
                  <span className="text-[11px] font-bold text-gray-500 uppercase tracking-wider">
                    Stage 2: Raw Verbose Reasoning
                  </span>
                  <span
                    className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${selectedItem.isCorrect
                      ? "bg-emerald-500/10 text-emerald-400"
                      : "bg-red-500/10 text-red-400"
                      }`}
                  >
                    {selectedItem.isCorrect ? "Correct" : "Incorrect"}
                  </span>
                </div>

                <ThinkingBubble content={selectedItem.raw.raw_thinking} />

                <div className="flex justify-between items-center text-[12px] border-t border-white/5 pt-3">
                  <div>
                    <span className="text-gray-500 font-medium">Model Output Answer: </span>
                    <strong
                      className={selectedItem.isCorrect ? "text-emerald-400" : "text-red-400"}
                    >
                      {selectedItem.raw.raw_answer || "N/A"}
                    </strong>
                  </div>
                  <div className="text-gray-500 font-mono text-[11px]">
                    Chars: {selectedItem.raw.raw_thinking.length} | Est. Tokens:{" "}
                    {Math.round(selectedItem.raw.raw_thinking.length / 4)}
                  </div>
                </div>
              </div>
            )}

            {/* Card 3: Compression comparison side-by-side */}
            {selectedItem.hasComp &&
              selectedItem.comp?.compressed_thinking &&
              selectedItem.raw?.raw_thinking && (
                <div className="rounded-xl border border-white/5 bg-white/[0.01] p-4 flex flex-col gap-3">
                  <span className="text-[11px] font-bold text-gray-500 uppercase tracking-wider">
                    Stage 3: Chain-of-thought Compression (Grug Style)
                  </span>

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    <div className="flex flex-col gap-2">
                      <span className="text-[10px] font-bold text-red-400 uppercase tracking-wider flex items-center gap-1 select-none">
                        Verbose Thinking (Raw)
                      </span>
                      <div className="max-h-[220px] overflow-y-auto border border-white/5 rounded-lg bg-[#070913]/30 p-3 font-mono text-xs leading-relaxed text-gray-400">
                        {selectedItem.raw.raw_thinking}
                      </div>
                    </div>

                    <div className="flex flex-col gap-2">
                      <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider flex items-center gap-1 select-none">
                        Telegraphic Thinking (Compressed)
                      </span>
                      <div className="max-h-[220px] overflow-y-auto border border-white/5 rounded-lg bg-[#070913]/30 p-3 font-mono text-xs leading-relaxed text-gray-300">
                        {selectedItem.comp.compressed_thinking}
                      </div>
                    </div>
                  </div>

                  {(() => {
                    const rawLen = selectedItem.raw?.raw_thinking?.length || 0
                    const compLen = selectedItem.comp?.compressed_thinking?.length || 0
                    const savings = rawLen > 0 ? ((rawLen - compLen) / rawLen * 100).toFixed(1) : "0.0"
                    return (
                      <div className="bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 p-2.5 rounded-lg text-center text-[12px] font-semibold mt-1">
                        Token Reduction: {savings}% (Raw: {rawLen} chars | Compressed: {compLen} chars | Saved: {rawLen - compLen} chars)
                      </div>
                    )
                  })()}
                </div>
              )}

            {/* Card 4: SFT Row */}
            {matchingSftRow && matchingSftRow.text && (
              <div className="rounded-xl border border-white/5 bg-white/[0.01] p-4 flex flex-col gap-3">
                <span className="text-[11px] font-bold text-gray-500 uppercase tracking-wider">
                  Stage 5: Final SFT Formatted text (Finetuning Input)
                </span>
                <div className="border border-white/5 bg-[#070913] p-4 rounded-lg overflow-x-auto whitespace-pre-wrap max-h-[300px] overflow-y-auto">
                  {renderSftText(matchingSftRow.text)}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
