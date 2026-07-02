import React, { useState, useRef, useMemo } from "react"
import { useWorkspace } from "@/hooks/useWorkspace"
import { WelcomeBanner } from "@/components/WelcomeBanner"
import { StatCards } from "@/components/layout/StatCards"
import { OverviewView } from "@/views/OverviewView"
import { ExplorerView } from "@/views/ExplorerView"
import { AnalyticsView } from "@/views/AnalyticsView"
import { EvaluationView } from "@/views/EvaluationView"
import { StyleGuideView } from "@/views/StyleGuideView"
import { Brain, FolderOpen, ChartPie, ListTodo, LineChart, Table2, BookOpen, AlertCircle, RefreshCw } from "lucide-react"

export function App() {
  const {
    workspaceData,
    isLoading,
    error,
    loadFromFiles,
    loadDemo,
  } = useWorkspace()

  const [activeTab, setActiveTab] = useState<"overview" | "explorer" | "analytics" | "results" | "styleguide">("overview")
  const fileInputRef = useRef<HTMLInputElement>(null)

  const isDataLoaded = useMemo(() => {
    return (
      Object.keys(workspaceData.prompts).length > 0 ||
      Object.keys(workspaceData.rawTraces).length > 0 ||
      Object.keys(workspaceData.results).length > 0
    )
  }, [workspaceData])

  // Count calculations
  const promptCount = Object.keys(workspaceData.prompts).length
  const rawCount = Object.keys(workspaceData.rawTraces).length
  const compressedCount = Object.keys(workspaceData.compressedTraces).length
  const validatedCount = Object.keys(workspaceData.validatedTraces).length
  const sftCount = workspaceData.sftFormatted.length

  const { rawAccuracy, rawCorrectCount, rawTotalEval } = useMemo(() => {
    let correctCount = 0
    let totalEvaluated = 0
    Object.values(workspaceData.rawTraces).forEach((trace) => {
      if (trace.raw_answer_correct !== undefined) {
        totalEvaluated++
        if (trace.raw_answer_correct) correctCount++
      }
    })
    const rawAccuracy =
      totalEvaluated > 0 ? ((correctCount / totalEvaluated) * 100).toFixed(1) : "0.0"
    return { rawAccuracy, rawCorrectCount: correctCount, rawTotalEval: totalEvaluated }
  }, [workspaceData.rawTraces])

  const compressionSavings = useMemo(() => {
    let totalRawLen = 0
    let totalCompLen = 0
    Object.keys(workspaceData.compressedTraces).forEach((id) => {
      const raw = workspaceData.rawTraces[id]
      const comp = workspaceData.compressedTraces[id]
      if (raw && comp && raw.raw_thinking && comp.compressed_thinking) {
        totalRawLen += raw.raw_thinking.length
        totalCompLen += comp.compressed_thinking.length
      }
    })
    return totalRawLen > 0
      ? ((totalRawLen - totalCompLen) / totalRawLen * 100).toFixed(0)
      : "0"
  }, [workspaceData.compressedTraces, workspaceData.rawTraces])

  const validationAccepted = workspaceData.validationReport?.accepted ?? validatedCount

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      loadFromFiles(Array.from(e.target.files))
    }
  }

  const triggerPicker = () => {
    fileInputRef.current?.click()
  }

  // Determine header model badge
  const modelBadge = useMemo(() => {
    const resultsKeys = Object.keys(workspaceData.results)
    if (resultsKeys.length > 0) {
      const firstRun = workspaceData.results[resultsKeys[0]]
      return firstRun?.metadata?.model || "Distill-Qwen-1.5B"
    }
    return "Distill-Qwen-1.5B"
  }, [workspaceData.results])

  return (
    <div className="min-h-screen bg-[#0b0f19] text-gray-100 flex flex-col font-sans relative antialiased selection:bg-blue-500/30 selection:text-white">
      {/* Top Header */}
      <header className="sticky top-0 z-50 bg-[#0b0f19]/80 backdrop-blur-md border-b border-white/5 px-6 py-4 flex flex-col sm:flex-row justify-between items-center gap-4">
        <div className="flex items-center gap-2.5">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center text-white shadow-lg shadow-blue-500/20">
            <Brain className="h-5 w-5" />
          </div>
          <span className="font-heading font-bold text-lg text-white tracking-tight">
            Grug Reasoning
          </span>
          <span className="text-[10px] font-semibold bg-blue-500/10 border border-blue-500/20 text-blue-400 px-2 py-0.5 rounded uppercase tracking-wider">
            {modelBadge}
          </span>
          {workspaceData.isDemo && (
            <span className="text-[10px] font-semibold bg-amber-500/10 border border-amber-500/20 text-amber-400 px-2 py-0.5 rounded uppercase tracking-wider">
              Demo Data
            </span>
          )}
        </div>

        {/* Controls */}
        <div className="flex items-center gap-4">
          {isDataLoaded && (
            <nav className="flex bg-white/[0.03] border border-white/5 rounded-lg p-0.5 select-none">
              <button
                onClick={() => setActiveTab("overview")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all cursor-pointer ${
                  activeTab === "overview"
                    ? "bg-white/10 text-white shadow"
                    : "text-gray-400 hover:text-white"
                }`}
              >
                <ChartPie className="h-3.5 w-3.5" />
                Overview
              </button>

              <button
                onClick={() => setActiveTab("explorer")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all cursor-pointer ${
                  activeTab === "explorer"
                    ? "bg-white/10 text-white shadow"
                    : "text-gray-400 hover:text-white"
                }`}
              >
                <ListTodo className="h-3.5 w-3.5" />
                Trace Explorer
              </button>

              <button
                onClick={() => setActiveTab("analytics")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all cursor-pointer ${
                  activeTab === "analytics"
                    ? "bg-white/10 text-white shadow"
                    : "text-gray-400 hover:text-white"
                }`}
              >
                <LineChart className="h-3.5 w-3.5" />
                Analytics
              </button>

              <button
                onClick={() => setActiveTab("results")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all cursor-pointer ${
                  activeTab === "results"
                    ? "bg-white/10 text-white shadow"
                    : "text-gray-400 hover:text-white"
                }`}
              >
                <Table2 className="h-3.5 w-3.5" />
                Evaluation
              </button>

              <button
                onClick={() => setActiveTab("styleguide")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all cursor-pointer ${
                  activeTab === "styleguide"
                    ? "bg-white/10 text-white shadow"
                    : "text-gray-400 hover:text-white"
                }`}
              >
                <BookOpen className="h-3.5 w-3.5" />
                Style Guide
              </button>
            </nav>
          )}

          {/* Directory Picker */}
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            style={{ display: "none" }}
            {...({
              webkitdirectory: "",
              directory: "",
            } as any)}
            multiple
          />
          
          {isDataLoaded && (
            <button
              onClick={triggerPicker}
              disabled={isLoading}
              className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-blue-500 hover:bg-blue-600 text-white font-medium text-xs transition-all cursor-pointer shadow-lg shadow-blue-500/10 disabled:opacity-50"
            >
              <FolderOpen className="h-3.5 w-3.5" />
              Change Folder
            </button>
          )}
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 w-full max-w-[1440px] mx-auto p-6 flex flex-col gap-8">
        {!isDataLoaded ? (
          <div className="my-auto">
            <WelcomeBanner
              onFilesSelected={loadFromFiles}
              onLoadDemo={loadDemo}
              isLoading={isLoading}
            />
          </div>
        ) : (
          <>
            {/* Stat Row */}
            <StatCards
              promptCount={promptCount}
              rawCount={rawCount}
              rawAccuracy={rawAccuracy}
              rawCorrectCount={rawCorrectCount}
              rawTotalEval={rawTotalEval}
              compressedCount={compressedCount}
              compressionSavings={compressionSavings}
              validatedCount={validatedCount}
              validationAccepted={validationAccepted}
            />

            {/* Sub-view Content */}
            <div className="flex-1">
              {activeTab === "overview" && (
                <OverviewView
                  data={workspaceData}
                  promptCount={promptCount}
                  rawCount={rawCount}
                  compressedCount={compressedCount}
                  validatedCount={validatedCount}
                  sftCount={sftCount}
                />
              )}
              {activeTab === "explorer" && <ExplorerView data={workspaceData} />}
              {activeTab === "analytics" && <AnalyticsView data={workspaceData} />}
              {activeTab === "results" && <EvaluationView data={workspaceData} />}
              {activeTab === "styleguide" && <StyleGuideView />}
            </div>
          </>
        )}
      </main>

      {/* Loading state indicator toast */}
      {isLoading && (
        <div className="fixed bottom-6 right-6 bg-blue-600 border border-blue-500 shadow-xl rounded-lg py-3 px-5 flex items-center gap-2.5 animate-bounce select-none">
          <RefreshCw className="h-4.5 w-4.5 text-white animate-spin" />
          <span className="text-white text-xs font-semibold">Processing files...</span>
        </div>
      )}

      {/* Error state alert toast */}
      {error && (
        <div className="fixed bottom-6 right-6 bg-red-900/90 border border-red-700/50 shadow-xl rounded-lg py-3.5 px-5 flex items-center gap-2.5 select-none animate-in fade-in slide-in-from-bottom-5">
          <AlertCircle className="h-5 w-5 text-red-400" />
          <span className="text-red-200 text-xs font-semibold max-w-[280px] leading-relaxed">
            {error}
          </span>
        </div>
      )}
    </div>
  )
}

export default App
