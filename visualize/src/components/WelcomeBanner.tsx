import { useState } from 'react'
import { Sparkles, Database, Loader2, ExternalLink } from 'lucide-react'

interface WelcomeBannerProps {
  onLoadFromHF: (iterationId: string) => void
  isLoading: boolean
}

export function WelcomeBanner({
  onLoadFromHF,
  isLoading,
}: WelcomeBannerProps) {
  const [selectedIteration, setSelectedIteration] = useState<string>('deepseek-r1-7b-full')

  return (
    <div className="w-full max-w-md mx-auto rounded-2xl border border-white/10 bg-gradient-to-b from-[#151c2e] to-[#0c101d] p-8 text-center flex flex-col items-center gap-6 shadow-2xl shadow-black/60 relative overflow-hidden">
      {/* Background glow accent */}
      <div className="absolute -top-24 left-1/2 -translate-x-1/2 w-64 h-64 bg-blue-500/10 rounded-full blur-3xl pointer-events-none" />

      {/* Top Badge & Icon */}
      <div className="flex flex-col items-center gap-3">
        <div className="w-14 h-14 bg-gradient-to-br from-blue-500/20 to-indigo-500/10 border border-blue-500/30 text-blue-400 rounded-xl flex items-center justify-center shadow-[0_0_20px_rgba(59,130,246,0.2)]">
          <Database className="h-7 w-7" />
        </div>
      </div>

      {/* Title & Description */}
      <div className="flex flex-col gap-1.5">
        <h2 className="font-heading font-bold text-2xl text-white tracking-tight">
          Grug Reasoning Visualizer
        </h2>
        <p className="text-gray-400 text-xs leading-relaxed max-w-sm">
          Inspect concise reasoning traces, format compliance, and full benchmark metrics across 1.5B and 7B model evaluations.
        </p>
      </div>

      {/* Form Controls */}
      <div className="w-full flex flex-col gap-3.5">
        <div className="flex flex-col gap-1.5 text-left">
          <label className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider">
            Select Evaluation Run
          </label>
          <div className="relative">
            <select
              value={selectedIteration}
              onChange={(e) => setSelectedIteration(e.target.value)}
              disabled={isLoading}
              className="w-full appearance-none bg-[#090d16] border border-white/10 hover:border-white/20 rounded-xl px-3.5 py-2.5 text-xs text-gray-100 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all cursor-pointer disabled:opacity-50 pr-8"
            >
              <option value="deepseek-r1-7b-full" className="bg-[#0b0f19] text-gray-100">
                DeepSeek-R1-7B: Full GSM8K (Baseline, SFT, DPO - 1,319)
              </option>
              <option value="iteration-2-regularized" className="bg-[#0b0f19] text-gray-100">
                DeepSeek-R1-1.5B: Iteration 2 (Regularized / Final)
              </option>
              <option value="iteration-1" className="bg-[#0b0f19] text-gray-100">
                DeepSeek-R1-1.5B: Iteration 1 (Proof of Concept)
              </option>
            </select>
            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2.5 text-gray-400">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
              </svg>
            </div>
          </div>
        </div>

        <button
          onClick={() => onLoadFromHF(selectedIteration)}
          disabled={isLoading}
          className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white font-medium text-xs shadow-lg shadow-blue-500/25 disabled:opacity-50 transition-all cursor-pointer hover:shadow-blue-500/40 active:scale-[0.99]"
        >
          {isLoading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin text-white" />
              <span>Fetching from Hugging Face...</span>
            </>
          ) : (
            <>
              <Sparkles className="h-3.5 w-3.5 text-blue-200" />
              <span>Load Evaluation Data</span>
            </>
          )}
        </button>
      </div>

      {/* Source Repo Link */}
      <div className="flex items-center justify-center gap-1 text-[11px] text-gray-500 border-t border-white/5 pt-4 w-full">
        <span>Dataset repo:</span>
        <a
          href="https://huggingface.co/datasets/hari31416/grug-reasoning-data-and-benchmarks"
          target="_blank"
          rel="noreferrer"
          className="text-blue-400 hover:text-blue-300 inline-flex items-center gap-1 transition-colors"
        >
          hari31416/grug-reasoning-data-and-benchmarks
          <ExternalLink className="h-3 w-3" />
        </a>
      </div>
    </div>
  )
}
