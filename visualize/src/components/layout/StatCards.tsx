import { ArrowDown, Check, Database, Sparkles } from "lucide-react"

interface StatCardsProps {
  promptCount: number
  rawCount: number
  rawAccuracy: string
  rawCorrectCount: number
  rawTotalEval: number
  compressedCount: number
  compressionSavings: string
  validatedCount: number
  validationAccepted: number
}

export function StatCards({
  promptCount,
  rawCount,
  rawAccuracy,
  rawCorrectCount,
  rawTotalEval,
  compressedCount,
  compressionSavings,
  validatedCount,
  validationAccepted,
}: StatCardsProps) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 w-full select-none">
      {/* Prompts */}
      <div className="relative overflow-hidden rounded-xl border border-white/5 bg-white/[0.02] p-6 flex flex-col gap-2 transition-all duration-300 hover:-translate-y-1 hover:border-blue-500/30 hover:shadow-[0_8px_30px_rgb(0,0,0,0.12)] group">
        <span className="text-[13px] font-medium text-gray-400">SFT Prompts</span>
        <span className="font-heading font-bold text-3xl text-white">
          {promptCount.toLocaleString()}
        </span>
        <span className="text-[11px] text-gray-500 flex items-center gap-1">
          <Database className="h-3 w-3" />
          Total initial samples
        </span>
        <div className="absolute bottom-0 left-0 right-0 h-[3px] bg-blue-500/80 transition-all duration-300 group-hover:h-[4px]" />
      </div>

      {/* Raw Traces */}
      <div className="relative overflow-hidden rounded-xl border border-white/5 bg-white/[0.02] p-6 flex flex-col gap-2 transition-all duration-300 hover:-translate-y-1 hover:border-amber-500/30 hover:shadow-[0_8px_30px_rgb(0,0,0,0.12)] group">
        <span className="text-[13px] font-medium text-gray-400">Raw Thinking Traces</span>
        <span className="font-heading font-bold text-3xl text-white">
          {rawCount.toLocaleString()}
        </span>
        <span className="text-[11px] text-amber-400 flex items-center gap-1 font-medium">
          <Sparkles className="h-3 w-3" />
          Accuracy: {rawAccuracy}% ({rawCorrectCount}/{rawTotalEval})
        </span>
        <div className="absolute bottom-0 left-0 right-0 h-[3px] bg-amber-500/80 transition-all duration-300 group-hover:h-[4px]" />
      </div>

      {/* Compressed */}
      <div className="relative overflow-hidden rounded-xl border border-white/5 bg-white/[0.02] p-6 flex flex-col gap-2 transition-all duration-300 hover:-translate-y-1 hover:border-emerald-500/30 hover:shadow-[0_8px_30px_rgb(0,0,0,0.12)] group">
        <span className="text-[13px] font-medium text-gray-400">Compressed Traces</span>
        <span className="font-heading font-bold text-3xl text-white">
          {compressedCount.toLocaleString()}
        </span>
        <span className="text-[11px] text-emerald-400 flex items-center gap-1 font-medium">
          <ArrowDown className="h-3 w-3" />
          {compressionSavings}% chars saved
        </span>
        <div className="absolute bottom-0 left-0 right-0 h-[3px] bg-emerald-500/80 transition-all duration-300 group-hover:h-[4px]" />
      </div>

      {/* Validated */}
      <div className="relative overflow-hidden rounded-xl border border-white/5 bg-white/[0.02] p-6 flex flex-col gap-2 transition-all duration-300 hover:-translate-y-1 hover:border-red-500/30 hover:shadow-[0_8px_30px_rgb(0,0,0,0.12)] group">
        <span className="text-[13px] font-medium text-gray-400">Validated Traces</span>
        <span className="font-heading font-bold text-3xl text-white">
          {validatedCount.toLocaleString()}
        </span>
        <span className="text-[11px] text-red-400 flex items-center gap-1 font-medium">
          <Check className="h-3 w-3" />
          Accepted: {validationAccepted}
        </span>
        <div className="absolute bottom-0 left-0 right-0 h-[3px] bg-red-500/80 transition-all duration-300 group-hover:h-[4px]" />
      </div>
    </div>
  )
}
