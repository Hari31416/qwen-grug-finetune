import type { WorkspaceData } from "@/types"
import { CheckCircle2, HelpCircle } from "lucide-react"

interface OverviewViewProps {
  data: WorkspaceData
  promptCount: number
  rawCount: number
  compressedCount: number
  validatedCount: number
  sftCount: number
}

export function OverviewView({
  data,
  promptCount,
  rawCount,
  compressedCount,
  validatedCount,
  sftCount,
}: OverviewViewProps) {
  const report = data.validationReport

  const trainCount = data.sftFormatted.filter((r) => r.type === "train").length
  const validCount = data.sftFormatted.filter((r) => r.type === "valid").length

  const renderStep = (
    stepNum: number,
    title: string,
    desc: string,
    badgeText: string,
    isCompleted: boolean,
    isPending: boolean
  ) => {
    return (
      <div className="flex gap-4 items-start relative pb-6 group">
        {stepNum < 5 && (
          <div
            className={`absolute left-[17px] top-9 bottom-0 w-[2px] transition-all duration-300 ${
              isCompleted ? "bg-blue-500" : "bg-white/10"
            }`}
          />
        )}

        <div
          className={`w-9 h-9 rounded-full flex items-center justify-center font-heading font-semibold text-[14px] flex-shrink-0 transition-all duration-300 border-2 ${
            isCompleted
              ? "bg-blue-500/10 border-blue-500 text-blue-500"
              : isPending
              ? "bg-amber-500/10 border-amber-500 text-amber-500 shadow-[0_0_10px_rgba(245,158,11,0.2)]"
              : "bg-white/5 border-white/10 text-gray-500"
          }`}
        >
          {isCompleted ? <CheckCircle2 className="h-5 w-5" /> : stepNum}
        </div>

        <div className="flex flex-col gap-1 pt-1">
          <span className={`font-semibold text-[15px] ${isCompleted ? "text-white" : "text-gray-300"}`}>
            {title}
          </span>
          <span className="text-[13px] text-gray-400 leading-relaxed max-w-xl">{desc}</span>
          <span
            className={`inline-block self-start mt-2 px-2 py-0.5 rounded text-[11px] font-semibold border ${
              isCompleted
                ? "bg-blue-500/10 border-blue-500/20 text-blue-400"
                : isPending
                ? "bg-amber-500/10 border-amber-500/20 text-amber-400"
                : "bg-white/5 border-white/10 text-gray-500"
            }`}
          >
            {badgeText}
          </span>
        </div>
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
      {/* Left side: Journey flow */}
      <div className="lg:col-span-2 rounded-xl border border-white/5 bg-white/[0.02] p-6 flex flex-col gap-6">
        <h3 className="font-heading font-semibold text-lg text-white">
          Dataset Journey & Status
        </h3>
        <div className="flex flex-col mt-2">
          {renderStep(
            1,
            "Sample SFT Prompts",
            "Prompts from StrategyQA, BoolQ, and LogiQA stored in sft/prompts.jsonl.",
            `${promptCount} prompts`,
            promptCount > 0,
            promptCount === 0
          )}
          {renderStep(
            2,
            "Generate Verbose Traces",
            "Model generates raw thinking traces in raw/{model}/traces.jsonl.",
            `${rawCount} traces`,
            rawCount > 0,
            promptCount > 0 && rawCount === 0
          )}
          {renderStep(
            3,
            "Compress CoT Traces",
            "Token-efficient, telegraphic 'Grug' reasoning created in compressed/{model}/traces.jsonl.",
            `${compressedCount} compressed`,
            compressedCount > 0,
            rawCount > 0 && compressedCount === 0
          )}
          {renderStep(
            4,
            "Validate Traces",
            "Filter correct reasoning samples, check details in validated/{model}/traces.jsonl and validation_report.json.",
            `${validatedCount} validated`,
            validatedCount > 0,
            compressedCount > 0 && validatedCount === 0
          )}
          {renderStep(
            5,
            "SFT Dataset splits",
            "Generate formatted splits: train.jsonl and valid.jsonl. Ready for fine-tuning.",
            `${sftCount} SFT rows (${trainCount} train / ${validCount} valid)`,
            sftCount > 0,
            validatedCount > 0 && sftCount === 0
          )}
        </div>
      </div>

      {/* Right side: Validation Report */}
      <div className="rounded-xl border border-white/5 bg-white/[0.02] p-6 flex flex-col gap-6">
        <h3 className="font-heading font-semibold text-lg text-white">
          Validation Report Details
        </h3>

        {!report ? (
          <div className="flex flex-col items-center justify-center gap-3 py-16 text-center">
            <HelpCircle className="h-10 w-10 text-gray-600 animate-pulse" />
            <p className="text-gray-400 text-sm max-w-[240px]">
              No validation report loaded yet. Load a valid data folder or demo data to inspect.
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-4 text-sm">
            <div className="flex justify-between items-center pb-3 border-b border-white/5">
              <span className="text-gray-400">Total Checked:</span>
              <span className="font-mono font-semibold text-blue-400">
                {(report.total_checked ?? validatedCount).toLocaleString()}
              </span>
            </div>
            <div className="flex justify-between items-center pb-3 border-b border-white/5">
              <span className="text-emerald-400">Accepted:</span>
              <span className="font-mono font-semibold text-emerald-400">
                {(report.accepted ?? validatedCount).toLocaleString()}
              </span>
            </div>
            <div className="flex justify-between items-center pb-3 border-b border-white/5">
              <span className="text-red-400">Rejected:</span>
              <span className="font-mono font-semibold text-red-400">
                {(report.rejected ?? 0).toLocaleString()}
              </span>
            </div>
            <div className="flex justify-between items-center pb-4 border-b border-white/5">
              <span className="text-gray-400">Rejection Rate:</span>
              <span className="font-mono font-semibold text-amber-500">
                {report.rejection_rate !== undefined
                  ? `${(report.rejection_rate * 100).toFixed(1)}%`
                  : "0.0%"}
              </span>
            </div>

            <div className="mt-2">
              <span className="text-[12px] font-semibold text-gray-500 uppercase tracking-wider block mb-3">
                Rejection Reasons:
              </span>
              <div className="flex flex-col gap-2">
                {!report.rejection_reasons ||
                Object.keys(report.rejection_reasons).length === 0 ? (
                  <span className="text-gray-500 text-[13px] italic">
                    No rejected items reported.
                  </span>
                ) : (
                  Object.entries(report.rejection_reasons).map(([reason, count]) => (
                    <div key={reason} className="flex justify-between items-center text-[13px]">
                      <span className="text-gray-400">{reason}:</span>
                      <span className="font-semibold text-red-400 font-mono">{count}</span>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
