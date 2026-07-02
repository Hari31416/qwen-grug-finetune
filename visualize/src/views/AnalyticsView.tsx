import { useMemo } from "react"
import type { WorkspaceData } from "@/types"
import { HelpCircle } from "lucide-react"
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from "chart.js"
import { Doughnut, Bar } from "react-chartjs-2"

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
)

interface AnalyticsViewProps {
  data: WorkspaceData
}

export function AnalyticsView({ data }: AnalyticsViewProps) {
  // Chart 1: Source Dataset Distribution
  const sourceChartData = useMemo(() => {
    const counts: Record<string, number> = {}
    Object.values(data.prompts).forEach((p) => {
      const src = p.source || "unknown"
      counts[src] = (counts[src] || 0) + 1
    })

    const labels = Object.keys(counts).map((l) => l.toUpperCase())
    const values = Object.values(counts)

    return {
      labels,
      datasets: [
        {
          label: "Prompts",
          data: values,
          backgroundColor: [
            "rgba(59, 130, 246, 0.6)",   // blue
            "rgba(139, 92, 246, 0.6)",  // purple
            "rgba(16, 185, 129, 0.6)",  // emerald
            "rgba(245, 158, 11, 0.6)",   // amber
          ],
          borderColor: [
            "#3b82f6",
            "#8b5cf6",
            "#10b981",
            "#f59e0b",
          ],
          borderWidth: 1.5,
        },
      ],
    }
  }, [data.prompts])

  // Chart 2: Accuracy by source
  const accuracyChartData = useMemo(() => {
    const totals: Record<string, number> = {}
    const corrects: Record<string, number> = {}

    Object.values(data.rawTraces).forEach((t) => {
      const src = t.source || "unknown"
      if (t.raw_answer_correct !== undefined) {
        totals[src] = (totals[src] || 0) + 1
        if (t.raw_answer_correct) {
          corrects[src] = (corrects[src] || 0) + 1
        }
      }
    })

    const labels = Object.keys(totals)
    const accuracyValues = labels.map((l) => {
      const tot = totals[l] || 0
      const corr = corrects[l] || 0
      return tot > 0 ? parseFloat(((corr / tot) * 100).toFixed(1)) : 0
    })

    return {
      labels: labels.map((l) => l.toUpperCase()),
      datasets: [
        {
          label: "Accuracy %",
          data: accuracyValues,
          backgroundColor: "rgba(245, 158, 11, 0.5)", // Amber
          borderColor: "#f59e0b",
          borderWidth: 1.5,
          borderRadius: 4,
        },
      ],
    }
  }, [data.rawTraces])

  // Chart 3: Token length reduction by dataset
  const compressionChartData = useMemo(() => {
    const rawSums: Record<string, number> = {}
    const compSums: Record<string, number> = {}
    const counts: Record<string, number> = {}

    // Loop through prompts to get their source mapping
    Object.keys(data.compressedTraces).forEach((id) => {
      const raw = data.rawTraces[id]
      const comp = data.compressedTraces[id]
      const prompt = data.prompts[id]

      if (raw && comp && raw.raw_thinking && comp.compressed_thinking) {
        const src = prompt?.source || raw.source || "unknown"
        rawSums[src] = (rawSums[src] || 0) + raw.raw_thinking.length
        compSums[src] = (compSums[src] || 0) + comp.compressed_thinking.length
        counts[src] = (counts[src] || 0) + 1
      }
    })

    const labels = Object.keys(counts)
    const avgRaw = labels.map((l) =>
      counts[l] > 0 ? Math.round(rawSums[l] / counts[l]) : 0
    )
    const avgComp = labels.map((l) =>
      counts[l] > 0 ? Math.round(compSums[l] / counts[l]) : 0
    )

    return {
      labels: labels.map((l) => l.toUpperCase()),
      datasets: [
        {
          label: "Avg Raw Chars",
          data: avgRaw,
          backgroundColor: "rgba(239, 68, 68, 0.5)", // Red-ish
          borderColor: "#ef4444",
          borderWidth: 1.5,
          borderRadius: 4,
        },
        {
          label: "Avg Compressed Chars",
          data: avgComp,
          backgroundColor: "rgba(16, 185, 129, 0.5)", // Green-ish
          borderColor: "#10b981",
          borderWidth: 1.5,
          borderRadius: 4,
        },
      ],
    }
  }, [data.compressedTraces, data.rawTraces, data.prompts])

  // Dark-mode grid and label config
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        grid: { color: "rgba(255, 255, 255, 0.05)" },
        ticks: { color: "#9ca3af", font: { family: "Inter Variable" } },
      },
      x: {
        grid: { display: false },
        ticks: { color: "#9ca3af", font: { family: "Inter Variable" } },
      },
    },
    plugins: {
      legend: {
        labels: {
          color: "#9ca3af",
          font: { family: "Inter Variable", size: 11 },
        },
      },
    },
  }

  const doughnutOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "right" as const,
        labels: {
          color: "#9ca3af",
          font: { family: "Inter Variable", size: 11 },
        },
      },
    },
  }

  const noAnalytics =
    Object.keys(data.prompts).length === 0 &&
    Object.keys(data.rawTraces).length === 0

  return (
    <div className="flex flex-col gap-8 w-full">
      {noAnalytics ? (
        <div className="rounded-xl border border-white/5 bg-white/[0.02] p-16 text-center flex flex-col items-center justify-center gap-3">
          <HelpCircle className="h-10 w-10 text-gray-600 animate-pulse" />
          <p className="text-gray-400 text-sm">
            Please load pipeline workspace files or demo data to display charts.
          </p>
        </div>
      ) : (
        <>
          {/* First row of charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="rounded-xl border border-white/5 bg-white/[0.02] p-6 flex flex-col gap-4 min-h-[350px]">
              <h3 className="font-heading font-semibold text-lg text-white">
                Source Dataset Distribution
              </h3>
              <div className="flex-1 relative min-h-[220px]">
                <Doughnut data={sourceChartData} options={doughnutOptions} />
              </div>
            </div>

            <div className="rounded-xl border border-white/5 bg-white/[0.02] p-6 flex flex-col gap-4 min-h-[350px]">
              <h3 className="font-heading font-semibold text-lg text-white">
                Raw Model Accuracy by Dataset
              </h3>
              <div className="flex-1 relative min-h-[220px]">
                <Bar data={accuracyChartData} options={chartOptions} />
              </div>
            </div>
          </div>

          {/* Character length reduction row */}
          <div className="rounded-xl border border-white/5 bg-white/[0.02] p-6 flex flex-col gap-4 min-h-[350px]">
            <h3 className="font-heading font-semibold text-lg text-white">
              Token Length Reduction (Raw vs Compressed Thinking)
            </h3>
            <div className="flex-1 relative min-h-[220px]">
              <Bar data={compressionChartData} options={chartOptions} />
            </div>
          </div>
        </>
      )}
    </div>
  )
}
