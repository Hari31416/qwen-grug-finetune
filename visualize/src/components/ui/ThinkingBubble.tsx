import { Lightbulb } from "lucide-react"

interface ThinkingBubbleProps {
  content: string
  title?: string
  className?: string
}

export function ThinkingBubble({
  content,
  title = "Thinking Process",
  className = "",
}: ThinkingBubbleProps) {
  return (
    <div
      className={`rounded-lg border border-amber-500/15 bg-amber-500/[0.03] p-4 text-[14px] leading-relaxed text-gray-200 ${className}`}
    >
      <div className="flex items-center gap-1.5 font-semibold text-amber-500 text-[12px] uppercase tracking-wider mb-2 select-none">
        <Lightbulb className="h-4 w-4 fill-amber-500/20" />
        <span>{title}</span>
      </div>
      <div className="whitespace-pre-wrap font-sans text-gray-300">{content}</div>
    </div>
  )
}
