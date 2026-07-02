import { useState } from "react"
import { Check, Copy } from "lucide-react"

interface CodeBlockProps {
  content: string
  className?: string
}

export function CodeBlock({ content, className = "" }: CodeBlockProps) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error("Failed to copy text:", err)
    }
  }

  return (
    <div className="relative group rounded-md border border-white/5 bg-[#070913] p-4 font-mono text-[13px] leading-relaxed text-gray-300 break-all whitespace-pre-wrap overflow-x-auto">
      <div className={`overflow-auto ${className}`}>{content}</div>
      <button
        onClick={handleCopy}
        className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity p-1.5 rounded-md hover:bg-white/10 text-gray-400 hover:text-white"
        title="Copy to clipboard"
      >
        {copied ? <Check className="h-4 w-4 text-emerald-400" /> : <Copy className="h-4 w-4" />}
      </button>
    </div>
  )
}
