import { useMemo } from 'react'
import katex from 'katex'
import { marked } from 'marked'

interface MarkdownMathProps {
  content: string
  className?: string
}

function processMathAndMarkdown(raw: string): string {
  if (!raw) return ''

  // 1. Format GSM8K calculation annotations <<...>> to clean badge or clean evaluation
  let processed = raw.replace(/<<([^>]+)>>/g, (_, calc) => {
    // e.g. "14*25*4=1400" -> show clean formula calculation if helpful
    const parts = calc.split('=')
    const result = parts.length > 1 ? parts[1].trim() : calc.trim()
    return ` <span class="text-[11px] font-mono px-1 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20" title="${calc}">[${result}]</span> `
  })

  // 2. Format GSM8K "#### <answer>" target delimiter
  processed = processed.replace(/####\s*([^\n]+)/g, (_, ans) => {
    return `\n\n<div class="inline-flex items-center gap-2 px-3 py-1.5 my-2 rounded-lg bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 font-bold font-mono text-xs shadow-sm"><span class="uppercase tracking-wider text-[10px] text-emerald-500/80 font-semibold">Target Answer:</span><span>${ans.trim()}</span></div>\n\n`
  })

  // 3. Process LaTeX Display Math: \[ ... \] or $$ ... $$
  processed = processed.replace(/\\\[([\s\S]*?)\\\]/g, (_, tex) => {
    try {
      return katex.renderToString(tex.trim(), { displayMode: true, throwOnError: false })
    } catch {
      return `\\[${tex}\\]`
    }
  })

  processed = processed.replace(/\$\$([\s\S]*?)\$\$/g, (_, tex) => {
    try {
      return katex.renderToString(tex.trim(), { displayMode: true, throwOnError: false })
    } catch {
      return `$$${tex}$$`
    }
  })

  // 4. Process LaTeX Inline Math: \( ... \) or $ ... $
  processed = processed.replace(/\\\(([\s\S]*?)\\\)/g, (_, tex) => {
    try {
      return katex.renderToString(tex.trim(), { displayMode: false, throwOnError: false })
    } catch {
      return `\\(${tex}\\)`
    }
  })

  // 5. Parse with marked (synchronous in marked v18)
  const html = marked.parse(processed, {
    breaks: true,
    gfm: true,
  })

  return typeof html === 'string' ? html : ''
}

export function MarkdownMath({ content, className = '' }: MarkdownMathProps) {
  const renderedHtml = useMemo(() => {
    return processMathAndMarkdown(content)
  }, [content])

  return (
    <div
      dangerouslySetInnerHTML={{ __html: renderedHtml }}
      className={`markdown-math-content text-xs leading-relaxed text-gray-200 break-words ${className}`}
    />
  )
}
