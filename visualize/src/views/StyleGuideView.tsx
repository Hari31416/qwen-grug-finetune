import { CheckCircle2 } from "lucide-react"

export function StyleGuideView() {
  const principles = [
    {
      title: "Sentence Fragments",
      desc: "Use short, telegraphic phrasing. Instead of complete subject-verb sentences, use quick inference markers.",
    },
    {
      title: "Drop Articles",
      desc: "Omit filler words like 'the', 'a', 'an', and transition fillers (e.g. 'so now let's see', 'let me calculate that') where possible.",
    },
    {
      title: "Preserve Math & Logic",
      desc: "Keep all numbers, formulas, variables, options, and intermediate calculations 100% accurate.",
    },
    {
      title: "Continuous Paragraph",
      desc: "Represent reasoning in one continuous paragraph of fragments, separated by periods, rather than vertical bullet lists.",
    },
    {
      title: "No Conclusion Restatement",
      desc: "Drop statements repeating the final answer (e.g. 'So the answer is yes') inside the think block. The final answer exists in the label.",
    },
  ]

  const examples = [
    {
      title: "Example 1: Math and Addition",
      verbose:
        "To find the total amount of money they have, first I need to calculate how much Sarah has. Sarah has $10. Then the problem says Mark has 3 times as much money as Sarah. So Mark has 3 * $10 = $30. Wait... let me check that. Yes, 3 times 10 is 30. Okay, so Mark has $30. Next, I need to add Sarah's money and Mark's money together. Sarah has $10 and Mark has $30. Wait, but this means the sum is $10 + $30. Let me add them. $10 + $30 = $40. So the total sum is $40. Yes, total is 40. Finally, I will state the final answer, which is 40.",
      verboseTokens: 105,
      grug: "Sarah 10. Mark 3 times Sarah, so 30. Total 10 + 30 = 40.",
      grugTokens: 14,
      reduction: "86.6%",
    },
    {
      title: "Example 2: Logic and Transitivity",
      verbose:
        "Let's look at the clues. Clue 1 says Alice is taller than Bob. Clue 2 says Bob is taller than Charlie. If Alice is taller than Bob and Bob is taller than Charlie, then by transitivity, Alice must be taller than Charlie. Wait, let me make sure. Yes, Alice > Bob > Charlie. So Alice > Charlie. Okay, but this means Alice is the tallest since she is taller than both. Yes, Alice is tallest. The question asks who is the tallest. Since Alice is taller than Bob and Charlie, Alice is the tallest. Therefore, the answer is Alice.",
      verboseTokens: 102,
      grug: "Clues Alice > Bob, Bob > Charlie. So Alice > Charlie. Alice tallest.",
      grugTokens: 14,
      reduction: "86.2%",
    },
    {
      title: "Example 3: Natural Language Inference (NLI)",
      verbose:
        "The premise states that a man is sitting on a park bench reading a book. The hypothesis is that a man is outdoors. We need to determine if the premise entails the hypothesis. A park bench is typically located outdoors in a public park. If a man is sitting on a park bench, he is almost certainly in a park, which is outdoors. Therefore, the premise entails the hypothesis. So the relationship is entailment.",
      verboseTokens: 81,
      grug: "Premise man on park bench. Hypothesis man outdoors. Park bench outdoors so man outdoors. Premise entails hypothesis.",
      grugTokens: 20,
      reduction: "75.3%",
    },
  ]

  return (
    <div className="flex flex-col gap-8 w-full pb-10">
      {/* Grug Principles */}
      <div className="rounded-xl border border-white/5 bg-white/[0.02] p-6 flex flex-col gap-5">
        <h3 className="font-heading font-semibold text-lg text-white">
          Telegraphic \"Grug\" Reasoning Principles
        </h3>
        <p className="text-gray-400 text-sm leading-relaxed max-w-3xl">
          The core objective of the Grug style guide is to drastically reduce token consumption
          in the thinking phase of fine-tuned reasoning models. By eliminating verbose syntactic
          structures while preserving logical flow and intermediate states, the model learns to
          think compactly.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mt-2">
          {principles.map((p, idx) => (
            <div
              key={idx}
              className="border border-white/5 bg-white/[0.01] rounded-lg p-5 flex flex-col gap-3 transition-all hover:border-blue-500/20"
            >
              <div className="flex items-center gap-2 font-semibold text-[14px] text-white">
                <CheckCircle2 className="h-4.5 w-4.5 text-blue-400 flex-shrink-0" />
                <span>{p.title}</span>
              </div>
              <p className="text-[13px] text-gray-400 leading-relaxed">{p.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Examples Comparison */}
      <div className="rounded-xl border border-white/5 bg-white/[0.02] p-6 flex flex-col gap-6">
        <h3 className="font-heading font-semibold text-lg text-white">
          Compression Benchmarks & Examples
        </h3>

        <div className="flex flex-col gap-8">
          {examples.map((ex, idx) => (
            <div key={idx} className="flex flex-col gap-3">
              <h4 className="text-[14px] font-semibold text-blue-400">{ex.title}</h4>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {/* Verbose */}
                <div className="rounded-lg border border-red-500/10 bg-red-500/[0.02] p-4 flex flex-col gap-2.5">
                  <div className="flex justify-between items-center text-[10px] font-bold text-red-400 uppercase tracking-wider">
                    <span>Verbose Reasoning</span>
                    <span className="font-mono">{ex.verboseTokens} tokens</span>
                  </div>
                  <p className="text-[13px] text-gray-300 leading-relaxed font-sans">
                    {ex.verbose}
                  </p>
                </div>

                {/* Grug */}
                <div className="rounded-lg border border-emerald-500/10 bg-emerald-500/[0.02] p-4 flex flex-col gap-2.5">
                  <div className="flex justify-between items-center text-[10px] font-bold text-emerald-400 uppercase tracking-wider">
                    <span>Compressed Reasoning (Grug)</span>
                    <span className="font-mono">{ex.grugTokens} tokens</span>
                  </div>
                  <p className="text-[13px] text-gray-200 leading-relaxed font-sans font-medium">
                    {ex.grug}
                  </p>
                </div>
              </div>

              <div className="text-[12px] font-bold text-emerald-400 text-right">
                Token Reduction: {ex.reduction}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
