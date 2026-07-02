import { useState } from "react"
import type { WorkspaceData } from "@/types"
import {
  createEmptyWorkspace,
  parsePromptsJsonl,
  parseRawTracesJsonl,
  parseCompressedTracesJsonl,
  parseValidatedTracesJsonl,
  parseSftFormattedJsonl,
  parseResultsJson,
} from "@/lib/loader"

export function useWorkspace() {
  const [workspaceData, setWorkspaceData] = useState<WorkspaceData>(
    createEmptyWorkspace
  )
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const resetWorkspace = () => {
    setWorkspaceData(createEmptyWorkspace())
    setError(null)
  }

  const loadFromFiles = async (files: File[]) => {
    setIsLoading(true)
    setError(null)
    const newWorkspace = createEmptyWorkspace()

    try {
      let filesFoundCount = 0

      for (const file of files) {
        // webkitRelativePath is available for directory upload. Fallback to name.
        const path = file.webkitRelativePath || file.name
        const content = await file.text()

        if (path.endsWith("sft/prompts.jsonl")) {
          parsePromptsJsonl(content, newWorkspace)
          filesFoundCount++
        } else if (path.includes("raw/") && path.endsWith("traces.jsonl")) {
          parseRawTracesJsonl(content, newWorkspace)
          filesFoundCount++
        } else if (path.includes("compressed/") && path.endsWith("traces.jsonl")) {
          parseCompressedTracesJsonl(content, newWorkspace)
          filesFoundCount++
        } else if (path.includes("validated/") && path.endsWith("traces.jsonl")) {
          parseValidatedTracesJsonl(content, newWorkspace)
          filesFoundCount++
        } else if (path.includes("validation_report.json")) {
          try {
            newWorkspace.validationReport = JSON.parse(content)
            filesFoundCount++
          } catch (err) {
            console.error("Error parsing validation report:", err)
          }
        } else if (path.endsWith("train.jsonl") || path.endsWith("valid.jsonl")) {
          parseSftFormattedJsonl(
            content,
            path.endsWith("train.jsonl") ? "train" : "valid",
            newWorkspace
          )
          filesFoundCount++
        } else if (path.endsWith(".json") && !path.includes("validation_report.json")) {
          parseResultsJson(content, path, newWorkspace)
          filesFoundCount++
        }
      }

      if (filesFoundCount > 0) {
        // Ensure sources is computed if not already populated
        const uniqueSources = new Set<string>()
        Object.values(newWorkspace.prompts).forEach((p) => {
          if (p.source) uniqueSources.add(p.source)
        })
        newWorkspace.sources = Array.from(uniqueSources)
        setWorkspaceData(newWorkspace)
      } else {
        setError("No recognized pipeline files were found in the uploaded directory.")
      }
    } catch (err: any) {
      setError(err?.message || "Failed to parse files.")
    } finally {
      setIsLoading(false)
    }
  }

  const loadDemo = () => {
    const demo = createEmptyWorkspace()
    demo.isDemo = true
    demo.sources = ["boolq", "strategyqa", "logiqa"]

    // StrategyQA
    demo.prompts["strategyqa-0001"] = {
      id: "strategyqa-0001",
      source: "strategyqa",
      prompt: "Did producer of Easy Rider ever star in a movie with Dean Cain's Princeton girlfriend?",
      choices: ["yes", "no"],
      ground_truth: "yes",
    }
    demo.rawTraces["strategyqa-0001"] = {
      id: "strategyqa-0001",
      source: "strategyqa",
      prompt: "Did producer of Easy Rider ever star in a movie with Dean Cain's Princeton girlfriend?",
      raw_thinking: "Easy Rider was produced by Peter Fonda. Dean Cain's Princeton girlfriend was Brooke Shields. Did Peter Fonda ever star in a movie with Brooke Shields? Yes, they both starred in the movie 'Wanda Nevada' (1979). Peter Fonda directed and starred in it, and Brooke Shields played Wanda Nevada. So the answer should be yes.",
      raw_answer: "yes",
      raw_answer_correct: true,
    }
    demo.compressedTraces["strategyqa-0001"] = {
      id: "strategyqa-0001",
      compressed_thinking: "Easy Rider producer Peter Fonda. Dean Cain Princeton girlfriend Brooke Shields. Wanda Nevada (1979) features both. Fonda stars, Shields stars. Entails yes.",
    }
    demo.validatedTraces["strategyqa-0001"] = { id: "strategyqa-0001" }

    // BoolQ 1
    demo.prompts["boolq-0002"] = {
      id: "boolq-0002",
      source: "boolq",
      prompt: "Passage: Me Talk Pretty One Day, published in 2000, is a bestselling collection of essays by American humorist David Sedaris. The book is separated into two parts. The first part consists of essays about Sedaris's life before his move to Normandy, France, including his upbringing in suburban Raleigh, North Carolina, his time working odd jobs in New York City, and a visit to New York from a childhood friend and her bumpkinish girlfriend. The second section, ``Deux'', tells of Sedaris's move to Normandy with his partner Hugh, often drawing humor from his efforts to live in France without speaking the French language and his frustrated attempts to learn it.\nQuestion: is me talk pretty one day an autobiography?",
      choices: ["yes", "no"],
      ground_truth: "yes",
    }
    demo.rawTraces["boolq-0002"] = {
      id: "boolq-0002",
      source: "boolq",
      raw_thinking: "Okay, so I'm trying to figure out whether \"Me Talk Pretty One Day\" by David Sedaris is an autobiography. The passage says it's a bestselling collection of essays, and the answer should be either yes or no. First, I know that an autobiography is something you write about your life, often reflecting personal emotions or experiences. The passage mentions that the book is divided into two parts: one about Sedaris's life before moving to France, including his upbringing in Raleigh and New York City, and a visit. The second part is about moving to Normandy with his partner Hugh, dealing with language barriers and frustration. Since the book is about Sedaris's life experiences, it seems like he's sharing his personal journey and thoughts through these essays. Autobiographies usually involve personal reflections, so each essay probably tells a part of his life.",
      raw_answer: "yes",
      raw_answer_correct: true,
    }
    demo.compressedTraces["boolq-0002"] = {
      id: "boolq-0002",
      compressed_thinking: "Autobiography involves writing about life, personal emotions, experiences. Book divided into two parts: Sedaris's life before France (upbringing in Raleigh, NYC) and moving to Normandy with Hugh, facing language barriers. Book covers Sedaris's life experiences, sharing personal journey and thoughts. Essays read on \"This American Life\" indicate personal reflection.",
    }
    demo.validatedTraces["boolq-0002"] = { id: "boolq-0002" }

    // BoolQ 2 (Incorrect example)
    demo.prompts["boolq-0007"] = {
      id: "boolq-0007",
      source: "boolq",
      prompt: "Passage: Tornadoes vary in intensity regardless of shape, size, and location, though strong tornadoes are typically larger than weak tornadoes. The association with track length and duration also varies, although longer track (and longer lived) tornadoes tend to be stronger.\nQuestion: is the size of a tornado directly related to the intensity of the tornado?",
      choices: ["yes", "no"],
      ground_truth: "no",
    }
    demo.rawTraces["boolq-0007"] = {
      id: "boolq-0007",
      source: "boolq",
      raw_thinking: "Let me check the passage. The passage states: 'Tornadoes vary in intensity regardless of shape, size, and location, though strong tornadoes are typically larger than weak tornadoes.' This means size is directly related since strong ones are typically larger. So size is related to intensity. The answer is yes.",
      raw_answer: "yes",
      raw_answer_correct: false,
    }

    // LogiQA
    demo.prompts["logiqa-0001"] = {
      id: "logiqa-0001",
      source: "logiqa",
      prompt: "Passage: Imitation: refers to the fact that an individual is stimulated by the actions of others, consciously or unconsciously, to make his behavior similar to others, without external control.\nQuestion: Which of the following is not imitation?\nA) she thinks that others wear this kind of clothes very beautiful, so she went to buy it.\nB) the child saw the pedestrian kicking the cans on the ground and kicked the plastic bottle in his hand.\nC) many experts predicted the same lottery number by accident, so Xiao Liu immediately bought 50 bets on the number.\nD) he originally wanted to cut in, but when he saw that everyone was lining up to buy tickets in order, he consciously walked to the end of the line.",
      choices: ["A", "B", "C", "D"],
      ground_truth: "C",
    }
    demo.rawTraces["logiqa-0001"] = {
      id: "logiqa-0001",
      source: "logiqa",
      raw_thinking: "Imitation requires making behavior similar to others because of others' actions. Option A is imitation (buying clothes because of others). Option B is imitation (kicking bottle because saw kicking cans). Option C: experts predicted same lottery number by accident. Xiao Liu buying bets is copying, but the experts did not copy. Wait. The experts predicted same number by accident, that's not imitation. Xiao Liu bought bets because experts predicted. Option D: lines up because others are lining up, this is copying/imitation. Option C is not imitation of experts imitating each other, they did it by accident. Thus, C is the right answer.",
      raw_answer: "C",
      raw_answer_correct: true,
    }
    demo.compressedTraces["logiqa-0001"] = {
      id: "logiqa-0001",
      compressed_thinking: "Imitation is copying others' actions without control. A: buying clothes copies others. B: child kicking copies pedestrian. D: lining up copies crowd. C: experts predict same number by accident, Xiao Liu buys bets based on prediction. Experts didn't copy. C is not imitation.",
    }
    demo.validatedTraces["logiqa-0001"] = { id: "logiqa-0001" }

    // Validation Report
    demo.validationReport = {
      total_checked: 4,
      accepted: 3,
      rejected: 1,
      rejection_rate: 0.25,
      rejection_reasons: {
        "Raw answer incorrect": 1,
      },
    }

    // Formatted SFT splits
    demo.sftFormatted = [
      {
        text: "<｜beginofsentence｜><｜User｜>Did producer of Easy Rider ever star in a movie with Dean Cain's Princeton girlfriend?<｜Assistant｜><think>\nEasy Rider producer Peter Fonda. Dean Cain Princeton girlfriend Brooke Shields. Wanda Nevada (1979) features both. Fonda stars, Shields stars. Entails yes.\n</think>\n\nyes<｜endofsentence｜>",
        type: "train",
      },
      {
        text: "<｜beginofsentence｜><｜User｜>Passage: Me Talk Pretty One Day... Question: is me talk pretty one day an autobiography?<｜Assistant｜><think>\nAutobiography involves writing about life, personal emotions, experiences. Book divided into two parts: Sedaris's life before France (upbringing in Raleigh, NYC) and moving to Normandy with Hugh, facing language barriers. Book covers Sedaris's life experiences, sharing personal journey and thoughts. Essays read on \"This American Life\" indicate personal reflection.\n</think>\n\nyes<｜endofsentence｜>",
        type: "valid",
      },
    ]

    // Mock Evaluation Results - baseline normal
    demo.results["deepseek-r1-1.5b-baseline-gsm8k_normal"] = {
      summary: {
        accuracy: 0.69,
        format_compliance_rate: 0.98,
        mean_thinking_tokens: 200.39,
        mean_answer_tokens: 263.94,
        mean_total_tokens: 464.33,
        mean_latency: 1.49,
        mean_tokens_per_second: 309.99,
        sample_count: 100,
      },
      results: [
        {
          id: 1,
          question: "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
          ground_truth: "18",
          thinking_content: "First, determine how many eggs Janet starts with each day: 16 eggs.\nShe eats 3 eggs for breakfast every morning, leaving 16 - 3 = 13 eggs.\nShe bakes muffins using 4 eggs, leaving 13 - 4 = 9 eggs.\nShe sells the remaining 9 eggs for $2 each. 9 * 2 = 18.\nTherefore she makes $18.",
          answer_content: "Janet makes $18 every day at the farmers' market.",
          predicted_answer: "18",
          correct: true,
          thinking_tokens: 161,
          answer_tokens: 229,
          total_tokens: 390,
          latency_seconds: 1.49,
          tokens_per_second: 260.3,
          format_compliance: true,
        },
        {
          id: 2,
          question: "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
          ground_truth: "3",
          thinking_content: "Robe takes 2 bolts of blue fiber. White fiber is half of blue, which is 2 / 2 = 1 bolt.\nTotal bolts = 2 blue + 1 white = 3 bolts.",
          answer_content: "It takes 3 bolts in total.",
          predicted_answer: "3",
          correct: true,
          thinking_tokens: 77,
          answer_tokens: 192,
          total_tokens: 269,
          latency_seconds: 1.49,
          tokens_per_second: 179.5,
          format_compliance: true,
        },
      ],
      metadata: {
        model: "deepseek-r1-1.5b",
        runType: "baseline",
        benchmark: "gsm8k",
        promptStyle: "normal",
      },
    }

    // Mock Evaluation Results - baseline grug
    demo.results["deepseek-r1-1.5b-baseline-gsm8k_grug_prompt"] = {
      summary: {
        accuracy: 0.66,
        format_compliance_rate: 0.92,
        mean_thinking_tokens: 549.3,
        mean_answer_tokens: 64.7,
        mean_total_tokens: 614.0,
        mean_latency: 2.12,
        mean_tokens_per_second: 289.0,
        sample_count: 100,
      },
      results: [
        {
          id: 1,
          question: "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
          ground_truth: "18",
          thinking_content: "Janet lays 16 eggs. Eats 3. 16 - 3 = 13. Muffins 4. 13 - 4 = 9. Sells 9 for 2. 9 * 2 = 18. Answer 18.",
          answer_content: "18",
          predicted_answer: "18",
          correct: true,
          thinking_tokens: 30,
          answer_tokens: 10,
          total_tokens: 40,
          latency_seconds: 0.2,
          tokens_per_second: 200,
          format_compliance: true,
        },
        {
          id: 2,
          question: "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
          ground_truth: "3",
          thinking_content: "Robe blue 2. White half blue, so 1. Total 2 + 1 = 3.",
          answer_content: "3",
          predicted_answer: "3",
          correct: true,
          thinking_tokens: 20,
          answer_tokens: 5,
          total_tokens: 25,
          latency_seconds: 0.15,
          tokens_per_second: 166.7,
          format_compliance: true,
        },
      ],
      metadata: {
        model: "deepseek-r1-1.5b",
        runType: "baseline",
        benchmark: "gsm8k",
        promptStyle: "grug_prompt",
      },
    }

    // Mock Evaluation Results - finetuned normal
    demo.results["deepseek-r1-1.5b-finetuned-gsm8k_normal"] = {
      summary: {
        accuracy: 0.71,
        format_compliance_rate: 1.0,
        mean_thinking_tokens: 82.5,
        mean_answer_tokens: 45.2,
        mean_total_tokens: 127.7,
        mean_latency: 0.45,
        mean_tokens_per_second: 283.7,
        sample_count: 100,
      },
      results: [
        {
          id: 1,
          question: "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
          ground_truth: "18",
          thinking_content: "Ducks lay 16. Breakfast 3. Left 13. Muffins 4. Left 9. Sells 9 * 2 = 18.",
          answer_content: "18",
          predicted_answer: "18",
          correct: true,
          thinking_tokens: 18,
          answer_tokens: 5,
          total_tokens: 23,
          latency_seconds: 0.1,
          tokens_per_second: 230,
          format_compliance: true,
        },
        {
          id: 2,
          question: "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
          ground_truth: "3",
          thinking_content: "Blue 2. White 1. Sum is 3.",
          answer_content: "3",
          predicted_answer: "3",
          correct: true,
          thinking_tokens: 10,
          answer_tokens: 4,
          total_tokens: 14,
          latency_seconds: 0.08,
          tokens_per_second: 175,
          format_compliance: true,
        },
      ],
      metadata: {
        model: "deepseek-r1-1.5b",
        runType: "finetuned",
        benchmark: "gsm8k",
        promptStyle: "normal",
      },
    }

    setWorkspaceData(demo)
  }

  return {
    workspaceData,
    isLoading,
    error,
    loadFromFiles,
    loadDemo,
    resetWorkspace,
  }
}
