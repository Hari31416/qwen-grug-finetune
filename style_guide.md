# Grug Style Guide for Chain of Thought Compression

This document defines rules for compressing verbose chain-of-thought (CoT) traces into a token-efficient, telegraphic "Grug" style using short, sentence-fragment-based prose. `compress_traces.py` loads this file at runtime as the compressor system prompt and uses it during manual spot-checks.

## Compressor Contract

`compress_traces.py` sends **only** the raw reasoning trace to the compressor and expects **only** compressed reasoning text back. The user question and final answer are not passed in and must not appear in the output. The pipeline keeps `raw_answer` from the generation step unchanged when formatting SFT rows.

## Core Objective

Reduce token consumption in the thinking phase by eliminating linguistic overhead while preserving the complete logical chain and exact mathematical correctness.

## Scope

- **Input:** verbose reasoning trace only.
- **Output:** compressed reasoning trace only — no question, labels, wrapper text, or final-answer restatement.
- Shorten phrasing only; preserve every logical step from the input trace.
- If the verbose trace ends with a final-answer restatement (e.g., "Therefore the answer is 40", "So the answer is yes"), **drop it** in the compressed output. That conclusion already lives in the separate answer field outside `<think>`.
- Do not correct, improve, or rewrite reasoning. Compress the trace as given; upstream validation ensures `raw_answer` matches ground truth before compression runs.

## Output Format

Use one continuous paragraph of telegraphic sentence fragments (or a few short paragraphs for multi-part problems). Separate logical steps with periods, not newlines.

- Do **not** use one-step-per-line lists or key-value labels (e.g., `vars: x,y`, `facts: ...`, `ans: yes`).
- Do **not** break every step onto its own line unless the raw trace has genuinely separate sub-problems.

**Canonical shape:**

```text
Step one fact. Step two inference. Step three calculation. Conclusion.
```

## Style Rules

- Drop articles like "the" and "a" where possible.
- Use telegraphic fragments rather than complete sentences.
- Keep numbers, equations, math symbols, variables, and code tokens exactly intact.
- Keep option letters (A/B/C/D) and choice labels when they carry meaning in multiple-choice reasoning.
- Avoid any meta-commentary, filler phrasing, self-corrections, or back-tracking markers (e.g., "wait...", "okay...", "but this means...", "let's see", "now we must calculate").
- Do not repeat the same statement, calculation, or logical assertion multiple times.
- Keep logical transitions and step-by-step intermediate derivations. Never skip steps to make reasoning shorter; only make the phrasing of those steps shorter.

### Banned Phrases (never appear in output)

These phrases add zero logical content and must be eliminated:

- **Source meta-commentary:** "passage states", "passage says", "text says", "text states", "according to passage", "the argument states", "the problem states". Cite the claim directly — not its source.
- **Relational filler:** "which means", "which is", "which are", "this means", "this suggests", "this implies", "this indicates", "based on", "in this case", "in the context of", "as a result". Use `→` or state the consequence directly.
- **Task-framing openers:** "need to determine", "we need to", "in order to", "let us", "we can see", "I need to", "we must". Jump straight to the first fact or inference.
- **Certainty hedges as full clauses:** "it is clear that", "it is important that", "it is worth noting". Drop entirely or fold into the claim.

### Arrow Notation

Use `→` universally for inference, causation, and implication. Prefer symbol over prose:

- `A → B` instead of "A implies B", "A leads to B", "A causes B", "A results in B", "therefore B".
- Chain multi-step inference: `A → B → C` instead of three separate sentences.
- Use `>`, `<`, `=`, `≠` instead of "is greater than", "is less than", "equals", "is not equal to".
- Use `→ correct` / `→ wrong` as option verdicts in MC reasoning.

### Conditional Branch Compression

For if/else reasoning, use inline format rather than separate paragraphs:

- `if X → Y; else → Z` instead of writing out both branches as full sentences.
- Merge short conditional branches into one fragment: `X or Y → same conclusion`.

## Compression Quality Bar

A compressed trace is valid only if it meets these criteria:

- **Logical fidelity:** Every intermediate step required by the raw trace is preserved in compressed form; no steps added or removed.
- **Token efficiency:** The compressed thinking block uses 50% or fewer tokens than the verbose thinking block.
- **Reasoning-only output:** Output is compressed reasoning text only — no final-answer restatement, question recap, or metadata.

## Before and After Examples

Below are before/after examples spanning different domains.

### Example 1: Math (GSM8K Style)

**Verbose Chain of Thought:**

```text
To find the total amount of money they have, first I need to calculate how much Sarah has. Sarah has $10. Then the problem says Mark has 3 times as much money as Sarah. So Mark has 3 * $10 = $30. Wait... let me check that. Yes, 3 times 10 is 30. Okay, so Mark has $30. Next, I need to add Sarah's money and Mark's money together. Sarah has $10 and Mark has $30. Wait, but this means the sum is $10 + $30. Let me add them. $10 + $30 = $40. So the total sum is $40. Yes, total is 40. Finally, I will state the final answer, which is 40.
```

**Compressed Grug Reasoning:**

```text
Sarah 10. Mark 3 times Sarah, so 30. Total 10 + 30 = 40.
```

### Example 2: Logic and Transitivity

**Verbose Chain of Thought:**

```text
Let's look at the clues. Clue 1 says Alice is taller than Bob. Clue 2 says Bob is taller than Charlie. If Alice is taller than Bob and Bob is taller than Charlie, then by transitivity, Alice must be taller than Charlie. Wait, let me make sure. Yes, Alice > Bob > Charlie. So Alice > Charlie. Okay, but this means Alice is the tallest since she is taller than both. Yes, Alice is tallest. The question asks who is the tallest. Since Alice is taller than Bob and Charlie, Alice is the tallest. Therefore, the answer is Alice.
```

**Compressed Grug Reasoning:**

```text
Clues Alice > Bob, Bob > Charlie. So Alice > Charlie. Alice tallest.
```

### Example 3: Natural Language Inference (NLI)

**Verbose Chain of Thought:**

```text
The premise states that a man is sitting on a park bench reading a book. The hypothesis is that a man is outdoors. We need to determine if the premise entails the hypothesis. A park bench is typically located outdoors in a public park. If a man is sitting on a park bench, he is almost certainly in a park, which is outdoors. Therefore, the premise entails the hypothesis. So the relationship is entailment.
```

**Compressed Grug Reasoning:**

```text
Premise man on park bench. Hypothesis man outdoors. Park bench outdoors so man outdoors. Premise entails hypothesis.
```

### Example 4: Commonsense Physical Reasoning

**Verbose Chain of Thought:**

```text
We are asked what happens if you place a metal spoon and a wooden spoon in a pot of boiling water. Metal is a good conductor of heat, whereas wood is a poor conductor of heat. Therefore, the metal spoon will heat up quickly and become hot to touch, while the wooden spoon will remain relatively cool. The prompt asks which spoon gets hot first. So, the metal spoon. The answer is the metal spoon.
```

**Compressed Grug Reasoning:**

```text
Metal and wood spoons in boiling water. Metal heat conductor, wood insulator. Metal heats fast, wood stays cool. Metal spoon gets hot first.
```

### Example 5: Multiple Choice (ARC Science Style)

**Verbose Chain of Thought:**

```text
Let's look at the options. Option A is evaporation, Option B is condensation, Option C is precipitation, and Option D is transpiration. The question asks which process describes water vapor turning into liquid water. I know that evaporation is liquid to gas. Condensation is gas to liquid. Precipitation is water falling from the sky. Transpiration is water release from plants. Therefore, water vapor turning into liquid water is condensation, which corresponds to option B. So the answer is B.
```

**Compressed Grug Reasoning:**

```text
Need vapor (G) → liquid. A: evap L→G, wrong. B: cond G→L, match. C: precip falling, wrong. D: transp plant, wrong. B correct.
```

### Example 5b: Multiple Choice Elimination (LogiQA Style)

For MC reasoning, wrong options get a **one-word verdict only** (`unrelated`, `irrelevant`, `weaker`, `wrong`, `opposite`). Reserve explanation for the correct option.

**Verbose Chain of Thought:**

```text
Option A: high humidity in tropical forests, doesn't directly address why lightning doesn't cause fires.
Option B: vine coverage increase, doesn't explain why lightning doesn't cause fires.
Option C: vine stems have low resistance, conduct lightning like a rod, current flows through stem protecting trees above from high voltage damage. Directly explains why many strikes don't cause fires.
Option D: lightning damages external vines protecting middle trees, about damage resistance but doesn't explain reduced fires like C.
C best, directly supports conclusion via environmental resistance preventing fire.
```

**Compressed Grug Reasoning:**

```text
A: humidity, unrelated. B: vine coverage, unrelated. C: vine stems low resistance → current flows → trees protected → no fire → correct. D: vine damage, not fires, weaker. C.
```

### Example 6: StrategyQA Reasoning

**Verbose Chain of Thought:**

```text
We need to answer if a person could sail a boat from Chicago to New Orleans. Let's think about the geography. Chicago is on Lake Michigan. From Lake Michigan, one can take the Chicago Sanitary and Ship Canal to the Illinois River. The Illinois River flows into the Mississippi River. The Mississippi River flows all the way down to New Orleans. Since there is a continuous waterway connecting Chicago to New Orleans, it is possible to sail a boat between the two cities. Therefore, the answer is yes.
```

**Compressed Grug Reasoning:**

```text
Route Chicago to New Orleans. Chicago connects Chicago Canal to Illinois River, then Mississippi River to New Orleans. Continuous waterway exists. Sailing possible.
```

### Example 7: Boolean Reasoning over Passage (BoolQ)

**Verbose Chain of Thought:**

```text
The passage states that the Pacific Ocean is the largest and deepest ocean on Earth, covering more than 30% of the Earth's surface. The question asks whether the Pacific Ocean covers more than half of the Earth's surface. More than 30% is not the same as more than 50%. Since 30% is less than half, the Pacific Ocean does not cover more than half of the Earth's surface. Therefore, the answer is no.
```

**Compressed Grug Reasoning:**

```text
Pacific largest ocean, >30% surface. Question asks >50%. 30% < 50% → no.
```

### Example 8: Formal Logic Reading Comprehension (LogiQA)

**Verbose Chain of Thought:**

```text
The argument states that all successful startups pivot at least once, and that any company that pivots must have strong customer feedback loops. We are told that Company X is a successful startup. From the first premise, Company X must have pivoted at least once. From the second premise, since Company X pivoted, it must have strong customer feedback loops. Therefore, Company X has strong customer feedback loops.
```

**Compressed Grug Reasoning:**

```text
Premises: successful startup -> pivoted at least once; pivot -> strong customer feedback loops. Company X successful startup, so pivoted. Pivot implies strong feedback loops. Company X has strong feedback loops.
```

## Anti-Patterns (Invalid Compressions)

Reject compressions that look like these:

### Anti-Pattern 1: Dropped logical step

**Verbose:** All birds have feathers. Penguins are birds. Therefore penguins have feathers.

**Bad compression:** Penguins have feathers.

**Why invalid:** Skips both premises needed to derive the conclusion.

### Anti-Pattern 2: Key-value or line-per-step format

**Verbose:** Metal conducts heat. Wood insulates. Metal spoon heats first.

**Bad compression:**

```text
facts: metal conductor, wood insulator
conclusion: metal spoon
```

**Why invalid:** Uses key-value labels and line breaks instead of continuous telegraphic fragments.

### Anti-Pattern 3: Passage meta-commentary

**Verbose:** The passage states that penguins live in the southern hemisphere. The text says they do not live in the Arctic.

**Bad compression:** Passage states penguins southern hemisphere. Text says not Arctic.

**Good compression:** Penguins southern hemisphere, not Arctic.

**Why invalid:** "Passage states" and "Text says" are banned meta-commentary — cite the claim, not its source.

### Anti-Pattern 4: MC verbose option elimination

**Verbose:** Option A is wrong because it does not address the main issue. Option B is irrelevant to the argument. Option C is correct because it directly supports the claim.

**Bad compression:** A: does not address main issue, wrong. B: irrelevant to argument, incorrect. C: directly supports claim, correct.

**Good compression:** A: unrelated. B: irrelevant. C: directly supports → correct.

**Why invalid:** Wrong options need only a one-word verdict. Full-sentence elimination prose wastes tokens without adding logical content.

### Anti-Pattern 5: Relational filler instead of arrow notation

**Verbose:** Since the vine stems have low resistance, this means current flows through them, which implies trees are protected from high voltage.

**Bad compression:** Vine stems low resistance, which means current flows, which implies trees protected.

**Good compression:** Vine stems low resistance → current flows → trees protected.

**Why invalid:** "which means" and "which implies" are banned filler. Use `→` chains.
