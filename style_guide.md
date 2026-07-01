# Grug Style Guide for Chain of Thought Compression

This document defines rules for compressing verbose chain-of-thought (CoT) traces into a token-efficient, telegraphic "Grug" style using short, sentence-fragment-based prose. Use these guidelines in compression prompts and during manual spot-checks.

## Core Objective

Reduce token consumption in the thinking phase by eliminating linguistic overhead while preserving the complete logical chain and exact mathematical correctness.

## Scope

- Compress **only** the reasoning trace (the thinking block). Do not alter, restate, shorten, or append the final answer.
- Do not fix, improve, or rewrite reasoning that leads to a wrong answer. Shorten the trace as given; upstream validation ensures the raw answer matches ground truth before compression.
- Grug style applies to `<think>` content only. Final answers in SFT data stay clear and unchanged.

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

## Compression Quality Bar

A compressed trace is valid only if it meets these criteria:

- **Same Answer:** The final answer matches the raw trace exactly.
- **Token Efficiency:** The compressed thinking block uses 50% or fewer tokens than the verbose thinking block.
- **Completeness:** No logical step required to arrive at the answer is omitted.

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
Options A evap, B cond, C precip, D transp. Need gas (vapor) to liquid. Evap L->G, cond G->L. Precipitation falling, transp plant. Condensation match B.
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
Passage Pacific largest ocean, covers >30% Earth surface. Question asks >50% surface. 30% < 50%, so not more than half.
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

### Anti-Pattern 2: Answer changed or restated in thinking block

**Verbose:** Sarah has $10. Mark has 3× Sarah = $30. Total $40.

**Bad compression:** Sarah 10. Mark 30. Total 40. **Answer: forty dollars.**

**Why invalid:** Final answer must not appear in the compressed thinking block; the answer field stays separate and unchanged.

### Anti-Pattern 3: Key-value or line-per-step format

**Verbose:** Metal conducts heat. Wood insulates. Metal spoon heats first.

**Bad compression:**

```text
facts: metal conductor, wood insulator
conclusion: metal spoon
```

**Why invalid:** Uses key-value labels and line breaks instead of continuous telegraphic fragments.
