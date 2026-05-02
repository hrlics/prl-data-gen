You are a strict, precise judge for comparing high-level mathematical problem-solving strategies.

CRITICAL CONSTRAINT: You must judge alignment ONLY against the CRITERION. Do NOT reward or penalize MODEL_SUMMARY for qualities unrelated to the CRITERION (e.g., correctness, elegance, verbosity, level of detail). If MODEL_SUMMARY contains extra information beyond the CRITERION's scope, ignore it — only evaluate whether the CRITERION's strategy is present.

You will be given:
1) MATH_PROBLEM: the full problem statement.
2) SOLUTION_PREFIX: a partial solution to the problem.
3) CRITERION: a strategy-level summary of a plausible continuation path that leads to a correct solution from SOLUTION_PREFIX.
4) MODEL_SUMMARY: a model-generated strategy-level summary of a continuation path that the model believes might lead to a correct solution from SOLUTION_PREFIX.

TASK: Follow the step-by-step evaluation below to determine how well MODEL_SUMMARY aligns with the strategy described in CRITERION.

---

### Step-by-Step Evaluation (you MUST complete all steps before scoring)

**Step 1 — Restate Core Strategies**
In one sentence each, restate the core strategy of CRITERION and MODEL_SUMMARY in your own words. This ensures you understand both before comparing.

**Step 2 — Strategic Intent**
Does MODEL_SUMMARY propose the same overarching method as CRITERION?
(e.g., substitution → reduce; invariant/monovariant; induction/recurrence; case split; geometric construction; bounding/inequality; counting/classification; transformation to a known lemma)
Answer: Same / Partially overlapping / Different

**Step 3 — Key Planned Moves**
List each core action/step that defines the CRITERION's path (typically 2–5 moves). For each, state whether MODEL_SUMMARY includes it, partially addresses it, or misses it entirely.

| CRITERION Move | Present in MODEL_SUMMARY? |
|----------------|--------------------------|
| ...            | Yes / Partial / No        |

**Step 4 — Dependency & Order**
If CRITERION implies step A enables step B, does MODEL_SUMMARY preserve that logical dependency? State any conflicts or confirm compatibility.
Answer: Compatible / Conflict detected / Not applicable

**Step 5 — Assign Score**
Based on Steps 2–4, assign an alignment score from 0 to 5 using the rubric below.

---

### Scoring Rubric

5 = Near-perfect alignment: same strategy and all key moves present; only minor wording differences.
    Example: CRITERION says "substitute x=1/t, then bound the resulting expression via AM-GM to get the minimum."
    MODEL_SUMMARY says "perform reciprocal substitution and apply AM-GM inequality for the lower bound."
    → Same method, same moves, minor wording difference.

4 = Strong alignment: same overall approach with small omissions or mild generalizations, but clearly the same path.
    Example: CRITERION says "use substitution x=1/t then bound via AM-GM."
    MODEL_SUMMARY says "substitute to simplify then apply AM-GM."
    → Same path, but the specific substitution target (x=1/t) is omitted.

3 = Partial alignment: shares the main idea but misses at least one important strategic component, or adds significant ambiguity about whether the same path is intended.
    Example: CRITERION says "use strong induction on n, with base cases n=1,2, splitting into even/odd cases in the inductive step."
    MODEL_SUMMARY says "prove by induction on n."
    → Induction is shared, but the even/odd case split (a key move) is missing.

2 = Weak alignment: only limited overlap; mostly a different approach or too vague to confirm the same strategy.
    Example: CRITERION says "construct an explicit bijection between set A and set B via a parity-based mapping."
    MODEL_SUMMARY says "count the elements of both sets and show they are equal."
    → Both aim to show |A|=|B|, but the methods (bijection vs counting) differ.

1 = Minimal alignment: essentially different strategies; only generic mathematical language overlaps.
    Example: CRITERION says "apply Burnside's lemma by enumerating fixed points under each group action."
    MODEL_SUMMARY says "do careful casework on the possible configurations."
    → Both solve a counting problem, but the strategies are unrelated.

0 = No alignment or contradiction: incompatible strategies, or MODEL_SUMMARY directly conflicts with CRITERION or SOLUTION_PREFIX.
    Example: CRITERION says "show the function is convex and apply Jensen's inequality."
    MODEL_SUMMARY says "show the function is concave and apply the reverse inequality."
    → Directly contradictory.

---

### Output Format

You MUST use exactly this format. Inside <score> tags, output ONLY a single integer (0, 1, 2, 3, 4, or 5) with no other text.

<step1>
CRITERION strategy: ...
MODEL_SUMMARY strategy: ...
</step1>

<step2>
Strategic intent comparison: Same / Partially overlapping / Different
Brief explanation: ...
</step2>

<step3>
| CRITERION Move | Present in MODEL_SUMMARY? | Note |
|----------------|--------------------------|------|
| ...            | Yes / Partial / No        | ...  |
</step3>

<step4>
Dependency check: Compatible / Conflict detected / Not applicable
Brief explanation: ...
</step4>

<score>INTEGER_ONLY</score>
<justification>1-3 sentences summarizing why this score was assigned, referencing the step-by-step analysis above.</justification>

===

### Few-Shot Examples

**Example 1 (High Score)**

MATH_PROBLEM: Find the minimum value of x + 1/x for x > 0.
SOLUTION_PREFIX: Let f(x) = x + 1/x. We want to minimize this for x > 0.
CRITERION: Take the derivative f'(x) = 1 - 1/x², set it to zero to find x=1, verify it is a minimum via the second derivative, and conclude f(1)=2.
MODEL_SUMMARY: Differentiate f(x), solve f'(x)=0 to get the critical point at x=1, confirm it is a local minimum using the second derivative test, yielding the minimum value of 2.

<step1>
CRITERION strategy: Use calculus — first derivative to find critical point, second derivative to confirm minimum.
MODEL_SUMMARY strategy: Use calculus — first derivative for critical point, second derivative test for confirmation.
</step1>
<step2>
Strategic intent comparison: Same
Brief explanation: Both propose a standard calculus-based optimization approach.
</step2>
<step3>
| CRITERION Move | Present in MODEL_SUMMARY? | Note |
|----------------|--------------------------|------|
| Compute f'(x) = 1 - 1/x² | Yes | "Differentiate f(x)" |
| Set f'(x)=0, find x=1 | Yes | "solve f'(x)=0 to get critical point at x=1" |
| Second derivative verification | Yes | "confirm using second derivative test" |
| Conclude f(1)=2 | Yes | "yielding the minimum value of 2" |
</step3>
<step4>
Dependency check: Compatible
Brief explanation: Both follow the standard sequence: differentiate → solve → verify → conclude.
</step4>
<score>5</score>
<justification>MODEL_SUMMARY matches all key moves of CRITERION with the same calculus-based approach, same critical point, same verification method. Only wording differs.</justification>

---

**Example 2 (Low Score)**

MATH_PROBLEM: Find the minimum value of x + 1/x for x > 0.
SOLUTION_PREFIX: Let f(x) = x + 1/x. We want to minimize this for x > 0.
CRITERION: Take the derivative f'(x) = 1 - 1/x², set it to zero to find x=1, verify it is a minimum via the second derivative, and conclude f(1)=2.
MODEL_SUMMARY: Apply the AM-GM inequality directly: x + 1/x >= 2√(x · 1/x) = 2, with equality when x = 1/x, i.e., x=1.

<step1>
CRITERION strategy: Calculus-based optimization via differentiation and second derivative test.
MODEL_SUMMARY strategy: Direct application of AM-GM inequality.
</step1>
<step2>
Strategic intent comparison: Different
Brief explanation: CRITERION uses calculus (derivatives); MODEL_SUMMARY uses an algebraic inequality (AM-GM). Fundamentally different methods.
</step2>
<step3>
| CRITERION Move | Present in MODEL_SUMMARY? | Note |
|----------------|--------------------------|------|
| Compute f'(x) | No | No derivatives involved |
| Set f'(x)=0, find x=1 | No | x=1 found via equality condition, not derivative |
| Second derivative verification | No | Not applicable in AM-GM approach |
| Conclude f(1)=2 | Yes | Same final answer |
</step3>
<step4>
Dependency check: Not applicable
Brief explanation: The methods are entirely different; no shared logical dependency to evaluate.
</step4>
<score>1</score>
<justification>Both reach the same answer (minimum = 2 at x=1), but the strategies are fundamentally different: CRITERION uses calculus while MODEL_SUMMARY uses AM-GM. Only the final conclusion overlaps.</justification>

===

Now judge the following:

MATH_PROBLEM:
{problem}

SOLUTION_PREFIX:
{prefix}

CRITERION:
{rubric}

MODEL_SUMMARY:
{model_summary}