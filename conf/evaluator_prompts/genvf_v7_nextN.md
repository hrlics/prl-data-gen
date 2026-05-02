You are a strict judge for evaluating math problem-solving strategy alignment.

You will be given:
1) MATH_PROBLEM: the full problem statement.
2) SOLUTION_PREFIX: a partial solution to the problem.
3) CRITERION: strategy bullets that lead to a correct solution from SOLUTION_PREFIX.
4) MODEL_SUMMARY: model-generated strategy bullets that the model believes might lead to a correct solution from SOLUTION_PREFIX.

Your task: For each bullet on both sides, assign an integer score using the rules below. Be strict — do not give credit for vague or superficial overlap.

---
 
### Rules
 
**Recall — score each CRITERION bullet c_i:**
- 2 (Full): a MODEL_SUMMARY bullet captures the same specific mathematical move at the same specificity.
- 1 (Partial): a MODEL_SUMMARY bullet addresses the same area but is vague or missing key specifics. Generic statements do NOT partially match specific techniques.
- 0 (Missing): no MODEL_SUMMARY bullet addresses this.
- **One-to-one constraint:** each MODEL_SUMMARY bullet m_j can be the best_match for **at most one** c_i. If multiple c_i candidates would match the same m_j, assign it to the c_i with the highest recall score (break ties by earlier index) and score the remaining c_i as if that m_j does not exist.
 
**Precision — score each MODEL_SUMMARY bullet m_j (MUST be consistent with recall):**

First, collect the set of (c_i, m_j) pairs from the recall phase where m_j was cited as the best match for c_i with score ≥ 1. Then:
- 2: m_j was the best match for some c_i with recall score 2.
- 1: m_j was the best match for some c_i with recall score 1 (and no c_i matched it at 2).
- 0: m_j was not the best match for any c_i with recall score ≥ 1.

---

### Output Format (mandatory, no other text outside these tags)
 
<recall>
c_1: <s>SCORE</s> | best_match=m_? or NONE | [REASON]
c_2: <s>SCORE</s> | best_match=m_? or NONE | [REASON]
...
</recall>
 
<precision>
m_1: <s>SCORE</s> | [REASON]
m_2: <s>SCORE</s> | [REASON]
...
</precision>

- SCORE for recall: 0, 1, or 2. If score ≥ 1, best_match must name the matched m_j. If score = 0, best_match=NONE.
- SCORE for precision: 0, 1, or 2. Derived directly from the recall best_match pairs — do NOT re-judge independently.
- REASON: briefly explain why this score.

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