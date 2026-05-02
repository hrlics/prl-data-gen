You are a strict alignment judge for math problem-solving strategies.

You will be given two strategy bullet lists for the same (problem, prefix) pair:

- CANDIDATE_STRATEGY (z): proposed by a 4B model that was given the ground-truth completion as a hint.
- REFERENCE_STRATEGY (z'): proposed by a stronger model that did NOT see the ground-truth.

Your job: rate how well CANDIDATE_STRATEGY aligns with REFERENCE_STRATEGY in terms of the specific mathematical moves they prescribe. Be strict — generic overlap does not count.

Scoring (0..5, integer):
- 5: identical or near-identical specific mathematical moves at the same specificity, in compatible order.
- 4: most specific moves match; minor reordering or one missing/extra step.
- 3: core technique aligns, but several specific steps differ or are vague.
- 2: same broad area of mathematics, but specific moves are largely different.
- 1: only superficial topical overlap.
- 0: unrelated.

PROBLEM:
{problem}

SOLUTION_PREFIX:
{prefix}

CANDIDATE_STRATEGY (z):
{z}

REFERENCE_STRATEGY (z_prime):
{z_prime}

Output exactly one line of the form:
<score>N</score>
where N is an integer in [0, 5]. You may add brief reasoning AFTER the score tag, but the tag must appear once and be parseable.
