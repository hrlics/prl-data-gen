You are a strategy planner. Given a math problem, a partial solution prefix, and the ground-truth full completion, predict the remaining high-level strategy bullets that bridge the prefix to the final answer.

PROBLEM:
{problem}

SOLUTION_PREFIX:
{prefix}

GROUND_TRUTH_COMPLETION:
{gt_completion}

Output the strategy in exactly this format (and nothing else after):

PREDICTED REMAINING STEPS:
- bullet 1 (concrete mathematical move)
- bullet 2
- ...
