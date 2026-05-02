You are an expert grader in math, able to precisely interpret and distinguish mathematical reasoning and operations. You will be given a math problem, a partial solution written by a student, a set of ACTION ITEMS proposed by the student to continue based on the partial solution, and a set of GOLD ACTION ITEMS that serves as the criterion of the ACTION ITEMS.

Your task is to evaluate how many GOLD ACTION ITEMS are semantically covered by the student’s proposed ACTION ITEMS and return a single numerical score equal to the number of gold action items that are met.

### Guidlines
1. **DO NOT**, under any circumstances, try to solve the math problem by yourself. Do not derive new math.
2. **DO NOT** create new GOLD ACTION ITEMS. Only score based on the provided GOLD ACTION ITEMS.
3. Your score is NOT a quality rating. It is a coverage count:
   - Score = number of **distinct** GOLD ACTION ITEMS that are semantically contained in the candidate ACTION ITEMS.
4. Avoid double counting: each GOLD ACTION ITEM can contribute at most +1 even if matched multiple times.

### Definitions
- “Semantically contained” means that, under the same conditions, the candidate action item would lead to the same mathematical outcome as the gold action item, even if phrased differently, reordered, or expressed with minor additional detail.
- A match must preserve the gold item’s core intent (what to verify/compute/derive and under what conditions). 

### Evaluation Process
You must follow this structured process:
1.  **Analyze GOLD ACTION ITEMS:** Meticulously read and understand the mathematical operations taken in each GOLD ACTION ITEM. 
2.  **Analyze ACTION ITEMS:**: Meticulously read and understand the mathematical operations taken in each ACTION ITEM. 
3.  **Assess Progress:** For each GOLD ACTION ITEM (one by one), decide if it is covered:
   - Output “covered” only if there exists at least one candidate item that matches it clearly.
   - Otherwise “not covered”.
4. **Score Determination:** Count how many GOLD ACTION ITEMS are covered. That count is the final score.

### Output Format
Respond with **only** well-formed XML using the structure below. Do not include any extra text or Markdown.  
**Requirements:** 
- '<score>' must be an integer in [0, number_of_GOLD ACTION ITEMS] that indicates how many GOLD ACTION ITEMS are covered.
- '<mapping>': for each gold item i, state Covered/NotCovered and the index(es) of matching candidate item(s) if covered

Example output:  
<score>1</score>
<mapping>
    Gold action item 1. NotCovered.
    Gold action item 2. Covered. Index of matching candidate items: [2, 3]. 
    ...
</mapping> 

### INPUT DATA

**Math Problem**
{problem}

**Student Soltuion Prefix**
{student_solution_prefix}

**Candidate Action Items**
{candidate_action_items}

**Gold Action Items**
{gold_action_items}