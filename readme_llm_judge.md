1. uv pip install torch>=2.6
3. delete torch in project.toml
2. uv pip install -e . --no-build-isolation

on a seperate node, launch a grader: run_grader_manual.sh
then get the head node ip on that node, put it in run_with_manual_grader.s, then you are good to go.


## How to modify the prompt the judge
under pipeline-rl/conf/evaluator_prompts