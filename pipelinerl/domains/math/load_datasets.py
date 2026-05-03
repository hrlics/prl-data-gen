import json
import logging
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import datasets
import hydra
from datasets import load_dataset
from omegaconf import DictConfig, ListConfig, OmegaConf
import pandas as pd
import numpy as np
from scipy.stats import norm
from transformers import AutoTokenizer

"""
math_verify expects the following LaTeX format for the gold answer (with $ or \\boxed).
For example, this will parse correctly:
\\boxed{\\begin{pmatrix} -\\frac{1}{3} \\ \\frac{2}{3} \\ \\frac{5}{3} \\end{pmatrix}}$
and this will not parse:
\\begin{pmatrix} -\\frac{1}{3} \\ \\frac{2}{3} \\ \\frac{5}{3} \\end{pmatrix}
"""

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _get_tokenizer(model_path: str):
    return AutoTokenizer.from_pretrained(model_path)


def process_genvf_summary_llm_judge(
    dataset,
    dataset_name,
    squash_score=False,
    model_path=None,
    max_input_tokens=25000,
):

    assert dataset['input_to_VF'] is not None, "input_to_VF field is required for genvf summary llm judge datasets"
    assert dataset['high_level_suffix_summary'] is not None, "high_level_suffix_summary field is required for genvf summary llm judge datasets"

    if model_path is None:
        raise ValueError(
            "process_genvf_summary_llm_judge requires model_path to tokenize and "
            "filter input_to_VF by length. Pass it via dataset_loader_params.model_path."
        )
    tokenizer = _get_tokenizer(model_path)

    total = 0
    dropped = 0
    max_len_seen = 0

    for item in dataset:
        total += 1
        summary_list = [i for i in item['high_level_suffix_summary'] if i!=""] # filter out empty summaries
        if summary_list == []:
            dropped += 1
          
        problem = item['problem']
        prefix = item['prefix']
        prompt = item['input_to_VF']

        n_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        if n_tokens > max_len_seen:
            max_len_seen = n_tokens
        if n_tokens > max_input_tokens:
            dropped += 1
            continue

        yield {
            "problem": problem,
            "prefix": prefix,
            "prefix_summary_steps": item.get('prefix_summary_steps', prefix),
            "answer": item["answer"],
            "gemini_summary_list": summary_list,
            "dataset": dataset_name,
            "task": prompt,
            "squash_score": squash_score,
        }

    bar = "!" * 88
    logger.warning(
        "\n" + bar + "\n"
        f"!!! [INPUT_TO_VF LENGTH FILTER]\n"
        f"!!!   dataset     : {dataset_name}\n"
        f"!!!   tokenizer   : {model_path}\n"
        f"!!!   threshold   : input_to_VF > {max_input_tokens} tokens  -> DROPPED\n"
        f"!!!   dropped     : {dropped} / {total} samples "
        f"({(dropped / total * 100) if total else 0:.2f}%)\n"
        f"!!!   kept        : {total - dropped}\n"
        f"!!!   max_len_seen: {max_len_seen} tokens\n"
        + bar
    )


def _pick_score7_suffix_index(proof_scores) -> int:
    """Return the suffix_id of the first proof_scores entry with points==7,
    falling back to 0 when proof_scores is missing/empty/no-7s.
    """
    if not proof_scores:
        return 0
    try:
        for entry in proof_scores:
            if isinstance(entry, dict) and int(entry.get("points", 0)) == 7:
                return int(entry.get("suffix_id", 0))
    except (TypeError, ValueError):
        return 0
    return 0


_DATA_GENERATOR_TASK_PROMPT_NAME = "data_generator_task_v1"


def _evaluator_prompts_dir() -> Path:
    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        if (parent / "conf" / "evaluator_prompts").is_dir():
            return (parent / "conf" / "evaluator_prompts").resolve()
    raise FileNotFoundError(
        f"Could not locate conf/evaluator_prompts/ relative to {module_path}"
    )


@lru_cache(maxsize=8)
def _load_data_generator_task_template(prompt_name: str) -> str:
    filename = prompt_name if prompt_name.endswith(".md") else f"{prompt_name}.md"
    prompt_path = _evaluator_prompts_dir() / filename
    if not prompt_path.is_file():
        raise FileNotFoundError(
            f"Data-generator task prompt '{filename}' not found in {prompt_path.parent}"
        )
    return prompt_path.read_text(encoding="utf-8")


def process_genvf_data_generator(
    dataset,
    dataset_name,
    model_path=None,
    max_input_tokens=32000,
    task_prompt_name=_DATA_GENERATOR_TASK_PROMPT_NAME,
):
    """Processor for the data-generator RL setup.

    Reads `problem`, `prefix`, `suffix_response` (list[str]), `proof_scores`,
    `rubrics`, `row_id` from each row. Picks the score-7 suffix as
    `gt_completion` (fallback to index 0). Builds the data-generator's user
    prompt that includes (problem, prefix, gt_completion). The prompt template
    is loaded from `conf/evaluator_prompts/<task_prompt_name>.md`.
    """
    if model_path is None:
        raise ValueError(
            "process_genvf_data_generator requires model_path to tokenize the "
            "constructed task and filter by length. Pass via dataset_loader_params.model_path."
        )
    tokenizer = _get_tokenizer(model_path)
    task_template = _load_data_generator_task_template(task_prompt_name)

    total = 0
    dropped_no_suffix = 0
    dropped_long = 0
    max_len_seen = 0

    for item in dataset:
        total += 1
        suffix_responses = item.get("suffix_response") or []
        if not suffix_responses:
            dropped_no_suffix += 1
            continue
        idx = _pick_score7_suffix_index(item.get("proof_scores"))
        if idx >= len(suffix_responses):
            idx = 0
        gt_completion = suffix_responses[idx]
        if not gt_completion:
            dropped_no_suffix += 1
            continue

        problem = item["problem"]
        prefix = item["prefix"]
        marking_scheme = item.get("rubrics") or ""

        task = task_template.format(
            problem=problem,
            prefix=prefix,
            gt_completion=gt_completion,
        )

        n_tokens = len(tokenizer.encode(task, add_special_tokens=False))
        if n_tokens > max_len_seen:
            max_len_seen = n_tokens
        if n_tokens > max_input_tokens:
            dropped_long += 1
            continue

        yield {
            "problem": problem,
            "prefix": prefix,
            "gt_completion": gt_completion,
            "marking_scheme": marking_scheme,
            "task": task,
            "dataset": dataset_name,
            "row_id": item.get("row_id"),
            "data_generator_mode": True,
        }

    bar = "!" * 88
    kept = total - dropped_no_suffix - dropped_long
    logger.warning(
        "\n" + bar + "\n"
        f"!!! [DATA_GENERATOR LENGTH FILTER]\n"
        f"!!!   dataset           : {dataset_name}\n"
        f"!!!   tokenizer         : {model_path}\n"
        f"!!!   threshold         : task > {max_input_tokens} tokens  -> DROPPED\n"
        f"!!!   dropped (no suffix): {dropped_no_suffix} / {total}\n"
        f"!!!   dropped (long)   : {dropped_long} / {total}\n"
        f"!!!   kept              : {kept}\n"
        f"!!!   max_len_seen      : {max_len_seen} tokens\n"
        + bar
    )


def process_genvf_summary_llm_judge_nextN(
    dataset,
    dataset_name,
    squash_score=False,
    model_path=None,
    max_input_tokens=24000,
):

    assert dataset['input_to_VF'] is not None, "input_to_VF field is required for genvf summary llm judge datasets"
    assert dataset['detailed_suffix_summary'] is not None, "detailed_suffix_summary field is required for genvf summary llm judge datasets"

    if model_path is None:
        raise ValueError(
            "process_genvf_summary_llm_judge_nextN requires model_path to tokenize and "
            "filter input_to_VF by length. Pass it via dataset_loader_params.model_path."
        )
    tokenizer = _get_tokenizer(model_path)

    total = 0
    dropped_empty = 0
    dropped_long = 0
    max_len_seen = 0

    for item in dataset:
        total += 1
        summary_list = [i for i in item['detailed_suffix_summary'] if i!=""] # filter out empty summaries
        if summary_list == []:
            dropped_empty += 1
            continue

        problem = item['problem']
        prefix = item['prefix']
        prompt = item['input_to_VF']

        n_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        if n_tokens > max_len_seen:
            max_len_seen = n_tokens
        if n_tokens > max_input_tokens:
            dropped_long += 1
            continue

        yield {
            "problem": problem,
            "prefix": prefix,
            "prefix_summary_steps": item.get('prefix_summary_steps', prefix),
            "answer": item["answer"],
            "gemini_summary_list": summary_list,
            "dataset": dataset_name,
            "task": prompt,
            "squash_score": squash_score,
        }

    bar = "!" * 88
    kept = total - dropped_empty - dropped_long
    logger.warning(
        "\n" + bar + "\n"
        f"!!! [INPUT_TO_VF LENGTH FILTER / nextN]\n"
        f"!!!   dataset        : {dataset_name}\n"
        f"!!!   tokenizer      : {model_path}\n"
        f"!!!   threshold      : input_to_VF > {max_input_tokens} tokens  -> DROPPED\n"
        f"!!!   dropped (long) : {dropped_long} / {total} samples "
        f"({(dropped_long / total * 100) if total else 0:.2f}%)\n"
        f"!!!   dropped (empty): {dropped_empty} / {total} samples\n"
        f"!!!   kept           : {kept}\n"
        f"!!!   max_len_seen   : {max_len_seen} tokens\n"
        + bar
    )


def process_VF(dataset, dataset_name, use_hl_gauss, sigma=0.06, reverse_kl=False):
    for item in dataset:
        prompt = item['prefix']
        answer = "\\boxed{" + item["answer"] + "}"
        
        if use_hl_gauss:
            mean_reward = item['mean_reward']

            def hl_gauss_reward_integration(target_val, sigma):
                """
                HL-Gauss based on integration over defined boundaries.
                """
                probs = []
                boundaries = np.array([0.0, 0.03125, 0.234375, 0.546875, 0.90625, 1.0]) # based on quintiles on train set

                for i in range(len(boundaries) - 1):
                    lower_bound = boundaries[i]
                    upper_bound = boundaries[i+1]
                    
                    # (Probability Mass) p = CDF(upper) - CDF(lower)
                    p = norm.cdf(upper_bound, loc=target_val, scale=sigma) - \
                        norm.cdf(lower_bound, loc=target_val, scale=sigma)
                    probs.append(p)
                
                probs = np.array(probs)
                
                # normalization, because gaussian is from -inf to +inf, but we truncate to [0,1] --> truncated gaussian distribution
                return probs / probs.sum()

            reward_probs = hl_gauss_reward_integration(mean_reward, sigma=sigma)
            yield {
                "dataset": dataset_name,
                "task": prompt,
                "answer": answer,
                "reward_probs": reward_probs,
                "reverse_kl": reverse_kl
            }
        else:
            yield {
                "dataset": dataset_name,
                "task": prompt,
                "answer": answer,
            }


def process_proof_problem(dataset, dataset_name):
    for row in dataset:
        yield {
            "dataset": dataset_name,
            "task": row["problem"],           # problem statement
            "answer": row["solution"],        # reference solution
            "schema": row["schema_0"],        # marking scheme
        }

def process_eurus(dataset):
    for item in dataset:
        if item["ability"] != "math":
            # discard the coding problems for now
            yield None
        answer = "\\boxed{" + str(item["reward_model"]["ground_truth"]) + "}"
        task = item["prompt"][1]["content"]
        task = task.replace("\n\nPresent the answer in LaTex format: \\boxed{Your answer}", "")
        yield {
            "dataset": item["data_source"],
            "task": task,
            "answer": answer,
        }


def process_math(dataset, dataset_name):
    for item in dataset:
        if "correctness_math_verify" in item:
            if not any(item["correctness_math_verify"]):
                # correctness cannot be verified with math_verify
                yield None
                continue
        if "problem" in item:
            question = item["problem"]
        elif "question" in item:
            question = item["question"]
        else:
            yield None
            continue
        if "subject" in item and "type" not in item:
            item["type"] = item["subject"]
        if "answer" in item:
            answer = "\\boxed{" + item["answer"] + "}"
        elif "solution" in item:
            answer = item["solution"]
        else:
            yield None
            continue
        sample = {
            "dataset": dataset_name,
            "level": item.get("level", ""),
            "type": item.get("type", ""),
            "task": question,
            "answer": answer,
        }
        yield sample


def process_gsm8k(dataset, dataset_name):
    for item in dataset:
        sample = {
            "dataset": dataset_name,
            "task": item["question"],
            "answer": item["answer"].split("####")[1],
        }
        yield sample


def process_limo(dataset):
    for item in dataset:
        task = item["question"]
        answer = "\\boxed{" + str(item["answer"]) + "}"
        yield {
            "dataset": "limo",
            "task": task,
            "answer": answer,
        }


def process_aime_and_amc(dataset, dataset_name):
    for item in dataset:
        task = item["problem"]
        answer = "\\boxed{" + str(item["answer"]) + "}"
        yield {
            "dataset": dataset_name,
            "task": task,
            "answer": answer,
        }


def process_open_reasoner(dataset, dataset_name):
    for item in dataset:
        # Note: Open Reasoner tasks sometimes have preamble, e.g.
        # - Example 31 (2004 College Entrance Examination Hunan Paper)
        # - 8.
        # - 4. (7 points)
        # We are currently ignoring the preamble
        task = item["0"]["value"]
        answer = "\\boxed{" + item["1"]["ground_truth"]["value"] + "}"
        yield {"dataset": dataset_name, "task": task, "answer": answer}

def process_pope_local(dataset, dataset_name):
    for _, item in dataset.iterrows():
        task = item['prompt'][0]['content']
        answer = "\\boxed{" + item['reward_model']['ground_truth'] + "}"
        yield {"dataset": dataset_name + f"_{item['data_source'].replace('-', '_')}", "task": task, "answer": answer}

def process_pope(dataset, dataset_name):
    for item in dataset:
        task = item['prompt'][0]['content']
        answer = "\\boxed{" + item['reward_model']['ground_truth'] + "}"
        yield {"dataset": dataset_name + f"_{item['data_source'].replace('-', '_')}", "task": task, "answer": answer}


def process_pope_mix(dataset, dataset_name):
    for item in dataset:
        task = item['prompt'][0]['content']
        answer = "\\boxed{" + item['reward_model']['ground_truth'] + "}"
        yield {"dataset": dataset_name + f"_{item['level'].replace('-', '_')}", "task": task, "answer": answer}


def process_gpqa(dataset, dataset_name):
    for item in dataset:
        yield {
            "dataset": dataset_name,
            "task": item["problem"],
            "answer": item["solution"],
        }


def process_countdown(dataset):
    counter = 0
    for item in dataset:
        problem = item["prompt"][0]["content"]
        problem = problem.split("<|im_start|>user")[-1]
        problem = problem.split("<|im_start|>assistant")[0]
        problem = problem.split("<|im_end|>")[0]
        problem = problem.strip()
        answer = "-".join(["countdown", str(item["target"]), str(item["nums"])])
        yield {"dataset": "countdown", "task": problem, "answer": answer, "id": counter}
        counter += 1


def load_math(split):
    # FIXME?
    data = []
    for config in [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]:
        dataset = load_dataset("EleutherAI/hendrycks_math", config, split=split, trust_remote_code=True)
        for sample in dataset:
            data.append(sample)
    return datasets.Dataset.from_list(data)


def _load_aime_dataset(year: int, upsample_factor: int = 0) -> list[dict]:
    if year == 2025:
        aime_dataset = load_dataset("MathArena/aime_2025", split="train", trust_remote_code=True)
    else:
        aime_dataset = load_dataset("AI-MO/aimo-validation-aime", split="train", trust_remote_code=True)
        aime_dataset = aime_dataset.filter(lambda x: str(year) in x["url"])

    dataset_name = f"aime_{year}" + ("" if upsample_factor > 0 else "_original")
    samples = [s for s in process_aime_and_amc(aime_dataset, dataset_name) if s is not None]

    original_size = len(samples)
    if upsample_factor > 0:
        samples *= upsample_factor

    logger.info(
        f"Loading aime {year} dataset: {len(samples)} samples"
        + (f" (upsampled from {original_size})" if upsample_factor > 0 else "")
    )
    return add_ids(samples)


def _load_amc_dataset(year: int, upsample_factor: int = 0) -> list[dict]:
    amc_dataset = load_dataset("AI-MO/aimo-validation-amc", split="train", trust_remote_code=True)
    amc_dataset = amc_dataset.filter(lambda x: str(year) in x["url"])

    dataset_name = f"amc_{year}" + ("" if upsample_factor > 0 else "_original")
    samples = [s for s in process_aime_and_amc(amc_dataset, dataset_name) if s is not None]

    original_size = len(samples)
    if upsample_factor > 0:
        samples *= upsample_factor

    logger.info(
        f"Loading amc {year} dataset: {len(samples)} samples"
        + (f" (upsampled from {original_size})" if upsample_factor > 0 else "")
    )
    return add_ids(samples)


def add_ids(dataset: list[dict]):
    for i, entry in enumerate(dataset):
        entry["id"] = i
    return dataset


def load_datasets(
    dataset_names: List[str | Dict[str, Any]] | Dict[str, Any] | str | None,
    seed: int | None = None,
    model_path: str | None = None,
) -> List[Tuple[str, Dict]]:
    if dataset_names is None:
        return []

    if isinstance(dataset_names, (DictConfig, ListConfig)):
        dataset_names = OmegaConf.to_container(dataset_names, resolve=True)

    if isinstance(dataset_names, dict):
        dataset_names = [dataset_names]
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    elif not isinstance(dataset_names, list):
        dataset_names = list(dataset_names)

    datasets = []

    for dataset_spec in dataset_names:
        if isinstance(dataset_spec, dict):
            hub_id = dataset_spec.get("hub_id")
            if not hub_id:
                raise ValueError("Hub dataset specs must include a 'hub_id' field.")
            config = dataset_spec.get("config")
            split = dataset_spec.get("split", "train")
            trust_remote_code = dataset_spec.get("trust_remote_code", True)
            load_args: Tuple[Any, ...] = (hub_id,)
            if config is not None:
                load_args += (config,)
            dataset = load_dataset(*load_args, split=split, trust_remote_code=trust_remote_code)
            if dataset_spec.get("data_generator"):
                # Data-generator RL: predict z from (problem, prefix, GT completion).
                dataset_name = f"{hub_id.split('/')[-1]}_{split}" if split != "train" else hub_id.split("/")[-1]
                task_prompt_name = dataset_spec.get("task_prompt", _DATA_GENERATOR_TASK_PROMPT_NAME)
                samples = [
                    s for s in process_genvf_data_generator(
                        dataset,
                        dataset_name,
                        model_path=model_path,
                        task_prompt_name=task_prompt_name,
                    ) if s is not None
                ]
            elif hub_id in ["hf-imo-colab/olympiads-proof-schema", "hf-imo-colab/olympiads-proof-schema-benchmark", "hf-imo-colab/olympiads-proof-schema-cleaned"]:
                samples = [s for s in process_proof_problem(dataset, hub_id.split("/")[-1]) if s is not None]
            # multi-class classification datasets
            elif hub_id in [
                'haoranli-ml/value_func_data_v1_with_Q_labels_and_sum_prefix-half_oracle_train-PRL', 'haoranli-ml/VF_10k_distr_out_balanced', 
                'haoranli-ml/VF_8k_distr_out_cdf_bin', "haoranli-ml/VF_8k_distr_out_cdf_bin_future_prompts"
            ]:
                use_hl_gauss = dataset_spec.get("use_hl_gauss", False)
                reverse_kl = dataset_spec.get("reverse_kl", False)
                sigma = dataset_spec.get("sigma", 0.06)
                # Include split name in dataset name to distinguish different test sets
                dataset_name = f"{hub_id.split('/')[-1]}_{split}" if split != "train" else hub_id.split("/")[-1]
                samples = [s for s in process_VF(dataset, dataset_name, use_hl_gauss, sigma, reverse_kl) if s is not None]
            elif hub_id.startswith("haoranli-ml/genvf"):
                dataset_name = f"{hub_id.split('/')[-1]}_{split}" if split != "train" else hub_id.split("/")[-1]
                squash_score = dataset_spec.get("squash_score", False)
                if squash_score:
                    logger.info(f"\n\n\n===== USING SQUASHED JUDGE SCORE FOR {dataset_name} ======\n\n\n")
                if "nextN" in hub_id:
                    logger.info(f"\n\n\n===== USING nextN TARGET for {dataset_name} ======\n\n\n")
                    samples = [
                        s for s in process_genvf_summary_llm_judge_nextN(
                            dataset,
                            dataset_name,
                            squash_score=squash_score,
                            model_path=model_path,
                        ) if s is not None
                    ]
                else:
                    samples = [
                        s for s in process_genvf_summary_llm_judge(
                            dataset,
                            dataset_name,
                            squash_score=squash_score,
                            model_path=model_path,
                        ) if s is not None
                    ]
            else:
                samples = [dict(row) for row in dataset]
            for sample in samples:
                sample.setdefault("dataset", hub_id)
            logger.info(
                f"Loading hub dataset {hub_id}"
                + (f"/{config}" if config else "")
                + f" split={split}: {len(samples)} samples"
            )
            datasets += add_ids(samples)
        elif isinstance(dataset_spec, str) and "/" in dataset_spec:
            dataset = load_dataset(dataset_spec, split="train", trust_remote_code=True)
            samples = [dict(row) for row in dataset]
            for sample in samples:
                sample.setdefault("dataset", dataset_spec)
            logger.info(f"Loading hub dataset {dataset_spec} split=train: {len(samples)} samples")
            datasets += add_ids(samples)

    if "eurus_train" in dataset_names:
        dataset = load_dataset("PRIME-RL/Eurus-2-RL-Data", split="train", trust_remote_code=True)
        samples = [s for s in process_eurus(dataset) if s is not None]
        logger.info(f"Loading eurus train dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    # great for debugging since its much smaller than eurus train
    if "eurus_validation" in dataset_names:
        dataset = load_dataset("PRIME-RL/Eurus-2-RL-Data", split="validation", trust_remote_code=True)
        samples = [s for s in process_eurus(dataset) if s is not None]
        logger.info(f"Loading eurus validation dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    if "math_train" in dataset_names:
        # math_dataset = load_math("train")
        dataset = load_dataset("hendrycks/competition_math", split="train", trust_remote_code=True)
        samples = [s for s in process_math(dataset, "math_train") if s is not None]
        logger.info(f"Loading math train dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    if "math_simplerl_train" in dataset_names:
        # SimpleRL MATH dataset
        #   level 3-5 math problems from both train and test sets of the original MATH dataset (excluding problems from MATH-500)
        # math_dataset = load_math("train")
        dataset = load_dataset(
            "json",
            data_files="https://raw.githubusercontent.com/hkust-nlp/simpleRL-reason/refs/heads/v0/train/data/math_level3to5_data_processed_with_qwen_prompt.json",
            split="train",
            trust_remote_code=True,
        )
        samples = [s for s in process_math(dataset, "math_simplerl_train") if s is not None]
        logger.info(f"Loading math simplerl train dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    if "simplerl_math_subset_1000" in dataset_names:
        # SimpleRL MATH dataset subset
        #   level 3-5 math problems from both train and test sets of the original MATH dataset (excluding problems from MATH-500)
        # math_dataset = load_math("train")
        dataset = load_dataset(
            "json",
            data_files="https://raw.githubusercontent.com/hkust-nlp/simpleRL-reason/refs/heads/v0/train/data/math_level3to5_data_processed_with_qwen_prompt.json",
            split="train",
            trust_remote_code=True,
        )
        samples = [s for s in process_math(dataset, "math_simplerl_subset") if s is not None]
        if seed is not None:
            random.seed(seed)
        random.shuffle(samples)
        samples = samples[:1000]
        logger.info(f"Loading math simplerl subset test dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    if "deepscaler_preview" in dataset_names:
        dataset = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train", trust_remote_code=True)
        samples = [s for s in process_math(dataset, "deepscaler") if s is not None]
        logger.info(f"Loading deepscaler preview train dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    if "math_test" in dataset_names:
        # math_dataset = load_math("test")
        dataset = load_dataset("hendrycks/competition_math", split="test", trust_remote_code=True)
        samples = [s for s in process_math(dataset, "math_test") if s is not None]
        logger.info(f"Loading math test dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    if "omni_math_500" in dataset_names:
        dataset = load_dataset("reliable-agents/Omni-MATH-500", split="test", trust_remote_code=True)
        samples = [s for s in process_math(dataset, "omni_math_500") if s is not None]
        logger.info(f"Loading omni math 500 dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    if "math_500" in dataset_names:
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test", trust_remote_code=True)
        samples = [s for s in process_math(dataset, "math_500") if s is not None]
        logger.info(f"Loading math 500 dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    if "open_r1_math_220k" in dataset_names:
        dataset = load_dataset("open-r1/OpenR1-Math-220k", split="default", trust_remote_code=True)
        samples = [s for s in process_math(dataset, "open_r1_math_220k") if s is not None]
        logger.info(f"Loading open r1 math 220k dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    if "gpqa_main" in dataset_names:
        dataset = load_dataset("hendrydong/gpqa_main", split="test", trust_remote_code=True)
        samples = [s for s in process_gpqa(dataset, "gpqa_main") if s is not None]
        logger.info(f"Loading gpqa main dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    if "gpqa_diamond" in dataset_names:
        dataset = load_dataset("hendrydong/gpqa_diamond", split="test", trust_remote_code=True)
        samples = [s for s in process_gpqa(dataset, "gpqa_diamond") if s is not None]
        logger.info(f"Loading gpqa diamond dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    if "gpqa_diamond" in dataset_names:
        pass

    if "gsm8k_train" in dataset_names:
        dataset = load_dataset("openai/gsm8k", "main", split="train", trust_remote_code=True)
        samples = [s for s in process_gsm8k(dataset, "gsm8k_train") if s is not None]
        logger.info(f"Loading gsm8k train dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    if "gsm8k_test" in dataset_names:
        dataset = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
        samples = [s for s in process_gsm8k(dataset, "gsm8k_test") if s is not None]
        logger.info(f"Loading gsm8k test dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    if "limo" in dataset_names:
        dataset = load_dataset("GAIR/LIMO", split="train", trust_remote_code=True)
        samples = [s for s in process_limo(dataset) if s is not None]
        logger.info(f"Loading limo dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    if "aime_2022" in dataset_names:
        datasets += _load_aime_dataset(2022, upsample_factor=16)

    if "aime_2022_original" in dataset_names:
        datasets += _load_aime_dataset(2022)

    if "aime_2023" in dataset_names:
        datasets += _load_aime_dataset(2023, upsample_factor=16)

    if "aime_2023_original" in dataset_names:
        datasets += _load_aime_dataset(2023)

    if "aime_2024" in dataset_names:
        datasets += _load_aime_dataset(2024, upsample_factor=16)

    if "aime_2025" in dataset_names:
        datasets += _load_aime_dataset(2025, upsample_factor=32)

    if "aime_2024_original" in dataset_names:
        datasets += _load_aime_dataset(2024)

    if "amc_2022" in dataset_names:
        # TODO: AMC 2022 is 43 problems, is that to be expected?
        datasets += _load_amc_dataset(2022, upsample_factor=16)

    if "amc_2022_original" in dataset_names:
        datasets += _load_amc_dataset(2022)

    if "amc_2023" in dataset_names:
        datasets += _load_amc_dataset(2023, upsample_factor=16)

    if "amc_2023_original" in dataset_names:
        datasets += _load_amc_dataset(2023)

    if "sometimes_success_data" in dataset_names:
        PATH = "data/sometimes_success_data/data.jsonl"
        with open(PATH, "r") as f:
            samples = [json.loads(line) for line in f]
        logger.info(f"Loading easy data dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    if "pope_512" in dataset_names:
        dataset = pd.read_parquet("/project/flame/yuxiaoq/datasets/POPE-hard-first_guide-no_guide-v2-verl/train.parquet")
        samples = [s for s in process_pope(dataset, "pope_512") if s is not None]
        logger.info(f"Loading Pope 512 dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    if "pope_mix" in dataset_names:
        ds = load_dataset("CohenQu/POPE-MIX-first_guide-no_guide-0.0-0.64-1024-verl")
        samples = [s for s in process_pope_mix(ds['train'], "pope_mix") if s is not None]
        logger.info(f"Loading Pope Mix dataset: {len(samples)} samples")
        datasets += add_ids(samples)
    
    for custom_dataset_name in [
        "POPE-MIX-first_guide-no_guide-0.0-0.125-1024-verl-train", 
        "POPE-MIX-first_guide-no_guide-0.0-0.125-1024-verl-test", 
        "olympiads-ref-base-exact-matching-train",
        "olympiads-ref-base-exact-matching-test",
    ]:
        if custom_dataset_name in dataset_names:
            custom_dataset_prefix = "-".join(custom_dataset_name.split("-")[:-1])
            split = custom_dataset_name.split("-")[-1]
            dataset = pd.read_parquet(f"tmp/datasets/{custom_dataset_prefix}/{split}.parquet")
            samples = [s for s in process_pope_local(dataset, custom_dataset_prefix) if s is not None]
            logger.info(f"Loading {custom_dataset_name} dataset: {len(samples)} samples")
            datasets += add_ids(samples)

    if "open_reasoner_zero_57k" in dataset_names:
        dataset = load_dataset(
            "json",
            data_files="https://raw.githubusercontent.com/Open-Reasoner-Zero/Open-Reasoner-Zero/refs/heads/main/data/orz_math_57k_collected.json",
            split="train",
            trust_remote_code=True,
        )
        samples = [s for s in process_open_reasoner(dataset, "open_reasoner_zero_57k") if s is not None]
        logger.info(f"Loading Open Reasoner Zero dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    if "open_reasoner_zero_extended_72k" in dataset_names:
        dataset = load_dataset(
            "json",
            data_files="https://raw.githubusercontent.com/Open-Reasoner-Zero/Open-Reasoner-Zero/refs/heads/main/data/orz_math_72k_collection_extended.json",
            split="train",
            trust_remote_code=True,
        )
        samples = [s for s in process_open_reasoner(dataset, "open_reasoner_zero_extended_72k") if s is not None]
        logger.info(f"Loading Open Reasoner Zero extended dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    if "open_reasoner_zero_hard_13k" in dataset_names:
        dataset = load_dataset(
            "json",
            data_files="https://raw.githubusercontent.com/Open-Reasoner-Zero/Open-Reasoner-Zero/refs/heads/main/data/orz_math_13k_collection_hard.json",
            split="train",
            trust_remote_code=True,
        )
        samples = [s for s in process_open_reasoner(dataset, "open_reasoner_zero_hard_13k") if s is not None]
        logger.info(f"Loading Open Reasoner Zero hard dataset: {len(samples)} samples")
        datasets += add_ids(samples)

    for dataset_name in dataset_names:
        if not isinstance(dataset_name, str):
            continue
        test_matched = re.match(r"multiplication_(\d+)_by_(\d+)_(\d+)_test", dataset_name)
        train_matched = re.match(r"multiplication(_upto)?_(\d+)_by_(\d+)_(\d+)_train", dataset_name)
        if test_matched:
            num_digits_1 = int(test_matched.group(1))
            num_digits_2 = int(test_matched.group(2))
            num_samples = int(test_matched.group(3))
            dataset = load_dataset(
                "json",
                data_files=f"data/ehsan_kamalloo/multiplication/multiplication_{num_digits_1}_by_{num_digits_2}_{num_samples}_test.jsonl",
                split="train",
            )
            samples = [
                s
                for s in process_math(dataset, f"multiplication_{num_digits_1}_by_{num_digits_2}_{num_samples}_test")
                if s is not None
            ]
            logger.info(f"Loading multiplication {num_digits_1}_by_{num_digits_2} dataset: {len(samples)} samples")
            datasets += add_ids(samples)
        elif train_matched:
            upto_prefix = train_matched.group(1) or ""
            num_digits_1 = int(train_matched.group(2))
            num_digits_2 = int(train_matched.group(3))
            num_samples = int(train_matched.group(4))
            dataset = load_dataset(
                "json",
                data_files=f"data/ehsan_kamalloo/multiplication/multiplication{upto_prefix}_{num_digits_1}_by_{num_digits_2}_{num_samples}_train.jsonl",
                split="train",
            )
            samples = [
                s
                for s in process_math(
                    dataset, f"multiplication{upto_prefix}_{num_digits_1}_by_{num_digits_2}_{num_samples}_train"
                )
                if s is not None
            ]
            logger.info(
                f"Loading multiplication {upto_prefix}_{num_digits_1}_by_{num_digits_2} dataset: {len(samples)} samples"
            )
            datasets += add_ids(samples)

    if "countdown" in dataset_names:
        dataset = load_dataset(
            "parquet", data_files="data/xiaoyin/train.parquet", trust_remote_code=True, split="train"
        )
        samples = [s for s in process_countdown(dataset) if s is not None]
        logger.info(f"Loading countdown dataset: {len(samples)} samples")
        datasets += samples

    if len(datasets) == 0:
        raise ValueError("No datasets loaded")

    return datasets


@hydra.main(
    config_path="../conf/",
    config_name="base",
    version_base="1.3.2",
)
def main(cfg: DictConfig):
    train_samples = load_datasets(cfg.train_dataset_names)
    test_samples = load_datasets(cfg.test_dataset_names)


if __name__ == "__main__":
    main()
