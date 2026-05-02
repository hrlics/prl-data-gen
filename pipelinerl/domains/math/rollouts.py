import asyncio
import time
import random

import aiohttp
import os
from omegaconf import DictConfig
from pydantic import BaseModel
from pipelinerl.rollouts import RolloutResult, BaseMetrics
from pipelinerl.world import Job
from tapeagents.core import Prompt
from tapeagents.llms.trainable import TrainableLLM

from pipelinerl.async_llm import llm_async_generate, make_training_text
from .verifier_api import (
    complete_with_z,
    generate_strategy,
    judge_alignment,
    judge_proofbench_no_gt,
    parse_schema,
    verify_answer_rpc,
    verify_proof,
)

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import numpy as np


def extract_answer_logprob(
    completion_text: str,
    logprobs: list,
    tokenizer
) -> float | None:
    import re
    match = re.search(r'\\boxed\{(\d)\}', completion_text)
    if not match:
        return None
    
    answer = match.group(1)
    answer_token_ids = set(tokenizer.encode(answer, add_special_tokens=False))
    
    # from back to front since it's more likely at the end
    for lp in reversed(logprobs):
        if lp.token_id in answer_token_ids:
            return lp.logprob
    
    return None


def extract_boxed_content(text: str) -> str | None:
    """
    Extract content from \\boxed{...} format.
    
    Args:
        text: The text potentially containing \\boxed{...}
    
    Returns:
        The content inside \\boxed{...}, or None if no boxed format found.
    """
    if not text:
        return None
    
    # Prefer a small parser over regex so we handle nested braces like:
    #   \boxed{\text{4}}
    # and also take the *last* boxed occurrence when multiple exist.
    needle = "\\boxed{"  # this matches both "\boxed{" and "\\boxed{" inputs

    search_upto = len(text)
    while True:
        start = text.rfind(needle, 0, search_upto)
        if start == -1:
            return None

        i = start + len(needle)
        depth = 1
        while i < len(text):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start + len(needle) : i].strip()
            i += 1

        # Unbalanced braces for this occurrence; try an earlier \boxed{...
        search_upto = start
        

def parse_summary(text: str):
    import re
    text = (text or "").strip()
    if not text:
        return None

    # Find all header occurrences WITHOUT DOTALL so each match is independent
    matches = list(re.finditer(
        r"PREDICTED\s+(?:REMAINING|NEXT)\s+STEPS\s*:",
        text,
        flags=re.IGNORECASE,
    ))
    if not matches:
        return None

    # Take everything after the LAST header
    result = text[matches[-1].end():].strip()
    # Remove trailing special tokens
    result = re.sub(r"<\|im_end\|>\s*$", "", result).strip()
    return result

def remove_reasoning(completion: str, reasoning_delimiters: list[str] = None) -> str:
    # Treat empty lists like None (no delimiter-based stripping).
    if not reasoning_delimiters:
        return completion
    else:
        # Split final answer from reasoning content
        for delim in reasoning_delimiters:
            if delim in completion:
                completion = completion.split(delim)[-1]
                return completion.strip()
        return ""
    

class Metrics(BaseMetrics):
    penalty: float

class RewardTable(BaseModel):
    wrong_answer_not_finished: float
    wrong_answer_finished: float
    no_answer_not_finished: float
    no_answer_finished: float
    unparsable_not_finished: float
    unparsable_finished: float
    correct_answer_not_finished: float
    correct_answer_finished: float
    buffer_tokens: int = 0 # 0 means no overlong reward shaping

def length_penalty(max_length: int, sequence_length: int, buffer_tokens: int) -> float:
    """
    Compute the overlong penalty
    """
    if sequence_length > (max_length - buffer_tokens) and sequence_length <= max_length:
        return ((max_length - buffer_tokens) - sequence_length) / buffer_tokens
    return 0.

async def generate_math_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    messages = []
    if cfg.actor.system_prompt:
        messages.append({"role": "system", "content": cfg.actor.system_prompt})
    messages.append({"role": "user", "content": cfg.actor.task_template.format(task=problem["task"])})
    prompt = Prompt(messages=messages)

    time_start = time.time()
    llm_call = await llm_async_generate(llm, prompt, session)
    latency = time.time() - time_start

    assert llm_call.output.content is not None
    generation_raw = llm_call.output.content
    reasoning_delimiters = (
        cfg.llm_grader.reasoning_delimiters
        if "reasoning_delimiters" in cfg.llm_grader
        else None
    )
    generation_final_answer = remove_reasoning(generation_raw, reasoning_delimiters=reasoning_delimiters)
    rewards = RewardTable(**dict(cfg.rewards))
    discount_factor = cfg.actor.discount_factor

    trace = make_training_text(llm, llm_call)

    # ===========================================================
    # LLM-as-A-Judge SCORING BRANCH
    # ===========================================================
    verifier_metrics: dict[str, float | int] = {}
    verifier_table_entry: dict[str, str | int | float] | None = None
    if "gemini_summary_list" in problem:
        llm_grader_cfg = cfg.get("llm_grader", None)
        wandb_table_cfg = llm_grader_cfg.get("wandb_table", None) if llm_grader_cfg is not None else None
        wandb_table_enabled = True
        if wandb_table_cfg is not None:
            wandb_table_enabled = bool(wandb_table_cfg.get("enabled", True))
        
        #schema_text = parse_schema(problem["schema"])
        generation_final_answer = parse_summary(generation_final_answer)
        verification = await verify_proof(
            problem=problem['problem'],
            # prefix=problem['prefix_summary_steps'], # bullet list prefix
            prefix=problem['prefix'], # raw prefix
            ref_solution=problem['answer'],
            schema=None,
            summaries=problem['gemini_summary_list'], # specified in ./load_dataset.py, the prediction target
            generation=generation_final_answer,
            prompt_name=getattr(cfg.llm_grader, "prompt_name", None),
            model=getattr(cfg.llm_grader, "name", None) if "/" in getattr(cfg.llm_grader, "name", "") else os.getenv("HF_ENDPOINT_REPO"),
            sampling_kwargs=getattr(cfg.llm_grader, "sampling_kwargs", None),
            log_wandb_metrics=cfg.wandb.use_wandb,
            collect_table_entry=bool(cfg.wandb.use_wandb and wandb_table_enabled),
        )

        # squash score if configured in load_datasets.py
        score = verification.score
        if problem.get('squash_score', False):
            k = 2.5
            score = (np.exp(k * score) - 1.0) / (np.exp(k) - 1.0)
        else:
            score = score

        verifier_metrics = verification.metrics
        verifier_table_entry = verification.table_entry
        # reward = (score / 7.0) * (discount_factor ** llm_call.output_length_tokens) -> this is used only for proof-based RL

        reward = score
        # Overlong penalty if configured
        overlong_penalty = 0
        # if rewards.buffer_tokens > 0:
        #     overlong_penalty = length_penalty(
        #         llm.parameters["max_tokens"],
        #         llm_call.output_length_tokens,
        #         rewards.buffer_tokens,
        #     )
        # reward += overlong_penalty
        trace.reward = reward

        metrics = Metrics(
            reward=reward,
            success=reward >= 0.8,           # treat 4/5 (normalized to >= 0.8) as success
            no_error=True,               # we don't track parse errors here
            no_answer=False,             # proof always produces output
            penalty=overlong_penalty,
        )

    # reverse KL wo entropy bonus of the answer (1/2/3/4/5) logprob
    elif "reward_logprobs" in problem and problem["reverse_kl"] is True:
        # ===========================================================
        # STANDARD VERIFIABLE-MATH BRANCH -- logprob reward --> reverse KL wo entropy bonus ((log p(c|x)) - log pi(c|x))
        # ===========================================================
        env_jobs = [Job(**job) for job in cfg.jobs if job["kind"] == "environment"]
        env_job = random.choice(env_jobs)
        assert env_job.port is not None

        answer_status = await verify_answer_rpc(
            session=session,
            host=env_job.hostname,
            port=env_job.port,
            prediction=llm_call.output.content,
            gold=problem["answer"],
            strict=True,
        )

        use_logprob_reward = False
        match (answer_status, trace.finished):
            case ("wrong", False):
                reward = rewards.wrong_answer_not_finished
            case ("wrong", True):
                reward = rewards.wrong_answer_finished
            case ("no_answer", False):
                reward = rewards.no_answer_not_finished
            case ("no_answer", True):
                reward = rewards.no_answer_finished
            case ("unparsable", False):
                reward = rewards.unparsable_not_finished
            case ("unparsable", True):
                reward = rewards.unparsable_finished
            case ("correct", False):
                reward = rewards.correct_answer_not_finished
            case ("correct", True):
                reward = rewards.correct_answer_finished
                use_logprob_reward = True
            case _:
                raise ValueError(f"Invalid answer_status/finished combination: {answer_status}/{trace.finished}")

        reward *= discount_factor ** llm_call.output_length_tokens
        overlong_penalty = 0
        if rewards.buffer_tokens > 0:
            overlong_penalty = length_penalty(
                llm.parameters["max_tokens"],
                llm_call.output_length_tokens,
                rewards.buffer_tokens,
            )
        reward += overlong_penalty

        # logprob reward

        # log p(c|x) - log pi(c|x)
        answer_logprob = extract_answer_logprob(
            generation_final_answer,
            llm_call.logprobs,
            llm.tokenizer
        )

        if answer_logprob is not None:
            gen = extract_boxed_content(generation_final_answer)
            if gen is not None and gen in ["1", "2", "3", "4", "5"]:
                c = int(gen) - 1
                reward = np.log(problem["reward_probs"][c] + 1e-12) - np.log(answer_logprob + 1e-12)
            else:
                reward = -5.0
        else:
            reward = -5.0

        trace.reward = reward

        metrics = Metrics(
            reward=reward,
            success=answer_status == "correct",
            no_error=answer_status != "unparsable",
            no_answer=answer_status == "no_answer",
            penalty=overlong_penalty,
        )
        
        # Store answer in metadata for classification metrics
        trace.metadata["answer"] = problem.get("answer", "")
        trace.metadata["prediction"] = generation_final_answer

    # ===========================================================
    # STANDARD VERIFIABLE-MATH BRANCH -- hl-gauss reward (only log p(c|x))
    # ===========================================================
    elif "reward_probs" in problem:
        env_jobs = [Job(**job) for job in cfg.jobs if job["kind"] == "environment"]
        env_job = random.choice(env_jobs)
        assert env_job.port is not None

        answer_status = await verify_answer_rpc(
            session=session,
            host=env_job.hostname,
            port=env_job.port,
            prediction=llm_call.output.content,
            gold=problem["answer"],
            strict=True,
        )

        use_hl_gauss = False
        match (answer_status, trace.finished):
            case ("wrong", False):
                reward = rewards.wrong_answer_not_finished
            case ("wrong", True):
                reward = rewards.wrong_answer_finished
                use_hl_gauss = True
            case ("no_answer", False):
                reward = rewards.no_answer_not_finished
            case ("no_answer", True):
                reward = rewards.no_answer_finished
            case ("unparsable", False):
                reward = rewards.unparsable_not_finished
            case ("unparsable", True):
                reward = rewards.unparsable_finished
            case ("correct", False):
                reward = rewards.correct_answer_not_finished
            case ("correct", True):
                reward = rewards.correct_answer_finished
                use_hl_gauss = True
            case _:
                raise ValueError(f"Invalid answer_status/finished combination: {answer_status}/{trace.finished}")

        reward *= discount_factor ** llm_call.output_length_tokens
        overlong_penalty = 0
        if rewards.buffer_tokens > 0:
            overlong_penalty = length_penalty(
                llm.parameters["max_tokens"],
                llm_call.output_length_tokens,
                rewards.buffer_tokens,
            )
        reward += overlong_penalty

        #  hl-gauss reward (only log p(c|x))
        if use_hl_gauss:
            gen = extract_boxed_content(generation_final_answer)
            if gen is not None and gen in ["1", "2", "3", "4", "5"]:
                c = int(gen) - 1
                reward = np.log(problem["reward_probs"][c] + 1e-12)
            else:
                reward = -5.0

        trace.reward = reward

        metrics = Metrics(
            reward=reward,
            success=answer_status == "correct",
            no_error=answer_status != "unparsable",
            no_answer=answer_status == "no_answer",
            penalty=overlong_penalty,
        )
        
        # Store answer in metadata for classification metrics
        trace.metadata["answer"] = problem.get("answer", "")
        trace.metadata["prediction"] = generation_final_answer

    # ===========================================================
    # STANDARD VERIFIABLE-MATH BRANCH
    # ===========================================================
    else:
        env_jobs = [Job(**job) for job in cfg.jobs if job["kind"] == "environment"]
        env_job = random.choice(env_jobs)
        assert env_job.port is not None

        answer_status = await verify_answer_rpc(
            session=session,
            host=env_job.hostname,
            port=env_job.port,
            prediction=llm_call.output.content,
            gold=problem["answer"],
            strict=True,
        )

        match (answer_status, trace.finished):
            case ("wrong", False):
                reward = rewards.wrong_answer_not_finished
            case ("wrong", True):
                reward = rewards.wrong_answer_finished
            case ("no_answer", False):
                reward = rewards.no_answer_not_finished
            case ("no_answer", True):
                reward = rewards.no_answer_finished
            case ("unparsable", False):
                reward = rewards.unparsable_not_finished
            case ("unparsable", True):
                reward = rewards.unparsable_finished
            case ("correct", False):
                reward = rewards.correct_answer_not_finished
            case ("correct", True):
                reward = rewards.correct_answer_finished
            case _:
                raise ValueError(f"Invalid answer_status/finished combination: {answer_status}/{trace.finished}")

        reward *= discount_factor ** llm_call.output_length_tokens
        overlong_penalty = 0
        if rewards.buffer_tokens > 0:
            overlong_penalty = length_penalty(
                llm.parameters["max_tokens"],
                llm_call.output_length_tokens,
                rewards.buffer_tokens,
            )
        reward += overlong_penalty
        trace.reward = reward

        metrics = Metrics(
            reward=reward,
            success=answer_status == "correct",
            no_error=answer_status != "unparsable",
            no_answer=answer_status == "no_answer",
            penalty=overlong_penalty,
        )
        
        # Store answer in metadata for classification metrics
        trace.metadata["answer"] = problem.get("answer", "")
        trace.metadata["prediction"] = generation_final_answer

    # ===========================================================
    # COMMON RETURN BLOCK
    # ===========================================================
    return RolloutResult(
        training_texts=[trace],
        metrics=metrics,
        latency=latency,
        dataset_name=problem.get("dataset"),
        verifier_metrics=verifier_metrics,
        verifier_table_entry=verifier_table_entry,
    )


def _resolve_grader_model(cfg: DictConfig) -> str:
    name = getattr(cfg.llm_grader, "name", None)
    if name and "/" in str(name):
        return str(name)
    return os.getenv("HF_ENDPOINT_REPO") or str(name or "")


def _merge_runtime_metrics(
    aggregate: dict[str, float | int],
    new: dict[str, float | int],
    *,
    prefix: str,
) -> None:
    """Tag judge runtime metrics with their stage so wandb scalars don't collide."""
    if not new:
        return
    for k, v in new.items():
        if k.startswith("verifier/runtime/") or k.startswith("verifier/failures/"):
            aggregate[f"{k}/{prefix}"] = v
        elif k.startswith("verifier/rollouts/"):
            aggregate[f"{k}/{prefix}"] = v
        else:
            aggregate[k] = v


async def generate_data_generator_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    """RL rollout for the data-generator setup.

    Policy: Qwen3-4B-Thinking-2507 sees (problem, prefix, GT completion) and
    produces strategy bullets `z` (after `PREDICTED REMAINING STEPS:`).
    Reward = Fittability * Performance after conditioning, both scored by gpt-oss.
    """
    messages = []
    if cfg.actor.system_prompt:
        messages.append({"role": "system", "content": cfg.actor.system_prompt})
    messages.append({"role": "user", "content": cfg.actor.task_template.format(task=problem["task"])})
    prompt = Prompt(messages=messages)

    time_start = time.time()
    llm_call = await llm_async_generate(llm, prompt, session)
    latency = time.time() - time_start

    assert llm_call.output.content is not None
    generation_raw = llm_call.output.content
    reasoning_delimiters = (
        cfg.llm_grader.reasoning_delimiters
        if "reasoning_delimiters" in cfg.llm_grader
        else None
    )
    generation_final = remove_reasoning(generation_raw, reasoning_delimiters=reasoning_delimiters)
    z = parse_summary(generation_final)

    rewards = RewardTable(**dict(cfg.rewards))
    discount_factor = cfg.actor.discount_factor

    trace = make_training_text(llm, llm_call)

    data_gen_cfg = cfg.get("data_generator", {})
    grader_model = _resolve_grader_model(cfg)
    judge_sampling_kwargs = getattr(cfg.llm_grader, "sampling_kwargs", None)
    complete_sampling_kwargs = (
        getattr(cfg.llm_grader, "complete_sampling_kwargs", None)
        or judge_sampling_kwargs
    )

    verifier_metrics: dict[str, float | int] = {}
    verifier_table_entry: dict[str, str | int | float] | None = None

    # If the policy didn't emit a parseable strategy, short-circuit with reward=0.
    if not z:
        reward = 0.0
        trace.reward = reward
        metrics = Metrics(
            reward=reward,
            success=False,
            no_error=False,
            no_answer=True,
            penalty=0.0,
        )
        verifier_table_entry = {
            "prompt": "(policy did not produce parseable PREDICTED REMAINING STEPS)",
            "reasoning": generation_raw[:4000],
            "output_text": generation_final[:4000],
            "score": "fitt=0.000 perf=0.000 final=0.000 (no z)",
        }
        return RolloutResult(
            training_texts=[trace],
            metrics=metrics,
            latency=latency,
            dataset_name=problem.get("dataset"),
            verifier_metrics=verifier_metrics,
            verifier_table_entry=verifier_table_entry,
        )

    problem_text = problem["problem"]
    prefix_text = problem["prefix"]
    marking_scheme = problem.get("marking_scheme", "")

    async def _fittability_chain() -> tuple:
        a_call = await generate_strategy(
            problem=problem_text,
            prefix=prefix_text,
            prompt_name=data_gen_cfg.get("predict_z_prompt", "predict_z_v1"),
            model=grader_model,
            sampling_kwargs=judge_sampling_kwargs,
        )
        z_prime = parse_summary(a_call.output_text) or a_call.output_text.strip()
        b_call = await judge_alignment(
            problem=problem_text,
            prefix=prefix_text,
            z=z,
            z_prime=z_prime,
            prompt_name=data_gen_cfg.get("align_judge_prompt", "align_z_judge_v1"),
            model=grader_model,
            sampling_kwargs=judge_sampling_kwargs,
        )
        return a_call, z_prime, b_call

    async def _performance_chain() -> tuple:
        c1_call = await complete_with_z(
            problem=problem_text,
            prefix=prefix_text,
            z=z,
            prompt_name=data_gen_cfg.get("complete_z_prompt", "complete_with_z_v1"),
            model=grader_model,
            sampling_kwargs=complete_sampling_kwargs,
        )
        c2_call = await judge_proofbench_no_gt(
            problem=problem_text,
            marking_scheme=marking_scheme,
            prefix=prefix_text,
            completion=c1_call.output_text,
            prompt_name=data_gen_cfg.get("perform_judge_prompt", "perform_proofbench_no_gt_v1"),
            model=grader_model,
            sampling_kwargs=judge_sampling_kwargs,
        )
        return c1_call, c2_call

    (a_call, z_prime, b_call), (c1_call, c2_call) = await asyncio.gather(
        _fittability_chain(), _performance_chain()
    )

    fittability = b_call.score
    performance = c2_call.score
    final_reward = fittability * performance

    trace.reward = final_reward

    _merge_runtime_metrics(verifier_metrics, a_call.metrics, prefix="A_predict_z")
    _merge_runtime_metrics(verifier_metrics, b_call.metrics, prefix="B_align")
    _merge_runtime_metrics(verifier_metrics, c1_call.metrics, prefix="C1_complete")
    _merge_runtime_metrics(verifier_metrics, c2_call.metrics, prefix="C2_grade")
    verifier_metrics["verifier/scores/fittability"] = float(fittability)
    verifier_metrics["verifier/scores/performance"] = float(performance)
    verifier_metrics["verifier/scores/final"] = float(final_reward)

    def _section(label, text):
        return f"\n\n=== {label} ===\n{text or ''}"

    verifier_table_entry = {
        "prompt": (
            _section("A predict_z", a_call.prompt_text)
            + _section("B align", b_call.prompt_text)
            + _section("C1 complete", c1_call.prompt_text)
            + _section("C2 grade", c2_call.prompt_text)
        ),
        "reasoning": (
            _section("A predict_z", a_call.reasoning_text)
            + _section("B align", b_call.reasoning_text)
            + _section("C1 complete", c1_call.reasoning_text)
            + _section("C2 grade", c2_call.reasoning_text)
        ),
        "output_text": (
            _section("policy z", z)
            + _section("A z_prime", z_prime)
            + _section("B align raw", b_call.output_text)
            + _section("C1 completion", c1_call.output_text)
            + _section("C2 grade raw", c2_call.output_text)
        ),
        "score": f"fitt={fittability:.3f} perf={performance:.3f} final={final_reward:.3f}",
    }

    metrics = Metrics(
        reward=final_reward,
        success=final_reward >= 0.5,
        no_error=True,
        no_answer=False,
        penalty=0.0,
    )

    return RolloutResult(
        training_texts=[trace],
        metrics=metrics,
        latency=latency,
        dataset_name=problem.get("dataset"),
        verifier_metrics=verifier_metrics,
        verifier_table_entry=verifier_table_entry,
    )
