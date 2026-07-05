# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# [MORALGYM PATCH 6] SDPO (Self-Distilled Policy Optimization) trainer.
#
# Port of Verl's SDPO onto the NeMo-RL GRPO scaffolding. Following the
# NeMo-RL convention (see distillation.py), this module clones grpo.py's
# setup()/grpo_train() rather than modifying them, so the GRPO baseline
# stays byte-identical. Shared helpers are imported from grpo.
#
# Verl ground truth:
#   - reprompt builder: SDPO/verl/trainer/ppo/ray_trainer.py:611-796
#   - loss:             SDPO/verl/trainer/ppo/core_algos.py:1085-1188
#     (ported as SDPOLossFn in nemo_rl/algorithms/loss_functions.py)
#   - teacher EMA:      SDPO/verl/workers/actor/dp_actor.py:132-151
#     (updated ONCE per policy.train(), after the full minibatch loop)
#
# Alignment note (differs from Verl mechanically, not semantically): Verl
# left-pads prompts so the response sits at a fixed offset in both the
# student and reprompted teacher contexts. NeMo-RL right-pads with variable
# per-row lengths, so the builder emits per-row `teacher_offsets` =
# len(teacher first message) - len(student first message); the DTensor V2
# worker gathers teacher log-probs back to student positions.
import os
import re
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional, TypeVar, cast

import numpy as np
import ray
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import (
    GRPOSaveState,
    MasterConfig,
    _default_grpo_save_state,
    _should_use_async_rollouts,
    _should_use_nemo_gym,
    compute_per_step_advantages_for_batch,
    dynamic_sampling,
    normalize_advantages_with_epsilon,
    refit_policy_generation,
    scale_rewards,
    validate,
)
from nemo_rl.algorithms.loss_functions import (
    SDPOLossConfig,
    SDPOLossDataDict,
    SDPOLossFn,
)
from nemo_rl.algorithms.reward_functions import apply_reward_shaping
from nemo_rl.algorithms.utils import (
    calculate_baseline_and_std_per_prompt,
    log_generation_metrics_to_wandb,
    print_performance_metrics,
    set_seed,
)
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
    run_multi_turn_rollout,
)
from nemo_rl.models.generation.interfaces import GenerationInterface
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.checkpoint import CheckpointManager
from nemo_rl.utils.logger import Logger
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import TimeoutChecker, Timer

TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)

# ===============================================================================
# Configuration
# ===============================================================================

# Verl SelfDistillationConfig defaults, verbatim
# (SDPO/verl/workers/config/actor.py:39-92).
SELF_DISTILLATION_DEFAULTS: dict[str, Any] = {
    "full_logit_distillation": True,
    "alpha": 0.0,
    "success_reward_threshold": 1.0,
    "teacher_regularization": "ema",
    "teacher_update_rate": 0.05,
    "distillation_topk": None,
    "distillation_add_tail": True,
    "max_reprompt_len": 10240,
    "reprompt_truncation": "right",
    "dont_reprompt_on_self_success": False,
    "remove_thinking_from_demonstration": False,
    "is_clip": None,
    "reprompt_template": (
        "{prompt}{solution}{feedback}\n\nCorrectly solve the original question.\n"
    ),
    "solution_template": "\nCorrect solution:\n\n{successful_previous_attempt}\n\n",
    "feedback_template": (
        "\nThe following is feedback from your unsuccessful earlier attempt:\n\n"
        "{feedback_raw}\n\n"
    ),
    "include_environment_feedback": False,
    "environment_feedback_only_without_solution": False,
}


def resolve_self_distillation_config(master_config: MasterConfig) -> dict[str, Any]:
    """Merge `policy.self_distillation` over the Verl defaults and validate.

    Mirrors Verl's SelfDistillationConfig.__post_init__ checks plus the
    Phase 1 restrictions of this port.
    """
    user_cfg = master_config["policy"].get("self_distillation") or {}
    unknown = set(user_cfg) - set(SELF_DISTILLATION_DEFAULTS)
    if unknown:
        raise ValueError(
            f"Unknown keys in policy.self_distillation: {sorted(unknown)}. "
            f"Valid keys: {sorted(SELF_DISTILLATION_DEFAULTS)}"
        )
    cfg = {**SELF_DISTILLATION_DEFAULTS, **user_cfg}

    if not 0.0 <= cfg["alpha"] <= 1.0:
        raise ValueError(
            f"policy.self_distillation.alpha must be in [0, 1], got {cfg['alpha']}"
        )
    if cfg["teacher_regularization"] not in ("ema", "trust-region"):
        raise ValueError(
            "policy.self_distillation.teacher_regularization must be 'ema' or "
            f"'trust-region', got {cfg['teacher_regularization']!r}"
        )
    if cfg["reprompt_truncation"] not in ("right", "error"):
        raise ValueError(
            "policy.self_distillation.reprompt_truncation must be 'right' or "
            f"'error', got {cfg['reprompt_truncation']!r}"
        )
    if cfg["include_environment_feedback"]:
        raise NotImplementedError(
            "policy.self_distillation.include_environment_feedback is Phase 2 "
            "(feedback synthesis); set it to false."
        )
    return cfg


# ===============================================================================
# Reprompt batch builder (port of Verl ray_trainer.py:611-796)
# ===============================================================================


def _collect_solutions_by_uid(
    uids: list[Any],
    rewards: torch.Tensor,
    success_reward_threshold: float,
) -> dict[Any, list[int]]:
    """Bucket successful sample indices by prompt UID.

    Port of Verl `_collect_solutions_by_uid` (ray_trainer.py:636-643). In
    NeMo-RL the UID is the datum `idx`, which `repeat_interleave` copies to
    every generation in a group.
    """
    success_by_uid: dict[Any, list[int]] = defaultdict(list)
    for i, uid in enumerate(uids):
        if float(rewards[i]) >= success_reward_threshold:
            success_by_uid[uid].append(i)
    return success_by_uid


def _remove_thinking_trace(text: str) -> str:
    """Remove <think>...</think> tags and their content from text."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


def _get_solution(
    idx: int,
    success_by_uid: dict[Any, list[int]],
    uids: list[Any],
    response_texts: list[str],
    dont_reprompt_on_self_success: bool = False,
    remove_thinking_from_demonstration: bool = False,
) -> Optional[str]:
    """Pick a successful sibling response as the demonstration for sample idx.

    Port of Verl `_get_solution` (ray_trainer.py:650-670).
    """
    uid = uids[idx]
    solution_idxs = success_by_uid[uid]
    if dont_reprompt_on_self_success:
        solution_idxs = [j for j in solution_idxs if j != idx]
    if len(solution_idxs) == 0:
        return None
    # Taking the first successful demonstration effectively selects a random one.
    solution_idx = solution_idxs[0]
    solution_str = response_texts[solution_idx]
    if remove_thinking_from_demonstration:
        solution_str = _remove_thinking_trace(solution_str)
    return solution_str


def build_reprompt_batch(
    repeated_batch: BatchedDataDict[DatumSpec],
    tokenizer: TokenizerType,
    sd_cfg: dict[str, Any],
    pad_token_id: int,
    make_sequence_length_divisible_by: int = 1,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Build the teacher (reprompted) tensors for one training batch.

    Port of Verl `_maybe_build_self_distillation_batch`
    (ray_trainer.py:672-796), adapted to NeMo-RL's right-padded multi-turn
    message logs:

      - The teacher context is the student context with the FIRST user
        message replaced by the reprompted message (original prompt +
        successful sibling demonstration). All later messages — including
        the student's own response tokens — are byte-identical, shifted
        right by a per-row delta.
      - Rows without a demonstration reuse the student's exact first-message
        tokens (delta 0) and get `self_distillation_mask = 0`; the teacher
        forward still runs on them but their loss contribution is zeroed,
        matching Verl.

    Requires the un-templated prompt text in `extra_env_info["raw_prompt"]`
    for any row that gets a demonstration (Verl reads the equivalent from
    `raw_prompt` in its non-tensor batch).

    Returns:
        tensors: teacher_input_ids [B, S_t], teacher_attention_mask [B, S_t],
            teacher_offsets [B], self_distillation_mask [B]
        metrics: self_distillation/* fractions (Verl names)
    """
    message_logs = repeated_batch["message_log"]
    batch_size = len(message_logs)
    rewards = repeated_batch["total_reward"]
    uids = [int(i) for i in repeated_batch["idx"]]
    response_texts = [
        "".join(m["content"] for m in log if m["role"] == "assistant")
        for log in message_logs
    ]

    success_by_uid = _collect_solutions_by_uid(
        uids, rewards, sd_cfg["success_reward_threshold"]
    )
    solution_strs = [
        _get_solution(
            i,
            success_by_uid,
            uids,
            response_texts,
            sd_cfg["dont_reprompt_on_self_success"],
            sd_cfg["remove_thinking_from_demonstration"],
        )
        for i in range(batch_size)
    ]

    teacher_rows: list[torch.Tensor] = []
    teacher_offsets: list[int] = []
    sd_mask: list[float] = []
    for i, log in enumerate(message_logs):
        student_first = log[0]["token_ids"]
        rest = [m["token_ids"] for m in log[1:]]

        teacher_first: Optional[torch.Tensor] = None
        if solution_strs[i] is not None:
            # extra_env_info starts as the datum's dict, but multi-turn envs
            # replace it with their own metadata object on each step — accept
            # a "raw_prompt" key or attribute on either.
            extra = repeated_batch["extra_env_info"][i]
            if isinstance(extra, dict):
                raw_prompt = extra.get("raw_prompt") or None
            else:
                raw_prompt = getattr(extra, "raw_prompt", None) or None
            if raw_prompt is None:
                raise ValueError(
                    "SDPO reprompting requires the un-templated prompt text "
                    "at extra_env_info['raw_prompt'] (the first user message "
                    "content is already chat-templated and cannot be wrapped "
                    "in the reprompt template again). Stash it in the datum "
                    "generator; if the environment replaces extra_env_info "
                    "with its own metadata each step, carry it through there "
                    "too (raw_prompt key or attribute)."
                )
            solution_section = sd_cfg["solution_template"].format(
                successful_previous_attempt=solution_strs[i]
            )
            reprompt_text = sd_cfg["reprompt_template"].format(
                prompt=raw_prompt, solution=solution_section, feedback=""
            )
            # Template + tokenize exactly like the student's first message so
            # the shared suffix tokens line up.
            templated = tokenizer.apply_chat_template(
                [{"role": "user", "content": reprompt_text}],
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=False,
            )
            candidate = tokenizer(
                templated, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0].to(student_first.dtype)
            if candidate.shape[0] > sd_cfg["max_reprompt_len"]:
                if sd_cfg["reprompt_truncation"] == "error":
                    raise ValueError(
                        f"Reprompted message for sample {i} has "
                        f"{candidate.shape[0]} tokens > max_reprompt_len="
                        f"{sd_cfg['max_reprompt_len']} and reprompt_truncation"
                        "='error'."
                    )
                candidate = candidate[: sd_cfg["max_reprompt_len"]]
            if candidate.shape[0] >= student_first.shape[0]:
                teacher_first = candidate
            # else: truncation made the reprompt shorter than the original
            # first message, which would break the delta >= 0 alignment
            # invariant — treat as "no demonstration" for this row.

        if teacher_first is None:
            teacher_first = student_first
            sd_mask.append(0.0)
        else:
            sd_mask.append(1.0)
        teacher_offsets.append(int(teacher_first.shape[0] - student_first.shape[0]))
        teacher_rows.append(
            torch.cat([teacher_first, *rest]) if rest else teacher_first
        )

    max_len = max(row.shape[0] for row in teacher_rows)
    d = make_sequence_length_divisible_by
    if d > 1:
        max_len = ((max_len + d - 1) // d) * d
    teacher_input_ids = torch.full(
        (batch_size, max_len), pad_token_id, dtype=torch.long
    )
    teacher_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for i, row in enumerate(teacher_rows):
        teacher_input_ids[i, : row.shape[0]] = row
        teacher_attention_mask[i, : row.shape[0]] = True

    uid_set = set(uids)
    num_with_solution = sum(s is not None for s in solution_strs)
    metrics = {
        "self_distillation/success_group_fraction": sum(
            1 for uid in uid_set if success_by_uid[uid]
        )
        / len(uid_set),
        "self_distillation/success_sample_fraction": num_with_solution / batch_size,
        # Phase 1: environment feedback disabled (Verl parity when
        # include_environment_feedback=False).
        "self_distillation/feedback_available_fraction": 0.0,
        "self_distillation/feedback_used_fraction": 0.0,
        "self_distillation/reprompt_sample_fraction": float(np.mean(sd_mask)),
        "self_distillation/mean_teacher_offset": float(np.mean(teacher_offsets)),
        "self_distillation/teacher_seq_len": float(max_len),
    }
    tensors = {
        "teacher_input_ids": teacher_input_ids,
        "teacher_attention_mask": teacher_attention_mask,
        "teacher_offsets": torch.tensor(teacher_offsets, dtype=torch.long),
        "self_distillation_mask": torch.tensor(sd_mask, dtype=torch.float32),
    }
    return tensors, metrics


# ===============================================================================
# Setup & Initialization (clone of grpo.setup with teacher + SDPO loss)
# ===============================================================================


def setup(
    master_config: MasterConfig,
    tokenizer: TokenizerType,
    dataset: AllTaskProcessedDataset,
    val_dataset: Optional[AllTaskProcessedDataset],
    processor: Optional[AutoProcessor] = None,
) -> tuple[
    ColocatablePolicyInterface,
    Optional[GenerationInterface],
    tuple[RayVirtualCluster, RayVirtualCluster],
    StatefulDataLoader,
    Optional[StatefulDataLoader],
    SDPOLossFn,
    Logger,
    CheckpointManager,
    GRPOSaveState,
    MasterConfig,
]:
    """Main entry point for running the SDPO algorithm.

    Clone of grpo.setup() with three changes: (1) Phase 1 compatibility
    asserts, (2) the Policy is constructed with the EMA teacher enabled,
    (3) the loss is SDPOLossFn built from policy.self_distillation.
    """
    setup_start_time = time.perf_counter()

    policy_config = master_config["policy"]
    generation_config = master_config["policy"]["generation"]
    env_configs = master_config["env"]
    grpo_config = master_config["grpo"]
    data_config = master_config["data"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]

    assert generation_config is not None, (
        "A generation config in the PolicyConfig is required for SDPO"
    )

    sd_cfg = resolve_self_distillation_config(master_config)

    # --- SDPO Phase 1 compatibility gates (fail fast, before Ray spin-up) ---
    assert policy_config.get("dtensor_cfg", {}).get("enabled", False) and (
        policy_config.get("dtensor_cfg", {}).get("_v2", False)
    ), "SDPO requires the DTensor V2 worker (policy.dtensor_cfg.enabled + _v2)"
    assert not policy_config.get("megatron_cfg", {}).get("enabled", False), (
        "SDPO Phase 1 does not support the Megatron policy backend"
    )
    assert not policy_config.get("dynamic_batching", {}).get("enabled", False), (
        "SDPO Phase 1 is not compatible with dynamic batching (its microbatch "
        "iterator slices the sequence dim, desyncing student/teacher layouts)"
    )
    assert not policy_config.get("sequence_packing", {}).get("enabled", False), (
        "SDPO Phase 1 is not compatible with sequence packing"
    )
    assert generation_config["backend"] == "vllm", (
        "SDPO Phase 1 requires the vLLM generation backend"
    )

    set_seed(grpo_config["seed"])

    # ==========================
    #         Logger
    # ==========================
    logger = Logger(logger_config)
    logger.log_hyperparams(master_config)

    # ==========================
    #      Checkpointing
    # ==========================
    checkpointer = CheckpointManager(master_config["checkpointing"])
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    grpo_save_state: Optional[GRPOSaveState] = cast(
        Optional[GRPOSaveState], checkpointer.load_training_info(last_checkpoint_path)
    )
    if grpo_save_state is None:
        grpo_save_state = _default_grpo_save_state()
    elif grpo_save_state.get("current_step", 0) > 0:
        # Both Verl and this port re-derive the teacher from the loaded actor
        # weights on resume (the EMA state itself is not checkpointed).
        print(
            "  ⚠ SDPO resume: teacher re-initialized from checkpointed actor "
            "weights (EMA history is not persisted)",
            flush=True,
        )

    # ==========================
    #           Data
    # ==========================
    batch_multiplier = grpo_config["batch_multiplier"]
    dataloader_batch_size = grpo_config["num_prompts_per_step"]
    if not grpo_config["use_dynamic_sampling"]:
        assert batch_multiplier == 1, (
            "batch_multiplier>1 can only be used if use_dynamic_sampling=True"
        )
    else:
        dataloader_batch_size = int(dataloader_batch_size * batch_multiplier)

    dataloader = StatefulDataLoader(
        dataset,
        batch_size=dataloader_batch_size,
        shuffle=data_config["shuffle"],
        collate_fn=rl_collate_fn,
        drop_last=True,
        num_workers=data_config["num_workers"],
    )
    if last_checkpoint_path is not None:
        dataloader_state_dict = torch.load(
            os.path.join(last_checkpoint_path, "train_dataloader.pt")
        )
        dataloader.load_state_dict(dataloader_state_dict)

    print(f"  ✓ Training dataloader loaded with {len(dataset)} samples", flush=True)

    val_dataloader: Optional[StatefulDataLoader] = None
    if grpo_config["val_period"] > 0 or grpo_config["val_at_start"]:
        assert val_dataset is not None, (
            "Validation dataset is required if validation is enabled"
        )
        val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=grpo_config["val_batch_size"],
            shuffle=False,
            collate_fn=rl_collate_fn,
            num_workers=data_config["num_workers"],
        )
        print(
            f"  ✓ Validation dataloader loaded with {len(val_dataset)} samples",
            flush=True,
        )

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...", flush=True)
    colocated_inference = generation_config["colocated"]["enabled"]
    reward_model_enabled = (
        "env_name" in data_config and data_config["env_name"] == "reward_model"
    )

    total_nodes = cluster_config["num_nodes"]
    if reward_model_enabled:
        rm_resource = env_configs["reward_model"]["resources"]
        rm_nodes = rm_resource["num_nodes"]
        rm_gpus_per_node = rm_resource["gpus_per_node"]
    else:
        rm_nodes = 0
        rm_gpus_per_node = 0

    if total_nodes == 1:
        policy_nodes = total_nodes
    else:
        policy_nodes = total_nodes - rm_nodes
        assert policy_nodes > 0, (
            "policy_nodes must be > 0, but got "
            f"policy_nodes:{policy_nodes} + rm_nodes:{rm_nodes} = total_nodes:{total_nodes}"
        )

    if colocated_inference:
        if total_nodes == 1:
            policy_gpus_per_node = cluster_config["gpus_per_node"] - rm_gpus_per_node
            assert policy_gpus_per_node > 0, (
                "policy.generation.colocated.resources.gpus_per_node must be > 0 "
                "when cluster.num_nodes = 1, "
                f"but got {policy_gpus_per_node}."
            )
        else:
            policy_gpus_per_node = cluster_config["gpus_per_node"]

        cluster = RayVirtualCluster(
            name="sdpo_policy_cluster",
            bundle_ct_per_node_list=[policy_gpus_per_node] * policy_nodes,
            use_gpus=True,
            num_gpus_per_node=policy_gpus_per_node,
            max_colocated_worker_groups=2,
        )
        train_cluster = cluster
        inference_cluster = cluster
        print(
            f"  ✓ Ray cluster for policy initialized with {policy_nodes} nodes",
            flush=True,
        )
    else:
        train_gpus_per_node = cluster_config["gpus_per_node"]
        train_nodes = policy_nodes

        inference_resources = generation_config["colocated"]["resources"]
        inference_gpus_per_node = inference_resources["gpus_per_node"]
        inference_nodes = inference_resources["num_nodes"]

        if policy_nodes == 1:
            assert (
                inference_gpus_per_node is not None and inference_gpus_per_node > 0
            ), (
                "policy.generation.colocated.resources.gpus_per_node must be explicitly set to a value > 0 "
                "when policy_nodes = 1 and inference is non-colocated, "
                f"but got {inference_gpus_per_node}."
            )
            assert inference_nodes is None or inference_nodes == 1, (
                "policy.generation.colocated.resources.num_nodes must be 1 or set to null "
                "when policy_nodes = 1 and inference is non-colocated, "
                f"but got {inference_nodes}."
            )

            inference_nodes = 1
            reward_gpus_to_subtract = (
                rm_gpus_per_node if total_nodes == 1 and reward_model_enabled else 0
            )
            train_gpus_per_node -= inference_gpus_per_node + reward_gpus_to_subtract
            assert train_gpus_per_node > 0, (
                "No enough GPUs for training, "
                f"train_gpus_per_node:{train_gpus_per_node} = cluster_config['gpus_per_node']:{cluster_config['gpus_per_node']} - inference_gpus_per_node:{inference_gpus_per_node}"
            )
        else:
            assert inference_nodes > 0, (
                "policy.generation.colocated.resources.num_nodes must be > 0 "
                "when cluster.num_nodes > 1 and inference is non-colocated, "
                f"but got {inference_nodes}."
            )
            assert (
                inference_gpus_per_node is not None
                and inference_gpus_per_node == cluster_config["gpus_per_node"]
            ), (
                "policy.generation.colocated.resources.gpus_per_node must be explicitly set and equal to cluster.gpus_per_node "
                "when cluster.num_nodes > 1 and inference is non-colocated, "
                f"but got inference_gpus_per_node={inference_gpus_per_node}, cluster.gpus_per_node={cluster_config['gpus_per_node']}."
            )
            train_nodes -= inference_nodes

        train_cluster = RayVirtualCluster(
            name="sdpo_train_cluster",
            bundle_ct_per_node_list=[train_gpus_per_node] * train_nodes,
            use_gpus=True,
            num_gpus_per_node=train_gpus_per_node,
            max_colocated_worker_groups=1,
        )
        print(
            f"  ✓ Ray train cluster initialized with {train_nodes} nodes with {train_gpus_per_node} GPUs per node",
            flush=True,
        )

        inference_cluster = RayVirtualCluster(
            name="sdpo_inference_cluster",
            bundle_ct_per_node_list=[inference_gpus_per_node] * inference_nodes,
            use_gpus=True,
            num_gpus_per_node=inference_gpus_per_node,
            max_colocated_worker_groups=1,
        )
        print(
            f"  ✓ Ray inference cluster initialized with {inference_nodes} nodes with {inference_gpus_per_node} GPUs per node",
            flush=True,
        )

    # ==========================
    #   Training and Inference
    # ==========================
    print("\n▶ Setting up model and training...", flush=True)

    generation_config["model_name"] = policy_config["model_name"]  # Needed for vLLM

    worker_init_timing_metrics: dict[str, Any] = {}

    if last_checkpoint_path:
        weights_path = Path(last_checkpoint_path) / "policy" / "weights"
        optimizer_path = Path(last_checkpoint_path) / "policy" / "optimizer"
    else:
        weights_path = None
        optimizer_path = None

    def init_policy():
        """Initialize policy training workers (with the SDPO EMA teacher)."""
        t0 = time.perf_counter()
        p = Policy(
            cluster=train_cluster,
            config=policy_config,
            tokenizer=tokenizer,
            processor=processor,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
            init_optimizer=True,
            init_teacher_model=True,
            teacher_regularization=sd_cfg["teacher_regularization"],
            teacher_update_rate=sd_cfg["teacher_update_rate"],
        )
        return p, time.perf_counter() - t0

    def init_vllm():
        """Initialize vLLM generation workers."""
        t0 = time.perf_counter()
        pg = VllmGeneration(cluster=inference_cluster, config=generation_config)
        pg.finish_generation()
        return pg, time.perf_counter() - t0

    generation_config = cast(VllmConfig, generation_config)
    assert generation_config["vllm_cfg"]["precision"] != "fp8", (
        "SDPO Phase 1 does not support fp8 generation"
    )

    ## make vllm hf overrides match the training policy
    generation_config["vllm_cfg"]["hf_overrides"] = policy_config.get(
        "hf_config_overrides", {}
    )

    use_parallel_init = not colocated_inference

    if use_parallel_init:
        print(
            "  ⚡ Using parallel worker initialization (non-colocated mode)",
            flush=True,
        )
        parallel_start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as executor:
            vllm_future = executor.submit(init_vllm)
            policy_future = executor.submit(init_policy)
            policy_generation, vllm_time = vllm_future.result()
            policy, policy_time = policy_future.result()
        parallel_wall_time = time.perf_counter() - parallel_start_time

        worker_init_timing_metrics["vllm_init_time_s"] = vllm_time
        worker_init_timing_metrics["policy_init_time_s"] = policy_time
        worker_init_timing_metrics["parallel_wall_time_s"] = parallel_wall_time
        worker_init_timing_metrics["parallel_init_enabled"] = True
    else:
        print(
            "  ⚙️  Using sequential worker initialization (colocated mode)",
            flush=True,
        )
        policy_generation, vllm_time = init_vllm()
        worker_init_timing_metrics["vllm_init_time_s"] = vllm_time

        policy, policy_time = init_policy()
        worker_init_timing_metrics["policy_init_time_s"] = policy_time
        worker_init_timing_metrics["parallel_init_enabled"] = 0.0

    print(
        f"  ✓ Using vLLM backend for generation with {policy_config['model_name']}",
        flush=True,
    )

    worker_init_complete_time = time.perf_counter() - setup_start_time

    policy.print_node_ip_and_gpu_id()

    if not colocated_inference:
        t0 = time.perf_counter()
        ip, port = train_cluster.get_master_address_and_port()
        print(f"Using ip: {ip}, port: {port} for collective communication", flush=True)
        train_world_size = train_cluster.world_size()
        inference_world_size = inference_nodes * inference_gpus_per_node
        world_size = train_world_size + inference_world_size
        futures_train = policy.init_collective(
            ip, port, world_size, train_world_size=train_world_size
        )
        futures_inference = policy_generation.init_collective(
            ip, port, world_size, train_world_size=train_world_size
        )  # type: ignore
        ray.get(futures_train + futures_inference)
        worker_init_timing_metrics["collective_init_time_s"] = time.perf_counter() - t0

    # prepare refit info
    state_dict_info = policy.prepare_refit_info()
    if policy_generation is not None:
        policy_generation.prepare_refit_info(state_dict_info)

    loss_fn = SDPOLossFn(
        cast(
            SDPOLossConfig,
            {
                "full_logit_distillation": sd_cfg["full_logit_distillation"],
                "alpha": sd_cfg["alpha"],
                "distillation_topk": sd_cfg["distillation_topk"],
                "distillation_add_tail": sd_cfg["distillation_add_tail"],
                "is_clip": sd_cfg["is_clip"],
            },
        )
    )

    total_setup_time = time.perf_counter() - setup_start_time
    worker_init_timing_metrics["total_setup_time_s"] = total_setup_time

    if worker_init_timing_metrics:
        print("\n▶ Worker Initialization Timing:")

        vllm_time = worker_init_timing_metrics.get("vllm_init_time_s", 0)
        policy_time = worker_init_timing_metrics.get("policy_init_time_s", 0)
        total_setup = worker_init_timing_metrics.get("total_setup_time_s", 0)

        if vllm_time:
            print(f"  vLLM init: {vllm_time:.1f}s")
        if policy_time:
            print(f"  Policy init: {policy_time:.1f}s")

        other_time = total_setup - worker_init_complete_time
        worker_init_timing_metrics["other_setup_time_s"] = other_time
        print(f"  Other setup: {other_time:.1f}s")
        print(f"  Total setup: {total_setup:.1f}s")

        logger.log_metrics(worker_init_timing_metrics, step=0, prefix="timing/setup")

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE (SDPO)")
    print(f"  Total setup time: {total_setup_time:.1f}s")
    print("=" * 60 + "\n", flush=True)

    return (
        policy,
        policy_generation,
        (train_cluster, inference_cluster),
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_save_state,
        master_config,
    )


# ===============================================================================
# Training (clone of grpo_train with reprompt builder + teacher EMA update)
# ===============================================================================


def sdpo_train(
    policy: ColocatablePolicyInterface,
    policy_generation: Optional[GenerationInterface],
    dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: TokenizerType,
    loss_fn: SDPOLossFn,
    task_to_env: dict[str, EnvironmentInterface],
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    logger: Logger,
    checkpointer: CheckpointManager,
    grpo_save_state: GRPOSaveState,
    master_config: MasterConfig,
    processor: Optional[AutoProcessor] = None,
) -> None:
    """Run SDPO training.

    Differences from grpo_train:
      1. After student/prev logprobs are computed, the reprompt builder
         attaches teacher_input_ids / teacher_attention_mask /
         teacher_offsets / self_distillation_mask to the train batch (the
         teacher tensors must NOT pass through get_logprobs, whose
         sequence-dim assert only knows the student layout).
      2. The reference-policy logprob pass is skipped: SDPO replaces the PG
         loss entirely and Phase 1 has no KL-to-reference term (Verl default
         use_kl_loss=False).
      3. `policy.update_teacher_ema()` is called ONCE per training step after
         `policy.train()`, mirroring Verl's `_update_teacher` placement after
         the full minibatch loop.
    """
    assert not _should_use_nemo_gym(master_config), (
        "SDPO Phase 1 does not support NeMo-Gym rollouts"
    )

    sd_cfg = resolve_self_distillation_config(master_config)

    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config["checkpointing"]["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    NEED_REFIT = True
    if policy_generation is None:
        policy_generation = policy  # type: ignore
        NEED_REFIT = False
    POLICY_GENERATION_STALE = True  # tracks if generation needs a refit before running
    assert policy_generation is not None  # for mypy type check

    current_step = grpo_save_state["current_step"]
    total_steps = grpo_save_state["total_steps"]
    max_num_steps = master_config["grpo"]["max_num_steps"]
    current_epoch = grpo_save_state["current_epoch"]
    max_num_epochs = master_config["grpo"]["max_num_epochs"]
    consumed_samples = grpo_save_state["consumed_samples"]
    total_valid_tokens = grpo_save_state.get("total_valid_tokens", 0)
    val_at_start = master_config["grpo"]["val_at_start"]
    val_period = master_config["grpo"]["val_period"]
    colocated_inference = master_config["policy"]["generation"]["colocated"]["enabled"]

    if val_at_start and current_step == 0:
        print("\n🔍 Running initial validation...", flush=True)
        if NEED_REFIT and POLICY_GENERATION_STALE:
            refit_policy_generation(policy, policy_generation, colocated_inference)
            POLICY_GENERATION_STALE = False
        else:
            policy_generation.prepare_for_generation()
        val_metrics, validation_timings = validate(
            policy_generation,
            val_dataloader,
            tokenizer,
            val_task_to_env,
            step=0,
            master_config=master_config,
        )
        policy_generation.finish_generation()
        logger.log_metrics(val_metrics, current_step, prefix="validation")
        logger.log_metrics(validation_timings, current_step, prefix="timing/validation")

    while current_epoch < max_num_epochs and total_steps < max_num_steps:
        print(f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_num_epochs} {'=' * 25}")
        batch_cache: BatchedDataDict[DatumSpec] = None
        dynamic_sampling_num_gen_batches = 0

        for batch in dataloader:
            print(
                f"\n{'=' * 25} Step {current_step + 1}/{min(len(dataloader), max_num_steps)} {'=' * 25}",
                flush=True,
            )
            maybe_gpu_profile_step(policy, total_steps + 1)
            if policy != policy_generation:
                maybe_gpu_profile_step(policy_generation, total_steps + 1)
            val_metrics, validation_timings = None, None

            with timer.time("total_step_time"):
                # Prepare batch
                print("▶ Preparing batch...", flush=True)
                with timer.time("data_processing"):
                    repeated_batch: BatchedDataDict[DatumSpec] = (
                        batch.repeat_interleave(
                            master_config["grpo"]["num_generations_per_prompt"]
                        )
                    )
                    batched_flat, input_lengths = batched_message_log_to_flat_message(
                        repeated_batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                    )
                    input_ids = batched_flat["token_ids"]

                print(
                    f"▶ Generating responses for batch of size {repeated_batch.size}...",
                    flush=True,
                )
                with timer.time("prepare_for_generation/total"):
                    if NEED_REFIT and POLICY_GENERATION_STALE:
                        refit_policy_generation(
                            policy,
                            policy_generation,
                            colocated_inference,
                            timer=timer,
                        )
                        POLICY_GENERATION_STALE = False
                    else:
                        if colocated_inference:
                            policy.offload_after_refit()
                        policy_generation.prepare_for_generation()

                dynamic_sampling_num_gen_batches += 1
                with timer.time("generation"):
                    if policy_generation is not None and hasattr(
                        policy_generation, "clear_vllm_logger_metrics"
                    ):
                        policy_generation.clear_vllm_logger_metrics()
                    if _should_use_async_rollouts(master_config):
                        (
                            repeated_batch,
                            rollout_metrics,
                        ) = run_async_multi_turn_rollout(
                            policy_generation=policy_generation,
                            input_batch=repeated_batch,
                            tokenizer=tokenizer,
                            task_to_env=task_to_env,
                            max_seq_len=master_config["policy"][
                                "max_total_sequence_length"
                            ],
                            max_rollout_turns=master_config["grpo"][
                                "max_rollout_turns"
                            ],
                            greedy=False,
                        )
                    else:
                        repeated_batch, rollout_metrics = run_multi_turn_rollout(
                            policy_generation=policy_generation,
                            input_batch=repeated_batch,
                            tokenizer=tokenizer,
                            task_to_env=task_to_env,
                            max_seq_len=master_config["policy"][
                                "max_total_sequence_length"
                            ],
                            max_rollout_turns=master_config["grpo"][
                                "max_rollout_turns"
                            ],
                            greedy=False,
                        )
                    policy_generation.finish_generation()
                    if policy_generation is not None and hasattr(
                        policy_generation, "get_vllm_logger_metrics"
                    ):
                        vllm_logger_metrics = (
                            policy_generation.get_vllm_logger_metrics()
                        )
                    else:
                        vllm_logger_metrics = {}

                # [MORALGYM PATCH 4] Call env post-processing to collect game metrics
                for env_actor in task_to_env.values():
                    repeated_batch, env_metrics = ray.get(
                        env_actor.global_post_process_and_metrics.remote(
                            repeated_batch
                        )
                    )
                    rollout_metrics.update(env_metrics)

                repeated_batch = scale_rewards(
                    repeated_batch, master_config["grpo"]["reward_scaling"]
                )
                if master_config["grpo"]["reward_shaping"]["enabled"]:
                    repeated_batch = apply_reward_shaping(
                        repeated_batch, master_config["grpo"]["reward_shaping"]
                    )

                # Calculate rewards & advantages
                print("▶ Processing rewards...,", flush=True)
                with timer.time("reward_calculation"):
                    rewards = repeated_batch["total_reward"]

                    print("▶ Computing advantages...", flush=True)
                    baseline, std = calculate_baseline_and_std_per_prompt(
                        input_ids,
                        rewards,
                        torch.ones_like(rewards),
                        leave_one_out_baseline=master_config["grpo"][
                            "use_leave_one_out_baseline"
                        ],
                    )
                    repeated_batch, is_batch_complete, batch_cache, ds_metrics = (
                        dynamic_sampling(
                            repeated_batch,
                            std,
                            baseline,
                            dynamic_sampling_num_gen_batches,
                            master_config,
                            timer,
                            batch_cache,
                        )
                    )
                    if ds_metrics:
                        ds_metrics["dynamic_sampling_num_gen_batches"] = (
                            dynamic_sampling_num_gen_batches
                        )
                    rewards = (
                        repeated_batch["total_reward"]
                        if not master_config["grpo"]["use_dynamic_sampling"]
                        else repeated_batch["filtered_reward"]
                    )
                    baseline = repeated_batch["baseline"]
                    std = repeated_batch["std"]

                    if not is_batch_complete:
                        continue
                    # Advantages are kept for logging/metrics parity with GRPO;
                    # SDPOLossFn does not consume them (the distillation loss
                    # replaces PG entirely).
                    advantages = (rewards - baseline).unsqueeze(-1)

                    if master_config["grpo"]["normalize_rewards"]:
                        advantages = normalize_advantages_with_epsilon(
                            advantages=advantages,
                            std=std,
                        )

                    # [MORALGYM PATCH 5] Per-step advantage computation
                    adv_cfg = master_config["grpo"].get("adv_estimator", {})
                    use_per_step = adv_cfg.get("per_step", False)
                    per_step_advs = None
                    if use_per_step and "per_round_rewards" in repeated_batch:
                        per_step_advs = compute_per_step_advantages_for_batch(
                            input_ids,
                            repeated_batch["per_round_rewards"],
                            leave_one_out=master_config["grpo"][
                                "use_leave_one_out_baseline"
                            ],
                            per_step_values=adv_cfg.get(
                                "per_step_values", "immediate"
                            ),
                            return_to_go_gamma=adv_cfg.get("return_to_go_gamma", 0.9),
                        )

                with timer.time("data_processing"):
                    use_overlong_filtering = master_config["grpo"]["overlong_filtering"]
                    if use_overlong_filtering:
                        loss_multiplier = repeated_batch["loss_multiplier"].clone()
                        truncated = repeated_batch["truncated"]

                        if isinstance(truncated, list):
                            truncated = torch.tensor(truncated, dtype=torch.bool)

                        loss_multiplier[truncated] = 0
                        repeated_batch["loss_multiplier"] = loss_multiplier
                    # Add loss mask and advantages to each message in LLMMessageLogType
                    for i, message_log in enumerate(repeated_batch["message_log"]):
                        assistant_round = 0
                        for j, message in enumerate(message_log):
                            if message["role"] == "assistant":
                                message["token_loss_mask"] = torch.ones_like(
                                    message["token_ids"]
                                )
                            else:
                                message["token_loss_mask"] = torch.zeros_like(
                                    message["token_ids"]
                                )
                            if "generation_logprobs" not in message:
                                message["generation_logprobs"] = torch.zeros_like(
                                    message["token_ids"], dtype=torch.float32
                                )
                            # [MORALGYM PATCH 5] Per-step: assign round-specific advantage
                            if (
                                per_step_advs is not None
                                and message["role"] == "assistant"
                            ):
                                round_adv = (
                                    per_step_advs[i, assistant_round].item()
                                    if assistant_round < per_step_advs.shape[1]
                                    else 0.0
                                )
                                message["advantages"] = torch.full_like(
                                    message["token_ids"],
                                    round_adv,
                                    dtype=torch.float32,
                                )
                                assistant_round += 1
                            else:
                                message["advantages"] = advantages[i].expand(
                                    message["token_ids"].shape
                                )

                    flat_messages, input_lengths = batched_message_log_to_flat_message(
                        repeated_batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        make_sequence_length_divisible_by=master_config["policy"][
                            "make_sequence_length_divisible_by"
                        ],
                    )

                    train_data = BatchedDataDict[SDPOLossDataDict](
                        {
                            "input_ids": flat_messages["token_ids"],
                            "input_lengths": input_lengths,
                            "advantages": flat_messages["advantages"],
                            "generation_logprobs": flat_messages["generation_logprobs"],
                            "token_mask": flat_messages["token_loss_mask"],
                            "sample_mask": repeated_batch["loss_multiplier"],
                        }
                    )
                    train_data.update(
                        flat_messages.get_multimodal_dict(as_tensors=False)
                    )
                    train_data.to("cpu")

                print("▶ Preparing for logprob inference...", flush=True)
                with timer.time("logprob_inference_prep"):
                    policy.prepare_for_lp_inference()

                print("▶ Computing logprobs...", flush=True)
                # Timer keeps grpo's name (despite no reference pass here)
                # because print_performance_metrics hardcodes this key.
                with timer.time("policy_and_reference_logprobs"):
                    # NOTE: no reference-policy logprob pass — SDPO replaces PG
                    # and Phase 1 has no KL-to-reference term (Verl default).
                    fprop_logprobs = policy.get_logprobs(train_data)["logprobs"]
                    train_data["prev_logprobs"] = fprop_logprobs

                # Teacher tensors are attached AFTER get_logprobs: the student
                # logprob path asserts a uniform sequence dim and must never
                # see the (longer) reprompted teacher layout.
                print("▶ Building reprompt (teacher) batch...", flush=True)
                with timer.time("reprompt_build"):
                    teacher_tensors, sd_batch_metrics = build_reprompt_batch(
                        repeated_batch=repeated_batch,
                        tokenizer=tokenizer,
                        sd_cfg=sd_cfg,
                        pad_token_id=tokenizer.pad_token_id,
                        make_sequence_length_divisible_by=master_config["policy"][
                            "make_sequence_length_divisible_by"
                        ],
                    )
                    train_data.update(teacher_tensors)

                print("▶ Preparing for training...", flush=True)
                with timer.time("training_prep"):
                    policy.prepare_for_training()
                    POLICY_GENERATION_STALE = True

                print("▶ Training policy...", flush=True)
                with timer.time("policy_training"):
                    train_results = policy.train(train_data, loss_fn)

                # Teacher EMA update: once per training step after the full
                # minibatch loop, mirroring Verl's _update_teacher placement
                # (dp_actor.py:930-935). Trust-region mode keeps the frozen
                # initial teacher and never updates.
                if sd_cfg["teacher_regularization"] == "ema":
                    with timer.time("teacher_ema_update"):
                        policy.update_teacher_ema(sd_cfg["teacher_update_rate"])

                is_last_step = (total_steps + 1 >= max_num_steps) or (
                    (current_epoch + 1 == max_num_epochs)
                    and (current_step + 1 == len(dataloader))
                )

                if val_period > 0 and (total_steps + 1) % val_period == 0:
                    if NEED_REFIT and POLICY_GENERATION_STALE:
                        refit_policy_generation(
                            policy,
                            policy_generation,
                            colocated_inference,
                        )
                        POLICY_GENERATION_STALE = False
                    else:
                        if colocated_inference:
                            policy.offload_after_refit()
                        policy_generation.prepare_for_generation()
                    val_metrics, validation_timings = validate(
                        policy_generation,
                        val_dataloader,
                        tokenizer,
                        val_task_to_env,
                        step=total_steps + 1,
                        master_config=master_config,
                    )
                    policy_generation.finish_generation()
                    logger.log_metrics(
                        validation_timings, total_steps + 1, prefix="timing/validation"
                    )
                    logger.log_metrics(
                        val_metrics, total_steps + 1, prefix="validation"
                    )

                flat_advantages = flat_messages["advantages"]
                flat_token_mask = flat_messages["token_loss_mask"]
                response_advantages = torch.masked_select(
                    flat_advantages, flat_token_mask.bool()
                )

                metrics = {
                    "loss": train_results["loss"].numpy(),
                    "grad_norm": train_results["grad_norm"].numpy(),
                    "reward": rewards.numpy(),
                    "mean_prompt_length": repeated_batch["length"].numpy(),
                    "total_num_tokens": input_lengths.numpy(),
                    "advantages/mean": torch.mean(response_advantages).detach().item()
                    if response_advantages.numel() > 0
                    else 0.0,
                    "advantages/max": torch.max(response_advantages).detach().item()
                    if response_advantages.numel() > 0
                    else 0.0,
                    "advantages/min": torch.min(response_advantages).detach().item()
                    if response_advantages.numel() > 0
                    else 0.0,
                    **ds_metrics,
                }
                if "moe_metrics" in train_results:
                    metrics.update(
                        {f"moe/{k}": v for k, v in train_results["moe_metrics"].items()}
                    )
                if master_config["grpo"]["use_dynamic_sampling"]:
                    metrics["filtered_reward"] = rewards.numpy()
                    metrics["reward"] = repeated_batch["total_reward"].numpy()

                metrics.update(train_results["all_mb_metrics"])
                for k, v in metrics.items():
                    if k in {"probs_ratio_min", "probs_ratio_clamped_min"}:
                        valid_values = [x for x in v if not np.isinf(x)]
                        metrics[k] = (
                            np.min(valid_values).item() if valid_values else -1.0
                        )
                    elif k in {"probs_ratio_max", "probs_ratio_clamped_max"}:
                        valid_values = [x for x in v if not np.isinf(x)]
                        metrics[k] = (
                            np.max(valid_values).item() if valid_values else -1.0
                        )
                    elif k in {
                        "lr",
                        "wd",
                        "reward",
                        "filtered_reward",
                        "global_valid_seqs",
                        "global_valid_toks",
                        "mean_prompt_length",
                    } or k.startswith("self_distillation/"):
                        metrics[k] = np.mean(v).item()
                    else:
                        metrics[k] = np.sum(v).item()

                metrics.update(sd_batch_metrics)
                metrics.update(rollout_metrics)
                metrics["vllm_logger_metrics"] = vllm_logger_metrics
                total_valid_tokens += metrics["global_valid_toks"]

                ## Checkpointing
                consumed_samples += master_config["grpo"]["num_prompts_per_step"]
                timeout.mark_iteration()

                should_save_by_step = (
                    is_last_step
                    or (total_steps + 1) % master_config["checkpointing"]["save_period"]
                    == 0
                )
                should_save_by_timeout = timeout.check_save()

                if master_config["checkpointing"]["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    policy.prepare_for_training()

                    grpo_save_state["current_step"] = current_step + 1
                    grpo_save_state["total_steps"] = total_steps + 1
                    grpo_save_state["current_epoch"] = current_epoch
                    grpo_save_state["total_valid_tokens"] = total_valid_tokens
                    if val_metrics is not None:
                        grpo_save_state["val_reward"] = val_metrics["accuracy"]
                    elif "val_reward" in grpo_save_state:
                        del grpo_save_state["val_reward"]
                    grpo_save_state["consumed_samples"] = consumed_samples

                    full_metric_name = master_config["checkpointing"]["metric_name"]
                    if full_metric_name is not None:
                        assert full_metric_name.startswith(
                            "train:"
                        ) or full_metric_name.startswith("val:"), (
                            f"metric_name={full_metric_name} must start with 'val:' or 'train:',\n"
                            f'followed by the corresponding name in the "val" or "train" metrics dictionary.'
                        )
                        prefix, metric_name = full_metric_name.split(":", 1)
                        metrics_source = metrics if prefix == "train" else val_metrics
                        if not metrics_source:
                            warnings.warn(
                                f"You asked to save checkpoints based on {metric_name} but no {prefix} metrics were collected. "
                                "This checkpoint will not be saved as top-k.",
                                stacklevel=2,
                            )
                            if full_metric_name in grpo_save_state:
                                del grpo_save_state[full_metric_name]
                        elif metric_name not in metrics_source:
                            raise ValueError(
                                f"Metric {metric_name} not found in {prefix} metrics"
                            )
                        else:
                            grpo_save_state[full_metric_name] = metrics_source[
                                metric_name
                            ]

                    with timer.time("checkpointing"):
                        print(
                            f"Saving checkpoint for step {total_steps + 1}...",
                            flush=True,
                        )
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            total_steps + 1, grpo_save_state, master_config
                        )
                        policy.save_checkpoint(
                            weights_path=os.path.join(
                                checkpoint_path, "policy", "weights"
                            ),
                            optimizer_path=os.path.join(
                                checkpoint_path, "policy", "optimizer"
                            ),
                            tokenizer_path=os.path.join(
                                checkpoint_path, "policy", "tokenizer"
                            ),
                            checkpointing_cfg=master_config["checkpointing"],
                        )
                        torch.save(
                            dataloader.state_dict(),
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        checkpointer.finalize_checkpoint(checkpoint_path)

            # Logging
            log_data = {"content": flat_messages["content"]}
            log_data["rewards"] = rewards.tolist()
            if master_config["grpo"]["use_dynamic_sampling"]:
                log_data["filtered_rewards"] = rewards.tolist()
                log_data["rewards"] = repeated_batch["total_reward"].tolist()

            log_data["generation_logprobs"] = train_data["generation_logprobs"].tolist()
            log_data["prev_logprobs"] = train_data["prev_logprobs"].tolist()
            log_data["input_lengths"] = input_lengths.tolist()
            log_data["self_distillation_mask"] = train_data[
                "self_distillation_mask"
            ].tolist()
            logger.log_batched_dict_as_jsonl(
                log_data, f"train_data_step{total_steps + 1}.jsonl"
            )

            timing_metrics: dict[str, float] = timer.get_timing_metrics(
                reduction_op="sum"
            )  # type: ignore
            if master_config["policy"]["generation"].get("vllm_cfg", {}).get(
                "enable_vllm_metrics_logger", False
            ) and master_config.get("logger", {}).get("wandb_enabled", False):
                log_generation_metrics_to_wandb(
                    vllm_logger_metrics,
                    total_steps + 1,
                    master_config["policy"]["generation"]["vllm_cfg"][
                        "vllm_metrics_logger_interval"
                    ],
                    logger,
                )

            if (
                master_config["policy"]["generation"]
                .get("vllm_cfg", {})
                .get("async_engine", False)
            ):
                for metric_name in metrics.keys():
                    if metric_name.startswith("histogram/"):
                        logger.log_histogram(
                            metrics[metric_name],
                            total_steps + 1,
                            f"generation_metrics/{metric_name}",
                        )

            print("\n📊 Training Results (SDPO):")
            print(f"  • Loss: {metrics['loss']:.4f}")
            print(
                "  • Reprompted sample fraction: "
                f"{metrics['self_distillation/reprompt_sample_fraction']:.3f}"
            )
            if master_config["grpo"]["use_dynamic_sampling"]:
                print(f"  • Avg Filtered Reward: {np.mean(rewards.numpy()):.4f}")
                print(
                    f"  • Avg Total Reward: {np.mean(repeated_batch['total_reward'].numpy()):.4f}"
                )
            else:
                print(f"  • Avg Reward: {np.mean(rewards.numpy()):.4f}")
            print(
                f"  • Mean Generation Length: {rollout_metrics['mean_gen_tokens_per_sample']:.4f}",
                flush=True,
            )

            print("\n⏱️  Timing:", flush=True)
            total_time = timing_metrics.get("total_step_time", 0)

            total_num_gpus = (
                master_config["cluster"]["num_nodes"]
                * master_config["cluster"]["gpus_per_node"]
            )

            print(f"  • Total step time: {total_time:.2f}s", flush=True)
            for k, v in sorted(
                timing_metrics.items(), key=lambda item: item[1], reverse=True
            ):
                if k != "total_step_time":
                    percent = (v / total_time * 100) if total_time > 0 else 0
                    print(f"  • {k}: {v:.2f}s ({percent:.1f}%)", flush=True)

            timing_metrics["valid_tokens_per_sec_per_gpu"] = (
                metrics["global_valid_toks"] / total_time / total_num_gpus
            )
            performance_metrics = print_performance_metrics(
                train_results, metrics, timing_metrics, master_config
            )

            logger.log_metrics(metrics, total_steps + 1, prefix="train")
            logger.log_metrics(
                performance_metrics, total_steps + 1, prefix="performance"
            )
            logger.log_metrics(timing_metrics, total_steps + 1, prefix="timing/train")

            batch_cache = None
            dynamic_sampling_num_gen_batches = 0

            timer.reset()
            current_step += 1
            total_steps += 1
            if should_save_by_timeout:
                print("Timeout has been reached, stopping training early", flush=True)
                return
            if total_steps >= max_num_steps:
                print(
                    "Max number of steps has been reached, stopping training early",
                    flush=True,
                )
                return

        current_epoch += 1
        current_step = 0  # Reset step counter for new epoch
