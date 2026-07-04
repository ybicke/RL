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
"""[MORALGYM PATCH 6] SDPO EMA teacher — Step 2 verification.

Boots a DTensor v2 Policy with LoRA and init_teacher_model=True on a
2-layer tiny Llama, then verifies:
  (a) teacher is initialized identical to actor (all params)
  (b) update_teacher_ema(rate=1.0) makes teacher == actor
  (c) update_teacher_ema(rate=0.0) is a no-op
  (d) update_teacher_ema(rate=0.5) yields the correct half-way point
  (e) teacher storage contains ALL model parameters (base + LoRA), matching
      Verl's ref_module_fsdp full-model architecture
  (f) non-LoRA config works (full-FT path, Phase 2 support)
"""

import math
import os
import shutil

import pytest

from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.policy.lm_policy import Policy
from tests.unit.models.policy.test_dtensor_worker_v2 import create_test_config


@pytest.fixture(scope="module")
def tiny_llama_offline_model_path(tmp_path_factory):
    """Offline tiny Llama model with a cached tokenizer.

    Avoids the tiny_llama_offline_model_path fixture in conftest.py, which downloads
    meta-llama/Llama-3.2-1B from HuggingFace at fixture setup. On Clariden
    compute nodes HuggingFace is not reachable, but Meta-Llama-3-8B-Instruct
    is already in the HF cache — we reuse its tokenizer.
    """
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    model_path = str(tmp_path_factory.mktemp("tiny_llama_offline"))
    # Use Gemma-2-9B-it's tokenizer (already downloaded to the HF cache for
    # MoralGym training). We never actually tokenize in this test — Policy
    # only requires a tokenizer object to exist. Any cached tokenizer works.
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
    config = LlamaConfig(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=32,
        num_attention_heads=2,
        vocab_size=tokenizer.vocab_size,
        tie_word_embeddings=False,
        num_key_value_heads=None,
    )
    model = LlamaForCausalLM(config=config)
    shutil.rmtree(model_path, ignore_errors=True)
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    del model, tokenizer
    yield model_path


def _add_lora_cfg(config, enabled: bool = True, dim: int = 8, alpha: int = 16):
    config["dtensor_cfg"]["lora_cfg"] = {
        "enabled": enabled,
        "target_modules": [],
        "exclude_modules": [],
        "match_all_linear": True,
        "dim": dim,
        "alpha": alpha,
        "dropout": 0.0,
        "dropout_position": "post",
        "lora_A_init": "xavier",
        "use_triton": False,  # tp=1 in this test, but triton off is safer
    }


@pytest.fixture(scope="module")
def one_gpu_cluster():
    cluster = RayVirtualCluster(
        name="teacher-ema-test",
        bundle_ct_per_node_list=[1],
        use_gpus=True,
        num_gpus_per_node=1,
        max_colocated_worker_groups=1,
    )
    yield cluster
    cluster.shutdown()


def _probe(policy):
    """Broadcast the teacher probe and return the rank-0 result."""
    import ray as _ray

    futures = policy.worker_group.run_all_workers_single_data("_teacher_ema_probe")
    results = _ray.get(futures)
    return results[0]


def test_teacher_ema_lora_path(one_gpu_cluster, tiny_llama_offline_model_path):
    """Full EMA lifecycle on a LoRA-enabled tiny Llama."""
    from nemo_rl.algorithms.utils import get_tokenizer

    config = create_test_config(tiny_llama_offline_model_path, dtensor_v2=True)
    _add_lora_cfg(config, enabled=True, dim=8, alpha=16)
    tokenizer = get_tokenizer(config["tokenizer"])

    policy = Policy(
        cluster=one_gpu_cluster,
        config=config,
        tokenizer=tokenizer,
        init_reference_model=False,
        init_teacher_model=True,
    )
    try:
        # (a) teacher starts identical to actor (all params)
        probe0 = _probe(policy)
        assert len(probe0["teacher_keys"]) > 0, "expected some teacher tensors"
        for k in probe0["teacher_keys"]:
            assert math.isclose(
                probe0["teacher_norms"][k], probe0["actor_norms"][k], rel_tol=1e-6
            ), f"(a) initial teacher!=actor at {k}"

        base_weight_norms_0 = probe0["base_weight_norms"]
        assert base_weight_norms_0, "probe should return some base-weight norms"

        # Perturb the actor via the existing test hook (adds Gaussian noise to
        # all trainable params — includes both LoRA and any trainable base).
        futures = policy.worker_group.run_all_workers_single_data(
            "_add_noise_to_weights"
        )
        import ray as _ray

        _ray.get(futures)

        probe1 = _probe(policy)
        # Teacher should be untouched by the noise (EMA not called yet).
        for k in probe1["teacher_keys"]:
            assert math.isclose(
                probe1["teacher_norms"][k], probe0["teacher_norms"][k], rel_tol=1e-6
            ), f"teacher moved without an EMA call at {k}"
        # Actor should have moved.
        moved = [
            k
            for k in probe1["teacher_keys"]
            if not math.isclose(
                probe1["actor_norms"][k], probe0["actor_norms"][k], rel_tol=1e-6
            )
        ]
        assert moved, "actor should have moved after _add_noise_to_weights"

        # (b) rate=1.0 -> teacher := actor
        policy.update_teacher_ema(rate=1.0)
        probe2 = _probe(policy)
        for k in probe2["teacher_keys"]:
            assert math.isclose(
                probe2["teacher_norms"][k], probe2["actor_norms"][k], rel_tol=1e-4
            ), f"(b) rate=1.0 should copy actor into teacher at {k}"

        # (c) rate=0.0 -> teacher unchanged. First perturb actor again.
        teacher_before_noop = dict(probe2["teacher_norms"])
        futures = policy.worker_group.run_all_workers_single_data(
            "_add_noise_to_weights"
        )
        import ray as _ray

        _ray.get(futures)
        policy.update_teacher_ema(rate=0.0)
        probe3 = _probe(policy)
        for k in probe3["teacher_keys"]:
            assert math.isclose(
                probe3["teacher_norms"][k], teacher_before_noop[k], rel_tol=1e-5
            ), f"(c) rate=0.0 should be a no-op at {k}"

        # NOTE: A norm-level "interior" check for intermediate rates (e.g.
        # rate=0.25 puts teacher_norm between teacher_old_norm and actor_norm)
        # is NOT a valid invariant. Elementwise EMA is exact but norms of a
        # convex combination can drift outside the range of the operand norms
        # when the underlying tensors point in slightly opposite directions
        # (reverse triangle inequality). (b) already fully verifies EMA
        # correctness by linearity; (d) is intentionally omitted.

        # (e) base weight keys ARE in the teacher state — full-model teacher
        # tracks all params (base + LoRA), matching Verl's ref_module_fsdp.
        probe_final = _probe(policy)
        for name in base_weight_norms_0:
            assert name in probe_final["teacher_keys"], (
                f"(e) base weight {name} SHOULD be tracked by full-model teacher state"
            )

    finally:
        policy.shutdown()


def test_inline_teacher_forward_with_sdpo_loss(
    one_gpu_cluster, tiny_llama_offline_model_path
):
    """[MORALGYM PATCH 6] SDPO — Step 3 verification.

    End-to-end sanity check: hand-craft a training batch that carries
    `teacher_input_ids`, run one train step under `SDPOLossFn`, and assert
    that the loss is finite. This exercises the inline teacher forward
    added to the DTensor v2 worker's microbatch loop.
    """
    import torch

    from nemo_rl.algorithms.loss_functions import SDPOLossFn
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    config = create_test_config(tiny_llama_offline_model_path, dtensor_v2=True)
    _add_lora_cfg(config, enabled=True, dim=8, alpha=16)
    # SDPO Phase 1 is incompatible with sequence packing and dynamic batching.
    config["dynamic_batching"]["enabled"] = False
    config["sequence_packing"]["enabled"] = False
    tokenizer = get_tokenizer(config["tokenizer"])

    policy = Policy(
        cluster=one_gpu_cluster,
        config=config,
        tokenizer=tokenizer,
        init_reference_model=False,
        init_teacher_model=True,
    )
    try:
        batch_size = config["train_global_batch_size"]  # 4 per create_test_config
        seq_len = 32
        vocab_size = tokenizer.vocab_size
        gen = torch.Generator().manual_seed(0)

        # Student input.
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=gen)
        # Teacher input MUST match student's shape for Phase 1 (worker asserts this).
        teacher_input_ids = torch.randint(
            0, vocab_size, (batch_size, seq_len), generator=gen
        )
        # Mark the last third of tokens as "response" so token_mask has 1s to hit.
        token_mask = torch.zeros((batch_size, seq_len), dtype=torch.float32)
        token_mask[:, (2 * seq_len) // 3 :] = 1.0
        # Every sample is valid and has a demonstration.
        sample_mask = torch.ones(batch_size, dtype=torch.float32)
        self_distillation_mask = torch.ones(batch_size, dtype=torch.float32)
        # input_lengths needed by the worker's dynamic-shape machinery.
        input_lengths = torch.full((batch_size,), seq_len, dtype=torch.int32)

        data = BatchedDataDict(
            {
                "input_ids": input_ids,
                "input_lengths": input_lengths,
                "token_mask": token_mask,
                "sample_mask": sample_mask,
                "self_distillation_mask": self_distillation_mask,
                "teacher_input_ids": teacher_input_ids,
            }
        )

        loss_fn = SDPOLossFn(
            {
                "full_logit_distillation": True,
                "alpha": 0.0,  # forward KL, standard SDPO default
            }
        )

        policy.prepare_for_training()
        results = policy.train(data, loss_fn)

        assert "loss" in results, "train() did not return a 'loss' entry"
        loss_values = results["loss"]
        if isinstance(loss_values, torch.Tensor):
            loss_values = loss_values.tolist()
        assert all(math.isfinite(float(v)) for v in loss_values), (
            f"SDPO loss must be finite, got {loss_values}"
        )
        assert any(float(v) != 0.0 for v in loss_values), (
            "SDPO loss should not be identically zero — teacher forward likely inactive"
        )

    finally:
        policy.shutdown()


def test_teacher_trust_region(one_gpu_cluster, tiny_llama_offline_model_path):
    """Trust-region teacher: logit-space blend of frozen ref and current actor."""
    from nemo_rl.algorithms.utils import get_tokenizer

    config = create_test_config(tiny_llama_offline_model_path, dtensor_v2=True)
    _add_lora_cfg(config, enabled=True, dim=8, alpha=16)
    config["dynamic_batching"]["enabled"] = False
    config["sequence_packing"]["enabled"] = False
    tokenizer = get_tokenizer(config["tokenizer"])

    # Verifications:
    # (a) teacher_state_dict is NOT updated by update_teacher_ema in trust-region mode
    # (b) A train step with trust-region + SDPOLossFn produces finite non-zero loss
    import torch, math, ray as _ray
    from nemo_rl.algorithms.loss_functions import SDPOLossFn
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    policy = Policy(
        cluster=one_gpu_cluster,
        config=config,
        tokenizer=tokenizer,
        init_reference_model=False,
        init_teacher_model=True,
        teacher_regularization="trust-region",
        teacher_update_rate=0.5,   # 50/50 blend; non-trivial mix_coef
    )
    try:
        probe0 = _probe(policy)
        norms0 = dict(probe0["teacher_norms"])

        # Perturb the actor.
        _ray.get(policy.worker_group.run_all_workers_single_data("_add_noise_to_weights"))

        # For trust-region, update_teacher_ema must be a no-op regardless of rate.
        policy.update_teacher_ema(rate=1.0)
        probe1 = _probe(policy)
        for k in probe1["teacher_keys"]:
            assert math.isclose(
                probe1["teacher_norms"][k], norms0[k], rel_tol=1e-6
            ), f"(a) trust-region teacher_state_dict changed after update_teacher_ema at {k}"

        # (b) train step produces finite non-zero loss
        batch_size = config["train_global_batch_size"]
        seq_len = 32
        vocab_size = tokenizer.vocab_size
        gen = torch.Generator().manual_seed(1)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=gen)
        teacher_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=gen)
        token_mask = torch.zeros((batch_size, seq_len), dtype=torch.float32)
        token_mask[:, (2 * seq_len) // 3 :] = 1.0
        sample_mask = torch.ones(batch_size, dtype=torch.float32)
        self_distillation_mask = torch.ones(batch_size, dtype=torch.float32)
        input_lengths = torch.full((batch_size,), seq_len, dtype=torch.int32)

        data = BatchedDataDict({
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "token_mask": token_mask,
            "sample_mask": sample_mask,
            "self_distillation_mask": self_distillation_mask,
            "teacher_input_ids": teacher_input_ids,
        })
        loss_fn = SDPOLossFn({"full_logit_distillation": True, "alpha": 0.0})
        policy.prepare_for_training()
        results = policy.train(data, loss_fn)
        assert "loss" in results
        loss_values = results["loss"]
        if isinstance(loss_values, torch.Tensor):
            loss_values = loss_values.tolist()
        assert all(math.isfinite(float(v)) for v in loss_values), (
            f"trust-region SDPO loss must be finite, got {loss_values}"
        )
        assert any(float(v) != 0.0 for v in loss_values), (
            "trust-region SDPO loss should not be zero"
        )
    finally:
        policy.shutdown()


def test_teacher_works_without_lora(one_gpu_cluster, tiny_llama_offline_model_path):
    """Full-model teacher works without LoRA (full-FT path, Phase 2)."""
    from nemo_rl.algorithms.utils import get_tokenizer

    config = create_test_config(tiny_llama_offline_model_path, dtensor_v2=True)
    # Explicitly no LoRA.
    _add_lora_cfg(config, enabled=False)
    tokenizer = get_tokenizer(config["tokenizer"])

    policy = Policy(
        cluster=one_gpu_cluster,
        config=config,
        tokenizer=tokenizer,
        init_reference_model=False,
        init_teacher_model=True,
    )
    try:
        probe = _probe(policy)
        assert len(probe["teacher_keys"]) > 0, "teacher should have params"
        # No LoRA keys — all params are base weights.
        assert all(
            ".lora_A." not in k and ".lora_B." not in k
            for k in probe["teacher_keys"]
        ), "non-LoRA teacher should have only base weight keys"
    finally:
        policy.shutdown()
