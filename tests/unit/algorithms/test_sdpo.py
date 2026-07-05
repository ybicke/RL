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
"""[MORALGYM PATCH 6] SDPO reprompt-builder unit tests (CPU-only).

Covers the Step 4 verification cases from the integration plan:
  (a) empty success group -> self_distillation_mask is all zeros
  (b) dont_reprompt_on_self_success correctly excludes self
  (c) teacher token layout is [reprompted first message] ++ [rest of the
      student message log], with offsets = len(teacher_first) - len(student_first)
plus config validation, raw_prompt requirement, and truncation fallback.
"""

import pytest
import torch

from nemo_rl.algorithms.sdpo import (
    SELF_DISTILLATION_DEFAULTS,
    _collect_solutions_by_uid,
    _get_solution,
    _remove_thinking_trace,
    build_reprompt_batch,
    resolve_self_distillation_config,
)

PAD_ID = 0


class MockTokenizer:
    """Deterministic char-level tokenizer: one token per character."""

    pad_token_id = PAD_ID

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    ):
        assert len(messages) == 1 and messages[0]["role"] == "user"
        assert tokenize is False and add_generation_prompt is True
        return "<|user|>" + messages[0]["content"] + "<|assistant|>"

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        assert return_tensors == "pt" and add_special_tokens is False
        ids = torch.tensor([[(ord(c) % 250) + 1 for c in text]], dtype=torch.long)
        return {"input_ids": ids}


def _sd_cfg(**overrides):
    master_config = {"policy": {"self_distillation": overrides}}
    return resolve_self_distillation_config(master_config)


def _row(prompt_len, response_text, raw_prompt="What is 2+2?"):
    """One-turn message log: user (templated tokens) + assistant response."""
    response_ids = torch.tensor(
        [(ord(c) % 250) + 1 for c in response_text], dtype=torch.long
    )
    log = [
        {
            "role": "user",
            "content": "<already templated>",
            "token_ids": torch.arange(1, prompt_len + 1, dtype=torch.long),
        },
        {"role": "assistant", "content": response_text, "token_ids": response_ids},
    ]
    return log, {"raw_prompt": raw_prompt}


def _batch(rows, rewards, uids):
    logs, extras = zip(*rows)
    return {
        "message_log": list(logs),
        "extra_env_info": list(extras),
        "total_reward": torch.tensor(rewards, dtype=torch.float32),
        "idx": torch.tensor(uids, dtype=torch.long),
    }


def _expected_teacher_first(tokenizer, sd_cfg, raw_prompt, solution_str):
    solution_section = sd_cfg["solution_template"].format(
        successful_previous_attempt=solution_str
    )
    reprompt_text = sd_cfg["reprompt_template"].format(
        prompt=raw_prompt, solution=solution_section, feedback=""
    )
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": reprompt_text}],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    return tokenizer(templated, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ][0]


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def test_resolve_config_defaults():
    cfg = _sd_cfg()
    assert cfg == SELF_DISTILLATION_DEFAULTS


def test_resolve_config_override_and_unknown_key():
    cfg = _sd_cfg(alpha=0.5, teacher_update_rate=0.1)
    assert cfg["alpha"] == 0.5
    assert cfg["teacher_update_rate"] == 0.1
    assert cfg["full_logit_distillation"] is True

    with pytest.raises(ValueError, match="Unknown keys"):
        _sd_cfg(no_such_key=1)


@pytest.mark.parametrize(
    "overrides, exc, match",
    [
        ({"alpha": 1.5}, ValueError, "alpha"),
        ({"alpha": -0.1}, ValueError, "alpha"),
        ({"teacher_regularization": "frozen"}, ValueError, "teacher_regularization"),
        ({"reprompt_truncation": "left"}, ValueError, "reprompt_truncation"),
        (
            {"include_environment_feedback": True},
            NotImplementedError,
            "Phase 2",
        ),
    ],
)
def test_resolve_config_validation(overrides, exc, match):
    with pytest.raises(exc, match=match):
        _sd_cfg(**overrides)


# ---------------------------------------------------------------------------
# Verl-ported helpers
# ---------------------------------------------------------------------------


def test_collect_solutions_by_uid():
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.5])
    uids = [0, 0, 1, 1]
    out = _collect_solutions_by_uid(uids, rewards, success_reward_threshold=1.0)
    assert out[0] == [0]
    assert out[1] == [2]
    # threshold is inclusive (>=)
    out = _collect_solutions_by_uid(uids, rewards, success_reward_threshold=0.5)
    assert out[1] == [2, 3]


def test_get_solution_excludes_self():
    uids = [0, 0]
    responses = ["good answer", "bad answer"]
    success = {0: [0]}
    # Default: self-success is a valid demonstration.
    assert _get_solution(0, success, uids, responses) == "good answer"
    # dont_reprompt_on_self_success: sample 0 has no sibling demo left.
    assert (
        _get_solution(0, success, uids, responses, dont_reprompt_on_self_success=True)
        is None
    )
    # ...but its sibling still gets sample 0's response.
    assert (
        _get_solution(1, success, uids, responses, dont_reprompt_on_self_success=True)
        == "good answer"
    )


def test_remove_thinking_trace():
    assert (
        _remove_thinking_trace("<think>secret\nplan</think>  the answer is 4")
        == "the answer is 4"
    )
    assert _remove_thinking_trace("no tags here") == "no tags here"


# ---------------------------------------------------------------------------
# build_reprompt_batch
# ---------------------------------------------------------------------------


def test_empty_success_group_masks_all_zero():
    tok = MockTokenizer()
    sd_cfg = _sd_cfg()
    rows = [_row(5, "resp a"), _row(7, "resp b")]
    batch = _batch(rows, rewards=[0.0, 0.0], uids=[0, 0])

    tensors, metrics = build_reprompt_batch(batch, tok, sd_cfg, PAD_ID)

    assert torch.equal(tensors["self_distillation_mask"], torch.zeros(2))
    assert torch.equal(tensors["teacher_offsets"], torch.zeros(2, dtype=torch.long))
    # Teacher rows are exactly the student token stream (then padding).
    for i, (log, _) in enumerate(rows):
        student = torch.cat([m["token_ids"] for m in log])
        n = student.shape[0]
        assert torch.equal(tensors["teacher_input_ids"][i, :n], student)
        assert tensors["teacher_attention_mask"][i, :n].all()
        assert not tensors["teacher_attention_mask"][i, n:].any()
        assert (tensors["teacher_input_ids"][i, n:] == PAD_ID).all()
    assert metrics["self_distillation/reprompt_sample_fraction"] == 0.0
    assert metrics["self_distillation/success_group_fraction"] == 0.0


def test_token_layout_and_offsets():
    tok = MockTokenizer()
    sd_cfg = _sd_cfg()
    rows = [_row(5, "44"), _row(6, "wrong")]
    batch = _batch(rows, rewards=[1.0, 0.0], uids=[0, 0])

    tensors, metrics = build_reprompt_batch(batch, tok, sd_cfg, PAD_ID)

    # Both rows share uid 0; row 0's response "44" is the demonstration.
    assert torch.equal(tensors["self_distillation_mask"], torch.ones(2))
    expected_first = _expected_teacher_first(tok, sd_cfg, "What is 2+2?", "44")
    for i, (log, _) in enumerate(rows):
        student_first = log[0]["token_ids"]
        rest = torch.cat([m["token_ids"] for m in log[1:]])
        expected_row = torch.cat([expected_first, rest])
        n = expected_row.shape[0]
        assert (
            tensors["teacher_offsets"][i]
            == expected_first.shape[0] - student_first.shape[0]
        )
        assert tensors["teacher_offsets"][i] > 0
        assert torch.equal(tensors["teacher_input_ids"][i, :n], expected_row)
        assert tensors["teacher_attention_mask"][i, :n].all()
        assert not tensors["teacher_attention_mask"][i, n:].any()
    assert metrics["self_distillation/reprompt_sample_fraction"] == 1.0
    assert metrics["self_distillation/success_group_fraction"] == 1.0
    assert metrics["self_distillation/success_sample_fraction"] == 1.0


def test_multi_turn_rest_is_preserved():
    tok = MockTokenizer()
    sd_cfg = _sd_cfg()
    log = [
        {
            "role": "user",
            "content": "<templated>",
            "token_ids": torch.arange(1, 6, dtype=torch.long),
        },
        {
            "role": "assistant",
            "content": "move C",
            "token_ids": torch.tensor([11, 12], dtype=torch.long),
        },
        {
            "role": "user",
            "content": "env feedback",
            "token_ids": torch.tensor([21, 22, 23], dtype=torch.long),
        },
        {
            "role": "assistant",
            "content": " then D",
            "token_ids": torch.tensor([31], dtype=torch.long),
        },
    ]
    batch = {
        "message_log": [log],
        "extra_env_info": [{"raw_prompt": "play the game"}],
        "total_reward": torch.tensor([1.0]),
        "idx": torch.tensor([0]),
    }

    tensors, _ = build_reprompt_batch(batch, tok, sd_cfg, PAD_ID)

    # Demonstration text joins ALL assistant turns.
    expected_first = _expected_teacher_first(
        tok, sd_cfg, "play the game", "move C then D"
    )
    rest = torch.cat([m["token_ids"] for m in log[1:]])
    expected_row = torch.cat([expected_first, rest])
    n = expected_row.shape[0]
    assert torch.equal(tensors["teacher_input_ids"][0, :n], expected_row)
    assert tensors["teacher_offsets"][0] == expected_first.shape[0] - 5


def test_dont_reprompt_on_self_success_in_batch():
    tok = MockTokenizer()
    sd_cfg = _sd_cfg(dont_reprompt_on_self_success=True)
    rows = [_row(5, "the demo"), _row(5, "a miss")]
    batch = _batch(rows, rewards=[1.0, 0.0], uids=[0, 0])

    tensors, _ = build_reprompt_batch(batch, tok, sd_cfg, PAD_ID)

    # Row 0 succeeded but has no *sibling* success -> no demo for itself.
    assert torch.equal(
        tensors["self_distillation_mask"], torch.tensor([0.0, 1.0])
    )
    assert tensors["teacher_offsets"][0] == 0
    assert tensors["teacher_offsets"][1] > 0


def test_raw_prompt_from_metadata_attribute():
    """Multi-turn envs replace extra_env_info with a metadata object; the
    builder must read raw_prompt from an attribute as well as a dict key."""

    class FakeGameMetadata:
        raw_prompt = "What is 2+2?"

    tok = MockTokenizer()
    sd_cfg = _sd_cfg()
    log, _ = _row(5, "44")
    batch = {
        "message_log": [log],
        "extra_env_info": [FakeGameMetadata()],
        "total_reward": torch.tensor([1.0]),
        "idx": torch.tensor([0]),
    }
    tensors, _ = build_reprompt_batch(batch, tok, sd_cfg, PAD_ID)
    assert tensors["self_distillation_mask"][0] == 1.0
    expected_first = _expected_teacher_first(tok, sd_cfg, "What is 2+2?", "44")
    assert tensors["teacher_offsets"][0] == expected_first.shape[0] - 5


def test_empty_raw_prompt_attribute_raises():
    class EmptyMetadata:
        raw_prompt = ""  # e.g. env default when the datum never stashed it

    tok = MockTokenizer()
    sd_cfg = _sd_cfg()
    log, _ = _row(5, "44")
    batch = {
        "message_log": [log],
        "extra_env_info": [EmptyMetadata()],
        "total_reward": torch.tensor([1.0]),
        "idx": torch.tensor([0]),
    }
    with pytest.raises(ValueError, match="raw_prompt"):
        build_reprompt_batch(batch, tok, sd_cfg, PAD_ID)


def test_missing_raw_prompt_raises():
    tok = MockTokenizer()
    sd_cfg = _sd_cfg()
    log, _ = _row(5, "resp")
    batch = {
        "message_log": [log],
        "extra_env_info": [{}],  # no raw_prompt
        "total_reward": torch.tensor([1.0]),
        "idx": torch.tensor([0]),
    }
    with pytest.raises(ValueError, match="raw_prompt"):
        build_reprompt_batch(batch, tok, sd_cfg, PAD_ID)


def test_truncation_fallback_to_no_demo():
    tok = MockTokenizer()
    # Reprompt truncated to 3 tokens < student first message (5 tokens):
    # delta would be negative, so the row falls back to no-demonstration.
    sd_cfg = _sd_cfg(max_reprompt_len=3)
    rows = [_row(5, "ok")]
    batch = _batch(rows, rewards=[1.0], uids=[0])

    tensors, metrics = build_reprompt_batch(batch, tok, sd_cfg, PAD_ID)

    assert tensors["self_distillation_mask"][0] == 0.0
    assert tensors["teacher_offsets"][0] == 0
    student = torch.cat([m["token_ids"] for m in rows[0][0]])
    assert torch.equal(tensors["teacher_input_ids"][0, : student.shape[0]], student)
    # success_sample_fraction counts rows with a demonstration available,
    # reprompt_sample_fraction counts rows actually reprompted.
    assert metrics["self_distillation/success_sample_fraction"] == 1.0
    assert metrics["self_distillation/reprompt_sample_fraction"] == 0.0


def test_truncation_error_mode_raises():
    tok = MockTokenizer()
    sd_cfg = _sd_cfg(max_reprompt_len=3, reprompt_truncation="error")
    batch = _batch([_row(5, "ok")], rewards=[1.0], uids=[0])
    with pytest.raises(ValueError, match="max_reprompt_len"):
        build_reprompt_batch(batch, tok, sd_cfg, PAD_ID)


def test_sequence_length_divisible_by():
    tok = MockTokenizer()
    sd_cfg = _sd_cfg()
    batch = _batch([_row(5, "resp a")], rewards=[0.0], uids=[0])
    tensors, _ = build_reprompt_batch(
        batch, tok, sd_cfg, PAD_ID, make_sequence_length_divisible_by=8
    )
    assert tensors["teacher_input_ids"].shape[1] % 8 == 0
    assert tensors["teacher_attention_mask"].shape == tensors["teacher_input_ids"].shape
