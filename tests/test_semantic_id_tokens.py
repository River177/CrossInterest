from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from new_interest.training.semantic_id_tokens import (
    MERA_MEAN_INITIALIZATION_VERSION,
    SEMANTIC_INITIALIZATION_REDUCTION_DEVICE,
    SEMANTIC_INITIALIZATION_REDUCTION_DTYPE,
    SEMANTIC_INITIALIZATION_VERSION,
    _semantic_row,
    apply_torch_numeric_backend,
    audit_interest_token_drift,
    capture_torch_numeric_backend,
    deterministic_reload_numeric_backend,
    deterministic_sdpa_math_context,
    initialize_semantic_interest_token_rows,
    prepare_semantic_adapter_base,
    register_semantic_interest_tokens,
    resolve_token_modules,
    trainable_token_indices,
)


class FakeTokenizer:
    def __init__(self) -> None:
        self.vocab = {
            "Alpha": 0,
            "Topic": 1,
            "Zulu": 2,
            "Other": 3,
        }
        self.unk_token_id = None

    def __len__(self) -> int:
        return len(self.vocab)

    def __call__(self, text: str, *, add_special_tokens: bool):
        del add_special_tokens
        if text in self.vocab:
            return {"input_ids": [self.vocab[text]]}
        return {"input_ids": [self.vocab[value] for value in text.split()]}

    def add_special_tokens(self, value):
        added = 0
        for token in value["additional_special_tokens"]:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
        return added

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab[token]


class FakeConfig:
    tie_word_embeddings = False


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(6, 3)
        self.lm_head = torch.nn.Linear(3, 6, bias=False)
        self.config = FakeConfig()
        with torch.no_grad():
            self.model.embed_tokens.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 0.0, 3.0],
                        [1.0, 1.0, 1.0],
                        [9.0, 9.0, 9.0],
                        [8.0, 8.0, 8.0],
                    ]
                )
            )
            self.lm_head.weight.copy_(
                torch.tensor(
                    [
                        [2.0, 0.0, 0.0],
                        [0.0, 4.0, 0.0],
                        [0.0, 0.0, 6.0],
                        [2.0, 2.0, 2.0],
                        [7.0, 7.0, 7.0],
                        [6.0, 6.0, 6.0],
                    ]
                )
            )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def _mapping(path: Path) -> None:
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "interest_index": 0,
                    "interest_token": "<interest_000>",
                    "target_subcategory": "Alpha Topic",
                },
                {
                    "interest_index": 1,
                    "interest_token": "<interest_001>",
                    "target_subcategory": "Zulu Topic",
                },
            ]
        ),
        path,
    )


def test_register_and_semantically_initialize_untied_rows(
    tmp_path: Path,
) -> None:
    mapping = tmp_path / "mapping.parquet"
    _mapping(mapping)
    tokenizer = FakeTokenizer()
    spec = register_semantic_interest_tokens(
        tokenizer,
        mapping,
        expected_count=2,
    )
    assert spec.original_vocabulary_size == 4
    assert spec.resized_vocabulary_size == 6
    assert spec.token_ids == [4, 5]
    assert spec.entries[0].label_token_ids == (0, 1)

    model = FakeModel()
    audit, initial = initialize_semantic_interest_token_rows(model, spec)
    expected_input = torch.tensor([0.5, 1.0, 0.0])
    expected_input *= 1.5 / expected_input.norm()
    expected_output = torch.tensor([1.0, 2.0, 0.0])
    expected_output *= 3.0 / expected_output.norm()
    torch.testing.assert_close(
        model.get_input_embeddings().weight[4], expected_input
    )
    torch.testing.assert_close(
        model.get_output_embeddings().weight[4], expected_output
    )
    assert audit["tie_word_embeddings"] is False
    assert audit["input_module_name"] == "model.embed_tokens"
    assert audit["output_module_name"] == "lm_head"
    assert audit["initialization_version"] == (
        "label-token-mean-norm-matched-cpu-fp32-v2"
    )
    assert audit["reduction_device"] == "cpu"
    assert audit["reduction_dtype"] == "float32"
    assert trainable_token_indices(model, spec) == {
        "model.embed_tokens": [4, 5],
        "lm_head": [4, 5],
    }
    assert trainable_token_indices(model, spec, output_only=True) == {
        "lm_head": [4, 5],
    }

    with torch.no_grad():
        model.get_input_embeddings().weight[4:].add_(0.25)
        model.get_output_embeddings().weight[4:].add_(0.5)
    drift = audit_interest_token_drift(model, spec, initial)
    assert drift["all_input_rows_changed"] is True
    assert drift["all_output_rows_changed"] is True
    assert drift["changed_input_rows"] == 2
    assert drift["changed_output_rows"] == 2
    assert drift["min_input_drift_l2"] > 0
    assert drift["min_output_drift_l2"] > 0


def test_semantic_row_uses_cpu_float32_reduction_and_restores_dtype() -> None:
    weight = torch.tensor(
        [
            [1.0, 0.25, -0.5],
            [0.5, -1.0, 2.0],
            [9.0, 9.0, 9.0],
        ],
        dtype=torch.bfloat16,
    )
    row = _semantic_row(weight, (0, 1))
    source = weight.index_select(0, torch.tensor([0, 1])).float().cpu()
    expected = source.mean(dim=0)
    expected *= source.norm(p=2, dim=1).mean() / expected.norm(p=2)

    assert SEMANTIC_INITIALIZATION_VERSION.endswith("cpu-fp32-v2")
    assert SEMANTIC_INITIALIZATION_REDUCTION_DEVICE == "cpu"
    assert SEMANTIC_INITIALIZATION_REDUCTION_DTYPE == "float32"
    assert row.device == weight.device
    assert row.dtype == weight.dtype
    torch.testing.assert_close(row, expected.to(dtype=weight.dtype))


def test_semantic_row_supports_mera_plain_mean_initialization() -> None:
    weight = torch.tensor(
        [[1.0, 2.0, 3.0], [3.0, 4.0, 7.0], [9.0, 9.0, 9.0]],
        dtype=torch.bfloat16,
    )
    row = _semantic_row(
        weight,
        (0, 1),
        initialization_mode="mean",
    )
    expected = weight[:2].float().mean(dim=0).to(dtype=torch.bfloat16)
    torch.testing.assert_close(row, expected)

    # The config-facing version string must distinguish MeRA's pure mean from
    # the existing norm-matched Semantic-ID initialization.
    assert MERA_MEAN_INITIALIZATION_VERSION == "label-token-mean-cpu-fp32-v1"


def test_register_rejects_reused_tokens(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.parquet"
    _mapping(mapping)
    tokenizer = FakeTokenizer()
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<interest_000>"]}
    )
    with pytest.raises(ValueError, match="tokenizer added"):
        register_semantic_interest_tokens(
            tokenizer,
            mapping,
            expected_count=2,
        )


def test_prepare_semantic_adapter_base_reconstructs_frozen_rows(
    tmp_path: Path,
) -> None:
    mapping = tmp_path / "mapping.parquet"
    _mapping(mapping)
    tokenizer = FakeTokenizer()
    spec = register_semantic_interest_tokens(
        tokenizer,
        mapping,
        expected_count=2,
    )
    first = FakeModel()
    second = FakeModel()
    with torch.no_grad():
        first.get_input_embeddings().weight[4:].fill_(91.0)
        first.get_output_embeddings().weight[4:].fill_(92.0)
        second.get_input_embeddings().weight[4:].fill_(-17.0)
        second.get_output_embeddings().weight[4:].fill_(-18.0)

    first_audit = prepare_semantic_adapter_base(first, spec)
    second_audit = prepare_semantic_adapter_base(second, spec)

    torch.testing.assert_close(
        first.get_input_embeddings().weight[4:],
        second.get_input_embeddings().weight[4:],
    )
    torch.testing.assert_close(
        first.get_output_embeddings().weight[4:],
        second.get_output_embeddings().weight[4:],
    )
    assert [
        row["input_initial_sha256"] for row in first_audit["tokens"]
    ] == [
        row["input_initial_sha256"] for row in second_audit["tokens"]
    ]
    assert [
        row["output_initial_sha256"] for row in first_audit["tokens"]
    ] == [
        row["output_initial_sha256"] for row in second_audit["tokens"]
    ]


def test_resolve_tied_peft_style_wrappers_uses_base_weights() -> None:
    shared = torch.nn.Embedding(6, 3)

    class TokenAdapter(torch.nn.Module):
        def __init__(self, base_layer) -> None:
            super().__init__()
            self.base_layer = base_layer

        def get_base_layer(self):
            return self.base_layer

    class Wrapper(torch.nn.Module):
        def __init__(self, base_layer) -> None:
            super().__init__()
            self.token_adapter = TokenAdapter(base_layer)

        @property
        def weight(self):
            # PEFT materializes an effective tensor with index_copy, so its
            # data pointer cannot be used to detect the underlying tie.
            return self.token_adapter.get_base_layer().weight.clone()

    class TiedConfig:
        tie_word_embeddings = True

    class TiedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_wrapper = Wrapper(shared)
            self.output_wrapper = Wrapper(shared)
            self.config = TiedConfig()

        def get_input_embeddings(self):
            return self.input_wrapper

        def get_output_embeddings(self):
            return self.output_wrapper

    modules = resolve_token_modules(TiedModel())
    assert modules["tied"] is True
    assert modules["input_module_name"] == "input_wrapper"
    assert modules["output_module_name"] == "output_wrapper"


def test_torch_numeric_backend_round_trip() -> None:
    original = capture_torch_numeric_backend(torch)
    changed = {
        "cuda_matmul_allow_tf32": not original["cuda_matmul_allow_tf32"],
        "cudnn_allow_tf32": not original["cudnn_allow_tf32"],
        "float32_matmul_precision": (
            "high"
            if original["float32_matmul_precision"] != "high"
            else "highest"
        ),
        "cuda_matmul_allow_bf16_reduced_precision_reduction": not original[
            "cuda_matmul_allow_bf16_reduced_precision_reduction"
        ],
        "cuda_matmul_allow_fp16_reduced_precision_reduction": not original[
            "cuda_matmul_allow_fp16_reduced_precision_reduction"
        ],
        "cudnn_deterministic": not original["cudnn_deterministic"],
        "cudnn_benchmark": not original["cudnn_benchmark"],
        "deterministic_algorithms": not original[
            "deterministic_algorithms"
        ],
        "deterministic_warn_only": False,
        "cublas_workspace_config": original["cublas_workspace_config"],
    }
    try:
        apply_torch_numeric_backend(torch, changed)
        assert capture_torch_numeric_backend(torch) == changed
    finally:
        apply_torch_numeric_backend(torch, original)


def test_deterministic_reload_numeric_backend_is_strict() -> None:
    backend = deterministic_reload_numeric_backend()
    assert backend == {
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_bf16_reduced_precision_reduction": False,
        "cuda_matmul_allow_fp16_reduced_precision_reduction": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "cublas_workspace_config": ":4096:8",
    }


def test_deterministic_sdpa_context_forces_math_and_restores() -> None:
    original = {
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
        "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
    }
    with deterministic_sdpa_math_context(torch):
        assert torch.backends.cuda.flash_sdp_enabled() is False
        assert torch.backends.cuda.math_sdp_enabled() is True
        assert torch.backends.cuda.mem_efficient_sdp_enabled() is False
        assert torch.backends.cuda.cudnn_sdp_enabled() is False
    assert {
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
        "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
    } == original


def test_torch_numeric_backend_rejects_missing_fields() -> None:
    with pytest.raises(ValueError, match="missing torch numeric backend"):
        apply_torch_numeric_backend(
            torch,
            {"cuda_matmul_allow_tf32": True},
        )
