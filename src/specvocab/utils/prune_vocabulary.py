import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from specforge.data.preprocessing import process_token_dict_to_mappings
from specforge.modeling.target.target_head import TargetHead
from transformers import AutoConfig, HfArgumentParser


@dataclass(frozen=True)
class Arguments:
    target_model_path: str
    draft_model_path: str
    frequencies_path: str
    output_path: str
    vocab_size: int = 32000


def prepare_output_directory(output_path: str, draft_model_path: str) -> Path:
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(Path(draft_model_path), output_dir, dirs_exist_ok=True)
    return output_dir


def protect_eos_tokens(
    frequencies: np.ndarray, eos_token_id: int | list[int] | None
) -> np.ndarray:
    # Ensure EOS tokens are always included:
    # https://github.com/thunlp/FR-Spec/blob/29d0136b43d372d7d48806db8702cc9c813fdccf/fr/fr.py#L34
    eos_token_ids = [] if eos_token_id is None else eos_token_id

    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]

    for eos_token_id in eos_token_ids:
        frequencies[eos_token_id] = frequencies.max() + 1

    return frequencies


def compute_token_mappings(
    frequencies: np.ndarray, vocab_size: int, target_vocab_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    token_dict = Counter(dict(enumerate(frequencies)))
    d2t, t2d = process_token_dict_to_mappings(
        token_dict=token_dict,
        draft_vocab_size=vocab_size,
        target_vocab_size=target_vocab_size,
    )
    return d2t, t2d


def update_model_weights(
    output_path: Path,
    target_model_path: str,
    target_lm_head_key: str,
    draft_lm_head_key: str,
    d2t: torch.Tensor,
    t2d: torch.Tensor,
) -> None:
    target_head = TargetHead(target_model_path)
    target_head.load_weights(
        model_path=target_model_path,
        lm_head_key=target_lm_head_key,
    )

    weights_path = output_path / "model.safetensors"
    tensors = load_file(weights_path)

    tensors[draft_lm_head_key] = target_head.fc.weight[t2d]
    tensors.update({"d2t": d2t, "t2d": t2d})

    save_file(tensors, weights_path)


def update_model_config(output_path: str, vocab_size: int) -> None:
    draft_config = AutoConfig.from_pretrained(output_path)
    draft_config.draft_vocab_size = vocab_size
    draft_config.tie_word_embeddings = False
    draft_config.save_pretrained(output_path)


def main(args: Arguments) -> None:
    output_path = prepare_output_directory(args.output_path, args.draft_model_path)

    frequencies = np.loadtxt(args.frequencies_path, dtype=np.int64)
    target_config = AutoConfig.from_pretrained(args.target_model_path)
    draft_config = AutoConfig.from_pretrained(args.output_path)

    frequencies = protect_eos_tokens(frequencies, draft_config.eos_token_id)

    d2t, t2d = compute_token_mappings(
        frequencies=frequencies,
        vocab_size=args.vocab_size,
        target_vocab_size=target_config.vocab_size,
    )

    target_lm_head_key = (
        "model.embed_tokens.weight"
        if target_config.tie_word_embeddings
        else "lm_head.weight"
    )
    draft_lm_head_key = "lm_head.weight"

    update_model_weights(
        output_path=output_path,
        target_model_path=args.target_model_path,
        target_lm_head_key=target_lm_head_key,
        draft_lm_head_key=draft_lm_head_key,
        d2t=d2t,
        t2d=t2d,
    )

    update_model_config(args.output_path, args.vocab_size)


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)  # type: ignore
    (args,) = parser.parse_args_into_dataclasses()
    main(args)
