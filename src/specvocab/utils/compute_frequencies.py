import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from datasets import Dataset, IterableDataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, PreTrainedTokenizerBase


@dataclass
class Arguments:
    model_path: str
    dataset_path: str
    output_path: str
    # https://github.com/thunlp/FR-Spec/blob/29d0136b43d372d7d48806db8702cc9c813fdccf/fr/fr.py#L63
    num_examples: int = 1000000
    timeout: int = 30


def select_target_tokens(example: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    loss_mask = example.pop("loss_mask").astype(np.bool)
    return {k: v[loss_mask] for k, v in example.items()}


def configure_timeouts(timeout: int) -> None:
    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(timeout)
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(timeout)


def load_parquet_dataset(dataset_path: str, num_examples: int) -> tuple[Dataset, int]:
    dataset = load_dataset("parquet", data_files=dataset_path, split="train")
    if not isinstance(dataset, Dataset):
        raise ValueError("Expected a `Dataset`.")

    num_examples = min(num_examples, len(dataset))
    dataset = dataset.take(num_examples)
    dataset = dataset.with_format("numpy")
    dataset = dataset.map(select_target_tokens)
    return dataset, num_examples


def load_streaming_dataset(
    dataset_path: str, tokenizer: PreTrainedTokenizerBase, num_examples: int
) -> IterableDataset:
    dataset = load_dataset(dataset_path, split="train", streaming=True)
    if not isinstance(dataset, IterableDataset):
        raise ValueError("Expected an `IterableDataset`.")

    dataset = dataset.take(num_examples)
    dataset = dataset.map(lambda example: tokenizer(example["text"]), batched=True)
    dataset = dataset.with_format("numpy")
    return dataset


def compute_frequencies(
    dataset: Dataset | IterableDataset, vocabulary_size: int, num_examples: int
) -> np.ndarray:
    return sum(
        (
            np.bincount(example["input_ids"], minlength=vocabulary_size)
            for example in tqdm(dataset, total=num_examples, desc="Counting")
        ),
        start=np.zeros(vocabulary_size, dtype=np.int64),
    )


def main(args: Arguments) -> None:
    configure_timeouts(args.timeout)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    vocabulary_size = len(tokenizer)

    if Path(args.dataset_path).is_file():
        dataset, num_examples = load_parquet_dataset(
            args.dataset_path, args.num_examples
        )
    else:
        num_examples = args.num_examples
        dataset = load_streaming_dataset(args.dataset_path, tokenizer, num_examples)

    frequencies = compute_frequencies(dataset, vocabulary_size, num_examples)
    np.savetxt(args.output_path, frequencies, fmt="%i")


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)  # type: ignore
    (args,) = parser.parse_args_into_dataclasses()
    main(args)
