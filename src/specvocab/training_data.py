import argparse
from enum import StrEnum
from typing import Any

from datasets import Dataset, load_dataset


class TrainingDataset(StrEnum):
    TULU_3_SFT_MIXTURE = "tulu-3-sft-mixture"
    ULTRACHAT = "ultrachat"

    def load(self, include_response: bool = True) -> Dataset:
        match self:
            case self.TULU_3_SFT_MIXTURE:
                dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train")

            case self.ULTRACHAT:
                dataset = load_dataset(
                    "HuggingFaceH4/ultrachat_200k", split="train_sft+train_gen"
                )

        if not isinstance(dataset, Dataset):
            raise ValueError("Expected `Dataset` object.")

        if not include_response:
            dataset = dataset.map(self._select_prompt)

        return dataset.select_columns("messages")

    def _select_prompt(self, example: dict[str, Any]) -> dict[str, Any]:
        return {"messages": [example["messages"][0]]}


def main(
    training_dataset: TrainingDataset,
    output_path: str,
    num_examples: int | None,
    seed: int,
) -> None:
    dataset = training_dataset.load()
    dataset = dataset.shuffle(seed=seed)
    print(f"Total examples: {len(dataset):,}")

    if num_examples is not None:
        dataset = dataset.select(range(num_examples))

    ids = list(range(len(dataset)))
    dataset = dataset.add_column("id", column=ids)  # type: ignore

    dataset.to_json(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare training data in the SpecForge format."
    )
    parser.add_argument(
        "training_dataset",
        type=TrainingDataset,
        choices=list(TrainingDataset),
        help="The dataset to extract examples from.",
    )
    parser.add_argument(
        "output_path", type=str, help="The output path for the .jsonl file."
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help="The number of examples to select, or `None` if unlimited.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="The random seed used for shuffling."
    )
    args = parser.parse_args()

    main(
        training_dataset=args.training_dataset,
        output_path=args.output_path,
        num_examples=args.num_examples,
        seed=args.seed,
    )
