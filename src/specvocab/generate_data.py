import argparse
import asyncio
from itertools import batched
from typing import Any

import torch
from datasets import Dataset
from sglang import Engine
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .training_data import TrainingDataset


def prepare_synthetic_example(example: dict[str, Any]) -> dict[str, list[int]]:
    input_ids, output_ids = example["input_ids"], example["output_ids"]
    attention_mask = [1] * (len(input_ids) + len(output_ids))
    loss_mask = [0] * len(input_ids) + [1] * len(output_ids)

    return {
        "input_ids": input_ids + output_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
    }


def prepare_example(
    example: dict[str, Any], tokenizer: PreTrainedTokenizerBase
) -> dict[str, list[int]]:
    # Workaround: `return_assistant_tokens_mask` is not supported by all models.
    # https://huggingface.co/docs/transformers/v4.52.3/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template.return_assistant_tokens_mask
    prompt = example["messages"][0]
    prompt_ids = tokenizer.apply_chat_template(
        [prompt],
        add_generation_prompt=True,
        enable_thinking=False,
    )

    input_ids = example["input_ids"]
    total_length = len(input_ids)
    prompt_length = len(prompt_ids)

    attention_mask = [1] * total_length
    loss_mask = [0] * prompt_length + [1] * (total_length - prompt_length)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
    }


async def _async_generate_all(
    engine: Engine,
    all_input_ids: list[list[int]],
    sampling_params: list[dict[str, Any]],
    external_batch_size: int,
) -> list[dict[str, Any]]:
    # Workaround for hang in SGLang. Potentially related:
    # https://github.com/sgl-project/sglang/issues/6778
    # https://github.com/sgl-project/sglang/issues/8463

    outputs = []
    for batch in batched(zip(all_input_ids, sampling_params), external_batch_size):
        batch_input_ids, batch_sampling_params = tuple(zip(*batch))

        # Workaround for error in SGLang:
        # `Received output for rid='[...]' but the state was deleted in TokenizerManager.`
        # Introduced in commit:
        # https://github.com/sgl-project/sglang/commit/eb7318f1c270a88b3c50a5e27fbf171fa4bad720

        batch_outputs = await asyncio.gather(
            *(
                engine.async_generate(
                    input_ids=input_ids, sampling_params=sampling_param
                )
                for input_ids, sampling_param in zip(
                    batch_input_ids, batch_sampling_params
                )
            )
        )
        outputs.extend(batch_outputs)

    return outputs


def load_dataset(
    training_dataset: TrainingDataset,
    num_examples: int | None,
    seed: int,
    synthesize: bool,
) -> Dataset:
    dataset = training_dataset.load(include_response=not synthesize)
    dataset = dataset.shuffle(seed=seed)
    print(f"Total examples: {len(dataset):,}")

    if num_examples is not None:
        dataset = dataset.select(range(num_examples))

    return dataset


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    date_string: str,
) -> Dataset:
    return dataset.map(
        lambda example: tokenizer.apply_chat_template(
            example["messages"],  # type: ignore
            add_generation_prompt=True,
            truncation=True,
            max_length=max_length,
            return_dict=True,
            date_string=date_string,
            enable_thinking=False,
        ),
        batched=True,
        desc="Tokenizing",
    )


def synthesize_responses(
    dataset: Dataset,
    model_path: str,
    max_length: int,
    external_batch_size: int,
    seed: int,
    log_level: str,
) -> Dataset:
    dp_size = torch.cuda.device_count()
    external_batch_size *= dp_size

    max_length += 1
    engine = Engine(
        model_path=model_path,
        context_length=max_length + 1,
        random_seed=seed,
        skip_tokenizer_init=True,
        dp_size=dp_size,
        log_level=log_level,
    )

    # https://github.com/sgl-project/sglang/blob/v0.5.1/python/sglang/srt/managers/tp_worker.py#L146
    max_request_length = max_length - 5

    # Skip examples beyond the maximum request length.
    dataset = dataset.filter(
        lambda example: len(example["input_ids"]) < max_request_length
    )

    all_input_ids = dataset["input_ids"]
    sampling_params = [
        {"max_new_tokens": max_length - len(input_ids), "temperature": 0.0}
        for input_ids in all_input_ids
    ]

    outputs = asyncio.run(
        _async_generate_all(engine, all_input_ids, sampling_params, external_batch_size)
    )
    engine.shutdown()

    all_output_ids = [output["output_ids"] for output in outputs]
    dataset = dataset.add_column("output_ids", all_output_ids)  # type: ignore

    return dataset.map(
        prepare_synthetic_example,
        remove_columns=dataset.column_names,
    )


def main(
    model_path: str,
    training_dataset: TrainingDataset,
    output_path: str,
    num_examples: int | None,
    synthesize: bool,
    max_length: int,
    external_batch_size: int,
    date_string: str,
    seed: int,
    log_level: str,
) -> None:
    dataset = load_dataset(training_dataset, num_examples, seed, synthesize)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = tokenize_dataset(dataset, tokenizer, max_length, date_string)

    if synthesize:
        dataset = synthesize_responses(
            dataset, model_path, max_length, external_batch_size, seed, log_level
        )
    else:
        dataset = dataset.map(
            prepare_example,
            remove_columns=dataset.column_names,
            fn_kwargs=dict(tokenizer=tokenizer),
        )

    dataset.to_parquet(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create data for draft model training, optionally with synthetic responses."
    )
    parser.add_argument(
        "model_path", type=str, help="The name of the model to tokenize data for."
    )
    parser.add_argument(
        "training_dataset",
        type=TrainingDataset,
        choices=list(TrainingDataset),
        help="The dataset to generate examples with.",
    )
    parser.add_argument(
        "output_path", type=str, help="The output path for the .parquet file."
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help="The number of examples to select, or `None` if unlimited.",
    )
    parser.add_argument(
        "--synthesize",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to generate synthetic responses using the model.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="The maximum length to use for generation.",
    )
    parser.add_argument(
        "--external_batch_size",
        type=int,
        default=2048,
        help="The amount of requests to send to SGLang at once. Workaround for a bug in SGLang.",
    )
    parser.add_argument(
        "--date_string",
        type=str,
        default="01 Aug 2025",
        help="The date string supplied to the model chat template.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="The random seed used for shuffling."
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="info",
        help="The logging level to use for SGLang.",
    )
    args = parser.parse_args()

    main(
        model_path=args.model_path,
        training_dataset=args.training_dataset,
        output_path=args.output_path,
        num_examples=args.num_examples,
        synthesize=args.synthesize,
        max_length=args.max_length,
        external_batch_size=args.external_batch_size,
        date_string=args.date_string,
        seed=args.seed,
        log_level=args.log_level,
    )
