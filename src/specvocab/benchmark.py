import atexit
import itertools
import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Generator

import sglang as sgl
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.lang.chat_template import (
    ChatTemplate,
    ChatTemplateStyle,
    register_chat_template,
    register_chat_template_matching_function,
)
from sglang.lang.interpreter import ProgramState
from sglang.utils import (
    download_and_cache_file,
    launch_server_cmd,
    read_jsonl,
    terminate_process,
    wait_for_server,
)
from transformers import HfArgumentParser

WARMUP_PROMPT = "Outline the plot of the Jane Austen novel 'Pride and Prejudice'."
OUTPUT_NAME_FORMAT = "response_{i}"

MTBENCH_CATEGORIES = {
    "writing",
    "roleplay",
    "reasoning",
    "math",
    "coding",
    "extraction",
    "stem",
    "humanities",
}

logger = logging.getLogger(__name__)


class Benchmark(StrEnum):
    MTBENCH = "mtbench"
    SPECBENCH = "specbench"

    def data_url(self) -> str:
        match self:
            case Benchmark.MTBENCH:
                return "https://raw.githubusercontent.com/lm-sys/FastChat/b494d0c6b4e7935f1764f8439e75da3e66beccc7/fastchat/llm_judge/data/mt_bench/question.jsonl"
            case Benchmark.SPECBENCH:
                return "https://raw.githubusercontent.com/hemingkx/Spec-Bench/66230f10cb0a02aced5ef3ce1e85163c16160454/data/spec_bench/question.jsonl"

    def load(
        self, num_examples: int | None, cache_path: str
    ) -> dict[str, list[list[str]]]:
        cache_dir = Path(cache_path)
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_file_path = download_and_cache_file(
            url=self.data_url(), filename=str(cache_dir / f"{self.value}.jsonl")
        )
        data = read_jsonl(cache_file_path)

        batches = defaultdict(list)
        for example in itertools.islice(data, num_examples):
            category, turns = example["category"], example["turns"]
            batches[self._normalize_category(category)].append(turns)

        return dict(batches)

    @staticmethod
    def _normalize_category(category: str) -> str:
        # MTBench is used in aggregate for speculative decoding benchmarks.
        return "multi_turn" if category in MTBENCH_CATEGORIES else category


@dataclass
class BenchmarkArguments:
    output_path: str
    benchmark_name: str = Benchmark.SPECBENCH
    num_examples: int | None = None
    batch_size: int | None = None
    temperature: float = 0.0
    max_new_tokens: int = 1024
    timeout: int = 30
    cache_path: str = "cache"


@dataclass
class SGLangArguments:
    model_path: str
    speculative_draft_model_path: str | None = None
    speculative_algorithm: str | None = None
    speculative_num_steps: int = 8
    speculative_eagle_topk: int = 10
    speculative_num_draft_tokens: int = 60
    speculative_vocabulary_num_candidates: int | None = None
    speculative_vocabulary_disable_kernel: bool = False
    mem_fraction_static: float = 0.9
    dtype: str | None = None
    random_seed: int = 42

    def create_launch_command(self, benchmark_args: BenchmarkArguments) -> str:
        args = {
            "--model-path": self.model_path,
            "--speculative-draft-model-path": self.speculative_draft_model_path,
            "--speculative-algorithm": self.speculative_algorithm,
            "--speculative-num-steps": self.speculative_num_steps,
            "--speculative-eagle-topk": self.speculative_eagle_topk,
            "--speculative-num-draft-tokens": self.speculative_num_draft_tokens,
            "--speculative-vocabulary-num-candidates": self.speculative_vocabulary_num_candidates,
            "--mem-fraction-static": self.mem_fraction_static,
            "--dtype": self.dtype,
            "--random-seed": self.random_seed,
        }
        flag_args = {
            "--speculative-vocabulary-disable-kernel": self.speculative_vocabulary_disable_kernel,
        }

        if benchmark_args.batch_size is not None:
            args |= {
                "--cuda-graph-max-bs": benchmark_args.batch_size,
                "--max-running-requests": benchmark_args.batch_size,
            }

        args = {k: v for k, v in args.items() if v is not None}

        launch_components = ["python -m sglang.launch_server"]
        launch_components.extend(f"{k} {v}" for k, v in args.items())
        launch_components.extend(f"{k}" for k, v in flag_args.items() if v)
        return " ".join(launch_components)


class Olmo2ChatTemplate(ChatTemplate):
    bos_token = "<|endoftext|>"
    eos_token = "<|endoftext|>"

    def get_prefix_and_suffix(
        self, role: str, hist_messages: list[dict[str, str]]
    ) -> tuple[str, str]:
        prefix, suffix = super().get_prefix_and_suffix(role, hist_messages)

        if not hist_messages:
            prefix = f"{self.bos_token}{prefix}"

        if role == "assistant":
            suffix = f"{self.eos_token}\n"

        return prefix, suffix


@register_chat_template_matching_function
def match_olmo2(model_path: str) -> str | None:
    if re.search(r"OLMo-2-[0-9]{4}-[0-9]+B-Instruct", model_path, re.IGNORECASE):
        return "olmo-2"


@register_chat_template_matching_function
def match_qwen3(model_path: str) -> str | None:
    if re.search(r"Qwen3-[0-9]+B(-A[0-9]+B){,1}$", model_path, re.IGNORECASE):
        return "qwen3"


def register_olmo2_chat_template() -> None:
    register_chat_template(
        Olmo2ChatTemplate(
            name="olmo-2",
            default_system_prompt=None,
            role_prefix_and_suffix={
                "system": ("<|system|>\n", "\n"),
                "user": ("<|user|>\n", "\n"),
                "assistant": ("<|assistant|>\n", ""),
            },
            style=ChatTemplateStyle.PLAIN,
            stop_str=("<|endoftext|>",),
        )
    )


def register_qwen3_chat_template() -> None:
    register_chat_template(
        ChatTemplate(
            name="qwen3",
            default_system_prompt=None,
            role_prefix_and_suffix={
                "system": ("<|im_start|>system\n", "<|im_end|>\n"),
                "user": ("<|im_start|>user\n", "<|im_end|>\n"),
                "assistant": (
                    "<|im_start|>assistant\n<think>\n\n</think>\n\n",
                    "<|im_end|>\n",
                ),
            },
            style=ChatTemplateStyle.PLAIN,
            stop_str=("<|im_end|>",),
        )
    )


@sgl.function
def conversation(s: ProgramState, messages: list[str]) -> None:
    for i, message in enumerate(messages):
        s += sgl.user(message)

        output_name = OUTPUT_NAME_FORMAT.format(i=i)
        s += sgl.assistant(sgl.gen(output_name))


def all_meta_info(state: ProgramState) -> Generator[dict[str, Any], None, None]:
    for i in itertools.count():
        output_name = OUTPUT_NAME_FORMAT.format(i=i)
        meta_info = state.get_meta_info(output_name)
        if meta_info is None:
            return

        yield meta_info


def sum_meta_info(states: list[ProgramState], key: str) -> int:
    return sum(
        meta_info[key]
        for state in states
        for meta_info in all_meta_info(state)
        if key in meta_info
    )


def compute_metrics(states: list[ProgramState], duration: float) -> dict[str, float]:
    output_token_count = sum_meta_info(states, "completion_tokens")
    throughput = output_token_count / duration

    verify_token_count = sum_meta_info(states, "spec_verify_ct")
    acceptance_length = (
        output_token_count / verify_token_count if verify_token_count > 0 else 1.0
    )

    return {
        "duration": duration,
        "throughput": throughput,
        "acceptance_length": acceptance_length,
    }


def run_benchmark(
    examples: list[list[str]],
    backend: RuntimeEndpoint,
    batch_size: int | None,
    max_new_tokens: int,
    temperature: float,
    progress_bar: bool = True,
    flush_cache: bool = True,
) -> dict[str, float]:
    if flush_cache:
        logger.info("Flushing cache...")
        backend.flush_cache()

    start = time.perf_counter()
    outputs = conversation.run_batch(  # type: ignore
        [{"messages": example} for example in examples],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        backend=backend,
        progress_bar=progress_bar,
        num_threads="auto" if batch_size is None else batch_size,
    )

    end = time.perf_counter()
    duration = end - start

    return compute_metrics(outputs, duration)


def main(sglang_args: SGLangArguments, benchmark_args: BenchmarkArguments) -> None:
    register_olmo2_chat_template()
    register_qwen3_chat_template()

    launch_command = sglang_args.create_launch_command(benchmark_args)
    process, port = launch_server_cmd(launch_command)
    atexit.register(terminate_process, process=process)

    logger.info("Loading benchmark data...")
    benchmark = Benchmark(benchmark_args.benchmark_name)
    batches = benchmark.load(
        num_examples=benchmark_args.num_examples, cache_path=benchmark_args.cache_path
    )

    server_url = f"http://localhost:{port}"
    logger.info(f"Waiting for server at {server_url}")
    wait_for_server(server_url, timeout=benchmark_args.timeout)

    backend = RuntimeEndpoint(server_url)

    logger.info("Starting warmup...")
    run_benchmark(
        examples=[[WARMUP_PROMPT]],
        backend=backend,
        batch_size=benchmark_args.batch_size,
        max_new_tokens=benchmark_args.max_new_tokens,
        temperature=benchmark_args.temperature,
        flush_cache=False,
    )
    logger.info("Warmup complete.")

    logger.info("Starting benchmark...")
    metrics = {
        category: run_benchmark(
            examples=batch,
            backend=backend,
            batch_size=benchmark_args.batch_size,
            max_new_tokens=benchmark_args.max_new_tokens,
            temperature=benchmark_args.temperature,
        )
        for category, batch in batches.items()
    }

    output_path = Path(benchmark_args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = json.dumps(metrics, indent=4, sort_keys=True)
    logger.info(output)

    output_path.write_text(output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = HfArgumentParser((SGLangArguments, BenchmarkArguments))  # type: ignore
    sglang_args, benchmark_args = parser.parse_args_into_dataclasses()
    main(sglang_args, benchmark_args)
