from dataclasses import dataclass, field
from itertools import product

import pandas as pd
import torch
from sglang.srt.layers.logits_processor import (
    batched_gather_dot,
    fused_batched_gather_dot,
)
from transformers import HfArgumentParser
from triton.testing import do_bench_cudagraph

DEFAULT_INPUT_DIMS = [1024, 4096, 2560, 4096]
DEFAULT_OUTPUT_DIMS = [100352, 100352, 151936, 151936]
DEFAULT_INDEX_SIZES = [128, 256, 512, 1024, 2048, 4096, 8192]


@dataclass
class Arguments:
    output_path: str
    input_dims: list[int] = field(default_factory=lambda: DEFAULT_INPUT_DIMS)
    output_dims: list[int] = field(default_factory=lambda: DEFAULT_OUTPUT_DIMS)
    index_sizes: list[int] = field(default_factory=lambda: DEFAULT_INDEX_SIZES)
    batch_size: int = 10
    repetition_ms: int = 1000
    device: str = "cuda"
    dtype: str = "bfloat16"


@torch.inference_mode()
def benchmark_kernels(args: Arguments) -> None:
    dtype = getattr(torch, args.dtype)
    assert isinstance(dtype, torch.dtype)

    batch_size, device, repetition_ms = args.batch_size, args.device, args.repetition_ms
    configurations = zip(args.input_dims, args.output_dims)
    results = []

    for (input_dim, output_dim), index_size in product(
        configurations, args.index_sizes
    ):
        input = torch.randn((batch_size, input_dim), device=device, dtype=dtype)
        other = torch.randn((output_dim, input_dim), device=device, dtype=dtype)
        index = torch.randint(0, output_dim, (batch_size, index_size), device=device)
        torch.cuda.synchronize()

        # Warm-up and iterations are handled by do_bench_cudagraph.
        baseline_times = do_bench_cudagraph(
            lambda: batched_gather_dot(input, other, index),
            rep=repetition_ms,
            return_mode="all",
        )
        custom_times = do_bench_cudagraph(
            lambda: fused_batched_gather_dot(input, other, index),
            rep=repetition_ms,
            return_mode="all",
        )

        assert isinstance(baseline_times, list)
        assert isinstance(custom_times, list)

        results.extend(
            (input_dim, output_dim, index_size, baseline_ms, custom_ms)
            for baseline_ms, custom_ms in zip(baseline_times, custom_times)
        )

    df = pd.DataFrame(
        results,
        columns=["input_dim", "output_dim", "index_size", "baseline_ms", "custom_ms"],
    )
    df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)  # type: ignore
    (args,) = parser.parse_args_into_dataclasses()
    benchmark_kernels(args)
