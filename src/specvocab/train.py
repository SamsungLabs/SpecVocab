from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Iterable, Iterator

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.tracking import LoggerType
from accelerate.utils import set_seed, tqdm
from datasets import Dataset, load_dataset
from specforge.core.eagle3 import OnlineEagle3Model
from specforge.data.preprocessing import generate_vocab_mapping_file
from specforge.modeling.auto import AutoDraftModelConfig, AutoEagle3DraftModel
from specforge.modeling.target import HFEagle3TargetModel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, HfArgumentParser
from transformers.optimization import get_cosine_schedule_with_warmup


@dataclass(frozen=True)
class TrainingArguments:
    target_model_path: str
    draft_model_path: str
    data_path: str
    output_path: str
    train_steps: int = 1024
    warmup_ratio: float = 0.015
    per_device_batch_size: int = 1
    learning_rate: float = 5e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    loss_weight: float = 0.8
    aux_loss_weight: float = 0.1
    max_grad_norm: float = 0.5
    attention_backend: str = "flex_attention"
    mixed_precision: str | None = "bf16"
    ttt_length: int = 7
    report_to: LoggerType | None = None
    logging_steps: int = 50
    seed: int = 42


class DataCollator:
    def __call__(
        self, examples: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        return {
            feature: torch.nn.utils.rnn.pad_sequence(
                [example[feature] for example in examples], batch_first=True
            )
            for feature in examples[0].keys()
        }


# https://github.com/pytorch/pytorch/issues/23900
def cycle(iterable: Iterable) -> Iterator:
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def _prepare_log_values(
    values: dict[str, torch.Tensor | float | None], accelerator: Accelerator
) -> dict[str, float]:
    float_values = {k: v for k, v in values.items() if isinstance(v, float)}

    # Detach and reduce tensors across all ranks.
    tensor_values = {
        k: v.detach().clone() for k, v in values.items() if isinstance(v, torch.Tensor)
    }
    tensor_values: dict[str, torch.Tensor] = dict(
        accelerator.reduce(tensor_values, reduction="mean")
    )

    # Handle both scalar and vector tensors.
    float_values |= {k: v.item() for k, v in tensor_values.items() if v.ndim == 0}
    float_values |= {
        k.format(i=i): x
        for k, v in tensor_values.items()
        if v.ndim > 0
        for i, x in enumerate(v.tolist())
    }

    return float_values


def _register_shared_weight_removal_hook(
    model: torch.nn.Module, target_model: torch.nn.Module
) -> None:
    target_pointers = {p.data_ptr() for p in target_model.parameters()}
    shared_keys = [
        name
        for name, tensor in model.state_dict(keep_vars=True).items()
        if isinstance(tensor, torch.Tensor) and tensor.data_ptr() in target_pointers
    ]

    def _remove_shared_weights(
        module: torch.nn.Module,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict,
    ) -> None:
        for key in shared_keys:
            del state_dict[key]

    model.register_state_dict_post_hook(_remove_shared_weights)


def main(args: TrainingArguments) -> None:
    set_seed(args.seed)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(1800))],
    )
    accelerator.init_trackers("", config=asdict(args))

    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model_path, dtype="auto", device_map=accelerator.device
    )

    eagle3_target_model = HFEagle3TargetModel(target_model)
    eagle3_target_model.set_aux_hidden_states_layers()

    # Disable gradients for the target model.
    for p in target_model.parameters():
        p.requires_grad = False

    is_new_model = args.draft_model_path.endswith(".json")
    if is_new_model:
        draft_config = AutoDraftModelConfig.from_file(args.draft_model_path)
        draft_model = AutoEagle3DraftModel.from_config(
            draft_config, attention_backend=args.attention_backend
        )
    else:
        draft_model = AutoEagle3DraftModel.from_pretrained(
            args.draft_model_path, attention_backend=args.attention_backend
        )
        draft_config = draft_model.config

    draft_config.dtype = target_model.dtype

    # Re-seed after initializing the models. This is necessary as the number of
    # parameters in the draft model can vary, affecting the global random state.
    set_seed(args.seed)

    # SpecForge requires the input embeddings as part of the draft model.
    draft_model.embed_tokens = target_model.get_input_embeddings()

    draft_vocab_size = getattr(draft_config, "draft_vocab_size", None)
    if draft_vocab_size is None:
        draft_model.lm_head = target_model.get_output_embeddings()

    # Avoid saving target model weights within the draft model.
    _register_shared_weight_removal_hook(draft_model, target_model)

    draft_model.train()
    draft_model.to(accelerator.device)

    dataset = load_dataset("parquet", data_files=args.data_path, split="train")
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected `Dataset`, not `{type(dataset)}`")

    dataset = dataset.with_format("torch")
    data_loader = DataLoader(
        dataset=dataset,  # type: ignore
        batch_size=args.per_device_batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=DataCollator(),
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    if is_new_model and draft_vocab_size is not None:
        with accelerator.main_process_first():
            vocab_mapping_path = generate_vocab_mapping_file(
                dataset=dataset,
                target_vocab_size=draft_config.vocab_size,
                draft_vocab_size=draft_config.draft_vocab_size,
            )
            draft_model.load_vocab_mapping(vocab_mapping_path)

    eagle3_model = OnlineEagle3Model(
        draft_model=draft_model,
        length=args.ttt_length,
        attention_backend=args.attention_backend,
        return_structured_output=True,
    )
    optimizer = AdamW(
        params=draft_model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        fused=True,
    )

    warmup_steps = int(args.warmup_ratio * args.train_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps * accelerator.num_processes,
        num_training_steps=args.train_steps * accelerator.num_processes,
    )

    eagle3_model, optimizer, data_loader, scheduler = accelerator.prepare(
        eagle3_model, optimizer, data_loader, scheduler
    )
    data_iterator = cycle(data_loader)
    loss_weights = args.loss_weight ** torch.arange(
        args.ttt_length, device=draft_model.device
    )

    pbar = tqdm(range(args.train_steps), desc="Training")
    for step in pbar:
        batch = next(data_iterator)

        output = eagle3_target_model.generate_eagle3_data(**batch)
        output = eagle3_model(
            input_ids=output.input_ids,
            attention_mask=output.attention_mask,
            loss_mask=output.loss_mask,
            target=output.target,
            hidden_states=output.hidden_states,
        )

        losses = output.losses * loss_weights
        loss = losses.sum()

        aux_losses = None
        aux_loss = None
        if output.aux_losses is not None:
            aux_losses = output.aux_losses * loss_weights
            aux_loss = aux_losses.sum()
            loss += args.aux_loss_weight * aux_loss

        if step % args.logging_steps == 0:
            values = _prepare_log_values(
                {
                    "train/acc": output.accuracies.mean(),
                    "train/acc_{i}": output.accuracies,
                    "train/loss": loss,
                    "train/loss_{i}": losses,
                    "train/aux_loss": aux_loss,
                    "train/aux_loss_{i}": aux_losses,
                    "train/lr": scheduler.get_last_lr()[0],
                },
                accelerator=accelerator,
            )
            accelerator.log(values, step)

            formatted_loss = "{:.2f}".format(values["train/loss"])
            pbar.set_postfix(loss=formatted_loss)

        accelerator.backward(loss)
        accelerator.clip_grad_norm_(draft_model.parameters(), args.max_grad_norm)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    accelerator.wait_for_everyone()
    draft_model.save_pretrained(
        args.output_path,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )

    accelerator.end_training()


if __name__ == "__main__":
    parser = HfArgumentParser(TrainingArguments)  # type: ignore
    (args,) = parser.parse_args_into_dataclasses()
    main(args)
