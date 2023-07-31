import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24"

from data import DailyDialog
import transformers


from dataclasses import dataclass, field
import math
from typing import Dict, Optional


@dataclass
class Arguments:
    model_name_or_path: Optional[str] = field(default="roberta-base")
    # model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    data_cache_dir: Optional[str] = field(default="./.cache/dataset")
    training_cache_dir: Optional[str] = field(default="./.cache/training")
    penalty_alpha: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    temperature: Optional[float] = field(default=1.0)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    diversity_penalty: Optional[float] = field(default=0.0)


def test():
    parser = transformers.HfArgumentParser(
        (Arguments,)
    )
    args, = parser.parse_args_into_dataclasses()
  
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        is_decoder=True,
    )
    model.config.use_cache = False

    # Set RoPE scaling factor
    orig_ctx_len = getattr(model.config, "max_position_embeddings", None)
    if orig_ctx_len and args.model_max_length > orig_ctx_len:
        scaling_factor = math.ceil(args.model_max_length / orig_ctx_len)
        model.config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.training_cache_dir,
        model_max_length=args.model_max_length,
        truncation_side="left",
        padding=False,
        use_fast=False,
    )

    generator = transformers.pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        min_length=20,
        max_length=args.model_max_length, 
        penalty_alpha=args.penalty_alpha,
        repetition_penalty=args.repetition_penalty,
        top_k=args.top_k,
        temperature=args.temperature,
        num_beams=args.num_beams,
        num_beam_groups=args.num_beam_groups,
        diversity_penalty=args.diversity_penalty,
        early_stopping=False,
        device="cuda"
    )

    history = []
    while True:
        user = input("user: ")
        if not user:
            break
        inputs = f" {tokenizer.eos_token} ".join(history) + f" {tokenizer.eos_token} " if len(history) > 0 else ""
        inputs += f"{user} {tokenizer.eos_token} response:"
        bot = generator(user)[0]["generated_text"]
        history.extend([user, bot])
        print("agent:", bot)

if __name__ == "__main__":
  test()