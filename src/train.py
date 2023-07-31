import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24"

from data import DailyDialog
import transformers
import evaluate
import torch
import datasets


from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Optional, List, Dict

import numpy as np
import torch
import transformers
from transformers import Seq2SeqTrainer
from transformers.trainer_pt_utils import LabelSmoother
from models import RobertaForCausalLMwithParallelStateTracking, OPTForCausalLMwithParallelStateTracking

IGNORE_TOKEN_ID = LabelSmoother.ignore_index        


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-350m")
    # model_name_or_path: Optional[str] = field(default="roberta-base")
    # model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default="./ijcnlp_dailydialog", metadata={"help": "Path to the training data."}
    )
    dataset_cache_dir: Optional[str] = field(default="./.cache/dataset")


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default="./.cache/training")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


local_rank = None 

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def train():
    def tokenize_function(examples):
        tokenize = lambda text: tokenizer(text, truncation=True, padding="max_length", return_tensors="pt")["input_ids"]
        examples["input_ids"] = [f" {tokenizer.eos_token} ".join(example[:-1]) for example in examples['dialog']]
        examples["input_ids"] = tokenize([f"{example} {tokenizer.eos_token} response:" for example in examples['input_ids']])
        examples["labels"] = tokenize([f"{example[-1]}" for example in examples['dialog']])
        max_uttr_len = max([len(example) for example in examples['dialog']] + [tokenizer.model_max_length])
        pad_dialog = lambda example: example + [-100] * (max_uttr_len-len(example))
        examples["act_state_labels"] = torch.LongTensor([pad_dialog(example) for example in examples["act"]])
        examples["emotion_state_labels"] = torch.LongTensor([pad_dialog(example) for example in examples["emotion"]])
        return examples

    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    if "opt" in model_args.model_name_or_path.lower():
        model = OPTForCausalLMwithParallelStateTracking
    elif "roberta" in model_args.model_name_or_path.lower():
        model = RobertaForCausalLMwithParallelStateTracking
    else:
        model = transformers.AutoModelForCausalLM

    model = model.from_pretrained(
        model_args.model_name_or_path, 
        is_decoder=True,
    )
    model.config.use_cache = False

    # Set RoPE scaling factor
    orig_ctx_len = getattr(model.config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = math.ceil(training_args.model_max_length / orig_ctx_len)
        model.config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        use_fast=False,
        truncation_side="left",
    )

    data = DailyDialog(data_args.data_path, cache_dir=data_args.dataset_cache_dir,).prepare()
    data.pop("test")
    data = data.map(tokenize_function, batched=True)
    data["validation"] = data["validation"].remove_columns(["act_state_labels", "emotion_state_labels"])

    training_args.label_names = ["labels"]
    trainer = Seq2SeqTrainer(
        model=model, tokenizer=tokenizer, args=training_args, 
        train_dataset = data["train"].with_format("torch"),
        eval_dataset = data["validation"].with_format("torch"),
        # callbacks=[transformers.EarlyStoppingCallback(5, 0.01)]
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    with torch.no_grad():
        model.eval()
        print(trainer.evaluate())

    model.config.use_cache = True
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)    
  

if __name__ == "__main__":
  train()