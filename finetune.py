import numpy as np
import torch

from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from sacrebleu import sacrebleu
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
                          DataCollatorForSeq2Seq, BitsAndBytesConfig)


# catch misconfigured environment
assert torch.cuda.is_available(), "CUDA GPU not found"

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
#  this is a fully open model
#    approx 165GB total storage
#    27GB VRAM (load_in_4bits=True) / 46GB (load_in_8bits=True)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_fast=True,
    trust_remote_code=True  # First run note:
                            #  watch the log for “Loaded MixtralForCausalLM”
                            #  if you see “LlamaForCausalLM” instead, the custom code wasn’t trusted.
)

tokenizer.pad_token = tokenizer.eos_token

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_compute_dtype=torch.float16,
)

sep_token = "<sep>"
if sep_token not in tokenizer.get_vocab():
    tokenizer.add_tokens([sep_token])

max_memory = {0: "30GiB", 1: "30GiB", "cpu": "64GiB"}  # 2 GB head-room

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map="auto",  # remove this line if using torchrun --nproc_per_node=2 (torchrun keeps getting vram errors)
    max_memory=max_memory,
    trust_remote_code=True,
)

model.config.pad_token_id = tokenizer.pad_token_id

if len(tokenizer) > model.config.vocab_size:
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

model = prepare_model_for_kbit_training(model, quant_config)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
)
model = get_peft_model(model, lora_config)

dataset = Dataset.from_json("training_data.jsonl")
split = dataset.train_test_split(test_size=0.1, seed=42)
train_data = split["train"]
eval_data = split["test"]

def tokenize(batch):
    sep = "<sep>"
    if sep not in batch["text"]:
        raise ValueError(f"Separator token '{sep}' not found in: {batch['text']}")

    source, target = batch["text"].split(sep)
    prompt = source.strip()
    answer = target.strip()

    full_text = f"{prompt} {sep} {answer}"
    tokenized = tokenizer(full_text, truncation=True, padding="longest", max_length=512)

    labels = tokenized["input_ids"].copy()
    sep_token_id = tokenizer.convert_tokens_to_ids(sep)
    sep_index = tokenized["input_ids"].index(sep_token_id) if sep_token_id in tokenized["input_ids"] else -1

    if sep_index == -1:
        raise ValueError("Separator token ID not found in tokenized input")

    tokenized["labels"] = [
        token if idx > sep_index and token != tokenizer.pad_token_id else -100
        for idx, token in enumerate(labels)
    ]

    return tokenized

train_data = train_data.map(tokenize)
eval_data = eval_data.map(tokenize)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    preds_text = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels_text = tokenizer.batch_decode(labels, skip_special_tokens=True)
    bleu = sacrebleu.corpus_bleu(preds_text, [labels_text]).score
    return {"bleu": bleu}

training_args = TrainingArguments(
    output_dir="mixtral-finetuned-enfr",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    bf16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=20,
    report_to="none",
    optim="paged_adamw_8bit",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model()
