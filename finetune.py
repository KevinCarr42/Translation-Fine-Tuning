import torch

from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
                          DataCollatorForSeq2Seq, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# catch misconfigured environment
assert torch.cuda.is_available(), "CUDA GPU not found"

# TODO: this is open weights for research, but we need to contact Unbabel if we want to deploy
#  approx 55GB storage (probably 2x), 18GB VRAM
model_id = "Unbabel/TowerInstruct-13B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

quant_config = BitsAndBytesConfig(
    load_in_8bit=True,  # or load_in_4bit=True
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map="auto"
)
model.config.pad_token_id = tokenizer.pad_token_id

if "<sep>" not in tokenizer.get_vocab():
    tokenizer.add_tokens(["<sep>"])
    model.resize_token_embeddings(len(tokenizer))

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    inference_mode=False
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
    tokenized = tokenizer(full_text, truncation=True, padding="max_length", max_length=512)

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

training_args = TrainingArguments(
    output_dir="towerinstruct-finetuned-enfr",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    fp16=True,
    save_total_limit=2,
    logging_steps=20,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
