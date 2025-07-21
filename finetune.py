import numpy as np
import torch

from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
                          DataCollatorForSeq2Seq, BitsAndBytesConfig)

# catch misconfigured environment
assert torch.cuda.is_available(), "CUDA GPU not found"

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_fast=True,
    trust_remote_code=True
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

max_memory = {0: "30GiB", 1: "30GiB", "cpu": "64GiB"}

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map="auto",
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


def build_prompt(source_text, target_text, source_lang="en"):
    if source_lang.lower().startswith("en"):
        instruction = "Translate the following English text to French. Output only the French translation, nothing else."
        prompt = f"[INST] {instruction}\n\nEnglish: {source_text}\n\nFrench: [/INST]"
    else:
        instruction = "Traduisez le texte français suivant en anglais. Ne donnez que la traduction anglaise, rien d'autre."
        prompt = f"[INST] {instruction}\n\nFrançais: {source_text}\n\nAnglais: [/INST]"

    # For training, we append the target after the prompt
    full_text = f"{prompt} {target_text}"

    return prompt, full_text


def tokenize(batch):
    source_text = batch.get("source", "")
    target_text = batch.get("target", "")
    source_lang = batch.get("source_lang", "en")

    prompt, full_text = build_prompt(source_text, target_text, source_lang)
    tokenized = tokenizer(full_text, truncation=True, padding="max_length", max_length=512)
    prompt_tokens = tokenizer(prompt, truncation=True, max_length=512)["input_ids"]
    prompt_length = len(prompt_tokens)

    labels = tokenized["input_ids"].copy()
    # Set prompt tokens to -100 so they're ignored in loss calculation
    for i in range(prompt_length):
        labels[i] = -100

    # Also set padding tokens to -100
    labels = [token if token != tokenizer.pad_token_id else -100 for token in labels]

    tokenized["labels"] = labels

    return tokenized


train_data = train_data.map(tokenize)
eval_data = eval_data.map(tokenize)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


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
    eval_accumulation_steps=1,
    eval_do_concat_batches=False,
    per_device_eval_batch_size=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

trainer.save_model()
tokenizer.save_pretrained(trainer.args.output_dir)

trainer.save_state()