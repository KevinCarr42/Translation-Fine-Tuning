from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, Dataset
import torch


# Model and tokenizer
model_id = "Unbabel/TowerInstruct-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto")

# PEFT / QLoRA config
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    inference_mode=False
)
model = get_peft_model(model, lora_config)

# Load your dataset (replace with your own)
# Format: {"text": "translate English to French: The ocean is blue. <sep> L’océan est bleu."}
train_data = Dataset.from_json("your_dataset_train.json")
eval_data = Dataset.from_json("your_dataset_eval.json")

def tokenize(batch):
    sep = "<sep>"
    inputs = tokenizer(batch["text"].split(sep)[0], truncation=True, padding="max_length", max_length=512)
    targets = tokenizer(batch["text"].split(sep)[1], truncation=True, padding="max_length", max_length=512)
    inputs["labels"] = targets["input_ids"]
    return inputs

train_data = train_data.map(tokenize)
eval_data = eval_data.map(tokenize)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Training
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

# trainer.train()
