import os, json, argparse, logging, math, torch
from dataclasses import dataclass
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model

MODELS = {
    "m2m100_418m": {
        "model_id": "facebook/m2m100_418M",
        "type": "seq2seq",
        "language_map": {"en": "en", "fr": "fr"},
    },
    "mbart50_mmt": {
        "model_id": "facebook/mbart-large-50-many-to-many-mmt",
        "type": "seq2seq",
        "language_map": {"en": "en_XX", "fr": "fr_XX"},
    },
    "opus_mt_en_fr": {
        "model_id": "Helsinki-NLP/opus-mt-tc-big-en-fr",
        "type": "seq2seq",
        "language_map": {"en": "en", "fr": "fr"},
        "restrict_source_language": "en",
    },
    "opus_mt_fr_en": {
        "model_id": "Helsinki-NLP/opus-mt-tc-big-fr-en",
        "type": "seq2seq",
        "language_map": {"en": "en", "fr": "fr"},
        "restrict_source_language": "fr",
    },
}

def setup_logging(output_directory, to_file=True):
    os.makedirs(output_directory, exist_ok=True)
    handlers = [logging.StreamHandler()]
    if to_file:
        handlers.append(logging.FileHandler(os.path.join(output_directory, "console_output.txt"), encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=handlers)

def load_tokenizer_and_model(model_id, use_qlora, use_bfloat16, device_map):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if use_bfloat16 else torch.float16,
        "trust_remote_code": True,
    }
    if use_qlora:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bfloat16 else torch.float16,
        )
        model_kwargs["device_map"] = device_map
    else:
        if device_map != "auto":
            model_kwargs["device_map"] = device_map
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **model_kwargs)
    if hasattr(model.config, "vocab_size") and len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.config.use_cache = False
    return tokenizer, model

def attach_lora(model):
    candidate_names = ["q", "k", "v", "o", "q_proj", "k_proj", "v_proj", "o_proj", "in_proj_weight"]
    detected = [name for name in candidate_names if any(hasattr(module, name) or name in type(module).__name__.lower() for _, module in model.named_modules())]
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
        target_modules=list(set(detected)) or None,
    )
    return get_peft_model(model, lora_config)

@dataclass
class Preprocessor:
    model_name
    tokenizer
    language_map
    max_source_length
    max_target_length
    restrict_source_language=None
    def __call__(self, example):
        if self.restrict_source_language and example["source_lang"] != self.restrict_source_language:
            return {}
        source_text = example["source"].strip()
        target_text = example["target"].strip()
        source_language = example["source_lang"]
        if self.model_name in {"mbart50_mmt", "m2m100_418m"}:
            self.tokenizer.src_lang = self.language_map[source_language]
        inputs = self.tokenizer(source_text, truncation=True, max_length=self.max_source_length)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(target_text, truncation=True, max_length=self.max_target_length)
        inputs["labels"] = labels["input_ids"]
        return inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--data", default="training_data.jsonl")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--max_source_len", type=int, default=512)
    parser.add_argument("--max_target_len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no_qlora", action="store_true")
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--disable_tqdm", action="store_true")
    args = parser.parse_args()

    model_info = MODELS[args.which]
    output_directory = os.path.join(args.output_dir, args.which)
    setup_logging(output_directory)
    logging.info(f"config | model={args.which} qlora={not args.no_qlora} bf16={args.bf16} fp16={args.fp16 and not args.bf16} out={output_directory}")

    tokenizer, base_model = load_tokenizer_and_model(
        model_info["model_id"],
        use_qlora=not args.no_qlora,
        use_bfloat16=args.bf16,
        device_map=args.device_map,
    )
    model = attach_lora(base_model)

    raw_dataset = load_dataset("json", data_files=args.data, split="train")
    split = raw_dataset.train_test_split(test_size=args.val_ratio, seed=args.seed)

    preprocessor = Preprocessor(
        model_name=args.which,
        tokenizer=tokenizer,
        language_map=model_info["language_map"],
        max_source_length=args.max_source_len,
        max_target_length=args.max_target_len,
        restrict_source_language=model_info.get("restrict_source_language"),
    )

    dataset_processed = {
        "train": split["train"].map(preprocessor, remove_columns=split["train"].column_names),
        "eval": split["test"].map(preprocessor, remove_columns=split["test"].column_names),
    }
    dataset_processed["train"] = dataset_processed["train"].filter(lambda x: "input_ids" in x)
    dataset_processed["eval"] = dataset_processed["eval"].filter(lambda x: "input_ids" in x)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    steps_per_epoch = math.ceil(len(dataset_processed["train"]) / (args.batch_size * args.grad_accum))
    save_total_limit = 3

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_directory,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=save_total_limit,
        predict_with_generate=False,
        report_to=["none"],
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        seed=args.seed,
        warmup_ratio=args.warmup_ratio,
        gradient_checkpointing=True,
        label_smoothing_factor=0.1,
        dataloader_num_workers=2,
        disable_tqdm=args.disable_tqdm,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_processed["train"],
        eval_dataset=dataset_processed["eval"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logging.info(f"sizes | train={len(dataset_processed['train'])} eval={len(dataset_processed['eval'])} steps/epoch≈{steps_per_epoch}")
    trainer.train()
    model.save_pretrained(os.path.join(output_directory, "lora"))
    tokenizer.save_pretrained(output_directory)
    with open(os.path.join(output_directory, "finished.json"), "w", encoding="utf-8") as f:
        json.dump({"status": "ok"}, f)

if __name__ == "__main__":
    main()
