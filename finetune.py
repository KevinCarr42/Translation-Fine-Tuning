import sys, csv, os, json, argparse, logging, math, torch
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

DEBUG_KWARGS = {
    "which": "opus_mt_en_fr",
    "data": "training_data.jsonl",
    "output_dir": "runs",
    "epochs": 0.2,
    "lr": 2e-4,
    "batch_size": 8,
    "grad_accum": 2,
    "val_ratio": 0.05,
    "max_source_len": 512,
    "max_target_len": 512,
    "seed": 42,
    "bf16": True,
    "fp16": False,
    "no_qlora": False,
    "save_steps": 200,
    "eval_steps": 200,
    "logging_steps": 20,
    "warmup_ratio": 0.03,
    "device_map": "auto",
    "disable_tqdm": True,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", required=False, choices=list(MODELS.keys()))
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
    parser.add_argument("--config")
    parser.add_argument("--use_debug_defaults", action="store_true")

    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep_lr", nargs="+", type=float, default=[1e-4, 2e-4, 3e-4])
    parser.add_argument("--sweep_r", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--sweep_dropout", nargs="+", type=float, default=[0.05])
    parser.add_argument("--sweep_alpha", nargs="+", type=int)
    parser.add_argument("--sweep_alpha_factor", type=float, default=2.0)
    parser.add_argument("--sweep_max_steps", type=int, default=3000)
    parser.add_argument("--sweep_train_samples", type=int, default=200000)
    parser.add_argument("--sweep_eval_samples", type=int, default=10000)
    parser.add_argument("--sweep_name", default="sweep")
    argv = sys.argv[1:]
    args = parser.parse_args(argv if argv else [])
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            for k, v in json.load(f).items():
                if hasattr(args, k):
                    setattr(args, k, v)
    if args.use_debug_defaults or not argv:
        debug = {
            "which": "opus_mt_en_fr",
            "data": "training_data.jsonl",
            "output_dir": "runs",
            "epochs": 0.5,
            "lr": 2e-4,
            "batch_size": 8,
            "grad_accum": 2,
            "val_ratio": 0.05,
            "max_source_len": 512,
            "max_target_len": 512,
            "seed": 42,
            "bf16": True,
            "fp16": False,
            "no_qlora": False,
            "save_steps": 500,
            "eval_steps": 500,
            "logging_steps": 50,
            "warmup_ratio": 0.03,
            "device_map": "auto",
            "disable_tqdm": True
        }
        for k, v in debug.items():
            if hasattr(args, k) and getattr(args, k) == parser.get_default(k):
                setattr(args, k, v)
        if not args.which:
            args.which = debug["which"]
    return args

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

def build_trainer(args, model_info, tokenizer, model, dataset_processed, output_directory, lr, max_steps=None):
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_directory,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        num_train_epochs=0.0 if max_steps else args.epochs,
        max_steps=max_steps if max_steps else -1,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
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
        lr_scheduler_type="linear",
        weight_decay=0.01,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_processed["train"],
        eval_dataset=dataset_processed["eval"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    return trainer

def attach_lora(model, r, alpha, dropout):
    candidate_names = ["q", "k", "v", "o", "q_proj", "k_proj", "v_proj", "o_proj", "in_proj_weight"]
    detected = [name for name in candidate_names if any(hasattr(module, name) or name in type(module).__name__.lower() for _, module in model.named_modules())]
    lora_config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", task_type="SEQ_2_SEQ_LM", target_modules=list(set(detected)) or None)
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
    args = parse_args()
    model_info = MODELS[args.which]

    raw_dataset = load_dataset("json", data_files=args.data, split="train")
    split = raw_dataset.train_test_split(test_size=args.val_ratio, seed=args.seed)
    train_ds = split["train"].shuffle(seed=args.seed)
    eval_ds = split["test"].shuffle(seed=args.seed)
    if args.sweep:
        if args.sweep_train_samples and len(train_ds) > args.sweep_train_samples:
            train_ds = train_ds.select(range(args.sweep_train_samples))
        if args.sweep_eval_samples and len(eval_ds) > args.sweep_eval_samples:
            eval_ds = eval_ds.select(range(args.sweep_eval_samples))

    tokenizer, _ = load_tokenizer_and_model(model_info["model_id"], use_qlora=not args.no_qlora, use_bfloat16=args.bf16, device_map=args.device_map)

    def preprocess(ds):
        preprocessor = Preprocessor(
            model_name=args.which,
            tokenizer=tokenizer,
            language_map=model_info["language_map"],
            max_source_length=args.max_source_len,
            max_target_length=args.max_target_len,
            restrict_source_language=model_info.get("restrict_source_language"),
        )
        out = ds.map(preprocessor, remove_columns=ds.column_names)
        out = out.filter(lambda x: "input_ids" in x)
        return out

    train_proc = preprocess(train_ds)
    eval_proc = preprocess(eval_ds)
    dataset_processed = {"train": train_proc, "eval": eval_proc}

    if not args.sweep:
        output_directory = os.path.join(args.output_dir, args.which)
        setup_logging(output_directory)
        logging.info(f"config | model={args.which} qlora={not args.no_qlora} bf16={args.bf16} fp16={args.fp16 and not args.bf16} out={output_directory}")
        _, base_model = load_tokenizer_and_model(model_info["model_id"], use_qlora=not args.no_qlora, use_bfloat16=args.bf16, device_map=args.device_map)
        model = attach_lora(base_model, r=16, alpha=32, dropout=0.05)
        steps_per_epoch = math.ceil(len(dataset_processed["train"]) / (args.batch_size * args.grad_accum))
        logging.info(f"sizes | train={len(dataset_processed['train'])} eval={len(dataset_processed['eval'])} steps/epoch≈{steps_per_epoch}")
        trainer = build_trainer(args, model_info, tokenizer, model, dataset_processed, output_directory, lr=args.lr, max_steps=None)
        trainer.train()
        model.save_pretrained(os.path.join(output_directory, "lora"))
        tokenizer.save_pretrained(output_directory)
        with open(os.path.join(output_directory, "finished.json"), "w", encoding="utf-8") as f:
            json.dump({"status": "ok"}, f)
        return

    alphas = args.sweep_alpha if args.sweep_alpha else [int(args.sweep_alpha_factor * r) for r in args.sweep_r]
    combos = []
    for r in args.sweep_r:
        for dropout in args.sweep_dropout:
            for lr in args.sweep_lr:
                alpha = int(args.sweep_alpha_factor * r) if not args.sweep_alpha else None
                combos.append((lr, r, int(alpha if alpha else alphas[0]), dropout))

    sweep_root = os.path.join(args.output_dir, f"{args.which}_{args.sweep_name}")
    os.makedirs(sweep_root, exist_ok=True)
    results_csv = os.path.join(sweep_root, "sweep_results.csv")
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_dir", "lr", "lora_r", "lora_alpha", "lora_dropout", "train_samples", "eval_samples", "final_train_loss", "final_eval_loss"])

    for i, (lr, r, alpha, dropout) in enumerate(combos, 1):
        trial_dir = os.path.join(sweep_root, f"lr{lr}_r{r}_a{alpha}_d{dropout}")
        setup_logging(trial_dir)
        logging.info(f"trial {i}/{len(combos)} | lr={lr} r={r} alpha={alpha} dropout={dropout} steps={args.sweep_max_steps}")
        _, base_model = load_tokenizer_and_model(model_info["model_id"], use_qlora=not args.no_qlora, use_bfloat16=args.bf16, device_map=args.device_map)
        model = attach_lora(base_model, r=r, alpha=alpha, dropout=dropout)
        trainer = build_trainer(args, model_info, tokenizer, model, dataset_processed, trial_dir, lr=lr, max_steps=args.sweep_max_steps)
        train_result = trainer.train()
        metrics = train_result.metrics if train_result and hasattr(train_result, "metrics") else {}
        eval_metrics = trainer.evaluate()
        with open(os.path.join(trial_dir, "finished.json"), "w", encoding="utf-8") as f:
            json.dump({"status": "ok", "train_metrics": metrics, "eval_metrics": eval_metrics}, f)
        with open(results_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([trial_dir, lr, r, alpha, dropout, len(dataset_processed["train"]), len(dataset_processed["eval"]), metrics.get("train_loss", None), eval_metrics.get("eval_loss", None)])


if __name__ == "__main__":
    main()

"""
# TESTING HYPERPARAMS
nohup torchrun --nproc_per_node=2 finetune.py --which opus_mt_en_fr --sweep \
  --bf16 --no_qlora --disable_tqdm \
  --sweep_name opusenfr --sweep_lr 1e-4 2e-4 3e-4 --sweep_r 8 16 32 --sweep_dropout 0.05 \
  --sweep_max_steps 4000 --sweep_train_samples 200000 --sweep_eval_samples 10000 \
  --batch_size 16 --grad_accum 2 --eval_steps 500 --logging_steps 50 \
  --output_dir runs > runs/nohup_opusenfr_sweep.out 2>&1 &

nohup torchrun --nproc_per_node=2 finetune.py --which opus_mt_fr_en --sweep \
  --bf16 --no_qlora --disable_tqdm \
  --sweep_name opusfren --sweep_lr 1e-4 2e-4 3e-4 --sweep_r 8 16 32 --sweep_dropout 0.05 \
  --sweep_max_steps 4000 --sweep_train_samples 200000 --sweep_eval_samples 10000 \
  --batch_size 16 --grad_accum 2 --eval_steps 500 --logging_steps 50 \
  --output_dir runs > runs/nohup_opusfren_sweep.out 2>&1 &

nohup torchrun --nproc_per_node=2 finetune.py --which m2m100_418m --sweep \
  --bf16 --no_qlora --disable_tqdm \
  --sweep_name m2m --sweep_lr 7e-5 1e-4 2e-4 --sweep_r 8 16 32 --sweep_dropout 0.05 \
  --sweep_max_steps 4000 --sweep_train_samples 200000 --sweep_eval_samples 10000 \
  --batch_size 12 --grad_accum 2 --eval_steps 500 --logging_steps 50 \
  --output_dir runs > runs/nohup_m2m_sweep.out 2>&1 &

nohup torchrun --nproc_per_node=2 finetune.py --which mbart50_mmt --sweep \
  --bf16 --no_qlora --disable_tqdm \
  --sweep_name mbart --sweep_lr 7e-5 1e-4 1.5e-4 --sweep_r 16 32 --sweep_dropout 0.05 \
  --sweep_max_steps 4000 --sweep_train_samples 200000 --sweep_eval_samples 10000 \
  --batch_size 8 --grad_accum 2 --eval_steps 500 --logging_steps 50 \
  --output_dir runs > runs/nohup_mbart_sweep.out 2>&1 &
"""
