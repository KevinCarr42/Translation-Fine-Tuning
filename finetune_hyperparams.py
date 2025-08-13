import os, sys, json, argparse, logging, math, torch, csv
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          DataCollatorForSeq2Seq, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model


def is_distributed():
    return int(os.environ.get("WORLD_SIZE", 1)) > 1


MODELS = {
    "m2m100_418m": {
        "model_id": "facebook/m2m100_418M",
        "type": "seq2seq",
        "language_map": {"en": "en", "fr": "fr"}
    },
    "mbart50_mmt_fr": {
        "model_id": "facebook/mbart-large-50-many-to-many-mmt",
        "type": "seq2seq",
        "language_map": {"en": "en_XX", "fr": "fr_XX"},
        "restrict_source_language": "en"
    },
    "mbart50_mmt_en": {
        "model_id": "facebook/mbart-large-50-many-to-many-mmt",
        "type": "seq2seq",
        "language_map": {"en": "en_XX", "fr": "fr_XX"},
        "restrict_source_language": "fr"
    },
    "opus_mt_en_fr": {
        "model_id": "Helsinki-NLP/opus-mt-tc-big-en-fr",
        "type": "seq2seq",
        "language_map": {"en": "en", "fr": "fr"},
        "restrict_source_language": "en"
    },
    "opus_mt_fr_en": {
        "model_id": "Helsinki-NLP/opus-mt-tc-big-fr-en",
        "type": "seq2seq",
        "language_map": {"en": "en", "fr": "fr"},
        "restrict_source_language": "fr"
    },
}


def setup_logging(output_directory, to_file=True):
    os.makedirs(output_directory, exist_ok=True)
    handlers = [logging.StreamHandler()]
    if to_file:
        handlers.append(logging.FileHandler(os.path.join(output_directory, "console_output.txt"), encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=handlers)


def setup_tokenizer_languages(tokenizer, model_config, source_lang):
    if "language_map" not in model_config:
        return tokenizer

    lang_map = model_config["language_map"]

    target_lang = "fr" if source_lang == "en" else "en"

    if hasattr(tokenizer, 'src_lang'):
        tokenizer.src_lang = lang_map[source_lang]
    if hasattr(tokenizer, 'tgt_lang'):
        tokenizer.tgt_lang = lang_map[target_lang]

    return tokenizer


def load_tokenizer_and_model(model_id, use_qlora, use_bfloat16, device_map):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {"torch_dtype": torch.bfloat16 if use_bfloat16 else torch.float16, "trust_remote_code": True}

    if use_qlora:
        # QLoRA can use device_map even in distributed mode
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bfloat16 else torch.float16
        )
        if device_map is not None:
            model_kwargs["device_map"] = device_map
    else:
        # When not using QLoRA, only use device_map if explicitly provided and not None
        if device_map is not None:
            model_kwargs["device_map"] = device_map

    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **model_kwargs)
    if hasattr(model.config, "vocab_size") and len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.config.use_cache = False

    if use_qlora and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    return tokenizer, model


def attach_lora(model, r, alpha, dropout, use_qlora=True):
    names = ["q", "k", "v", "o", "q_proj", "k_proj", "v_proj", "o_proj", "in_proj_weight"]
    detected = [n for n in names if
                any(hasattr(m, n) or n in type(m).__name__.lower() for _, m in model.named_modules())]
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
        target_modules=list(set(detected)) or None
    )

    peft_model = get_peft_model(model, cfg)
    peft_model.train()
    peft_model.print_trainable_parameters()  # Ensure LoRA parameters require gradients

    return peft_model


class Preprocessor:
    def __init__(self, model_name, tokenizer, language_map, max_source_length, max_target_length,
                 restrict_source_language=None):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.language_map = language_map
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.restrict_source_language = restrict_source_language

    def _setup_tokenizer_languages(self, source_language, target_language):
        if not hasattr(self.tokenizer, 'src_lang'):
            return  # Tokenizer doesn't support language codes

        mapped_source = self.language_map.get(source_language, source_language)
        mapped_target = self.language_map.get(target_language, target_language)

        if self.model_name in ["m2m100_418m", "mbart50_mmt_fr", "mbart50_mmt_en"]:
            self.tokenizer.src_lang = mapped_source
            self.tokenizer.tgt_lang = mapped_target

    def __call__(self, example):
        if self.restrict_source_language and example["source_lang"] != self.restrict_source_language:
            return {}

        source_text = example["source"].strip()
        target_text = example["target"].strip()
        source_language = example["source_lang"]
        target_language = "en" if source_language == "fr" else "fr"

        if not target_text:
            return {}

        self._setup_tokenizer_languages(source_language, target_language)

        source_tokens = self.tokenizer(source_text, truncation=True, max_length=self.max_source_length)
        target_tokens = self.tokenizer(text_target=target_text, truncation=True, max_length=self.max_target_length)

        if not target_tokens.get("input_ids"):
            return {}

        source_tokens["labels"] = target_tokens["input_ids"]

        if self.model_name == "m2m100_418m":
            mapped_target = self.language_map[target_language]
            target_language_id = self.tokenizer.get_lang_id(mapped_target)
            pad_token_id = self.tokenizer.pad_token_id
            labels = source_tokens["labels"]

            decoder_input_ids = [target_language_id] + [
                (pad_token_id if token == -100 else token) for token in labels[:-1]
            ]
            source_tokens["decoder_input_ids"] = decoder_input_ids

        return source_tokens


class M2MDataCollator:
    def __init__(self, tokenizer, model, label_pad_token_id=-100):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id
        self.pad_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    def __call__(self, features):
        for f in features:
            f.pop("decoder_input_ids", None)

        batch = self.pad_collator(features)

        labels = batch["labels"]
        pad_id = self.tokenizer.pad_token_id
        labels_for_shift = torch.where(labels == -100, torch.tensor(pad_id, device=labels.device), labels)
        first_tok = labels_for_shift[:, :1]
        shifted = torch.cat([first_tok, labels_for_shift[:, :-1]], dim=1)
        batch["decoder_input_ids"] = shifted
        return batch


def build_trainer(args, tokenizer, model, dataset_processed, output_directory, lr, max_steps=None):
    if args.which == "m2m100_418m":
        data_collator = M2MDataCollator(tokenizer, model)
    else:
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    use_gradient_checkpointing = not args.no_qlora

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_directory,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        num_train_epochs=0.0 if max_steps else args.epochs,
        max_steps=max_steps if max_steps else -1,
        eval_strategy="steps",
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
        gradient_checkpointing=use_gradient_checkpointing,
        label_smoothing_factor=0.1,
        dataloader_num_workers=2,
        disable_tqdm=args.disable_tqdm,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        ddp_find_unused_parameters=False if is_distributed() else None,
        label_names=["labels"],
    )

    try:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset_processed["train"],
            eval_dataset=dataset_processed["eval"],
            processing_class=tokenizer,  # new API
            data_collator=data_collator,
        )
    except TypeError:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset_processed["train"],
            eval_dataset=dataset_processed["eval"],
            tokenizer=tokenizer,  # old API
            data_collator=data_collator,
        )
    return trainer


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--which", required=True, choices=list(MODELS.keys()))
    p.add_argument("--data", default="training_data.jsonl")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--max_source_len", type=int, default=512)
    p.add_argument("--max_target_len", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--no_qlora", action="store_true")
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--eval_steps", type=int, default=1000)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--device_map", default="auto")
    p.add_argument("--disable_tqdm", action="store_true")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--sweep_lr", nargs="+", type=float, default=[1e-4, 2e-4, 3e-4])
    p.add_argument("--sweep_r", nargs="+", type=int, default=[8, 16, 32])
    p.add_argument("--sweep_dropout", nargs="+", type=float, default=[0.05])
    p.add_argument("--sweep_alpha", nargs="+", type=int)
    p.add_argument("--sweep_alpha_factor", type=float, default=2.0)
    p.add_argument("--sweep_max_steps", type=int, default=4000)
    p.add_argument("--sweep_train_samples", type=int, default=200000)
    p.add_argument("--sweep_eval_samples", type=int, default=10000)
    p.add_argument("--sweep_name", default="sweep")
    return p.parse_args(argv)


def filter_dataset_by_model(dataset, model_key, model_config):
    if "restrict_source_language" not in model_config:
        return dataset

    allowed_lang = model_config["restrict_source_language"]
    return dataset.filter(lambda x: x["source_lang"] == allowed_lang)


def train_or_sweep(args):
    model_info = MODELS[args.which]
    raw = load_dataset("json", data_files=args.data, split="train")
    raw = filter_dataset_by_model(raw, args.which, MODELS[args.which])

    split = raw.train_test_split(test_size=args.val_ratio, seed=args.seed)
    train_ds = split["train"].shuffle(seed=args.seed)
    eval_ds = split["test"].shuffle(seed=args.seed)

    if args.sweep:
        if args.sweep_train_samples and len(train_ds) > args.sweep_train_samples:
            train_ds = train_ds.select(range(args.sweep_train_samples))
        if args.sweep_eval_samples and len(eval_ds) > args.sweep_eval_samples:
            eval_ds = eval_ds.select(range(args.sweep_eval_samples))

    if not args.no_qlora:
        device_map = args.device_map
    elif is_distributed():
        device_map = None
    else:
        device_map = args.device_map

    tokenizer, _ = load_tokenizer_and_model(
        model_info["model_id"],
        use_qlora=not args.no_qlora,
        use_bfloat16=args.bf16,
        device_map=device_map
    )

    def preprocess(ds):
        pre = Preprocessor(
            model_name=args.which,
            tokenizer=tokenizer,
            language_map=model_info["language_map"],
            max_source_length=args.max_source_len,
            max_target_length=args.max_target_len,
            restrict_source_language=model_info.get("restrict_source_language")
        )
        out = ds.map(pre, remove_columns=ds.column_names, load_from_cache_file=False)
        out = out.filter(
            lambda x: "input_ids" in x and "labels" in x and x["labels"] is not None and len(x["labels"]) > 0,
            load_from_cache_file=False)
        return out

    dataset_processed = {"train": preprocess(train_ds), "eval": preprocess(eval_ds)}

    if not args.sweep:
        output_directory = os.path.join(args.output_dir, args.which)
        setup_logging(output_directory)
        _, base = load_tokenizer_and_model(
            model_info["model_id"],
            use_qlora=not args.no_qlora,
            use_bfloat16=args.bf16,
            device_map=device_map
        )
        model = attach_lora(base, r=16, alpha=32, dropout=0.05, use_qlora=not args.no_qlora)
        steps_per_epoch = math.ceil(len(dataset_processed["train"]) / (args.batch_size * args.grad_accum))
        logging.info(
            f"sizes | train={len(dataset_processed['train'])} eval={len(dataset_processed['eval'])} steps/epoch≈{steps_per_epoch}")
        trainer = build_trainer(args, tokenizer, model, dataset_processed, output_directory, lr=args.lr, max_steps=None)
        trainer.train()
        model.save_pretrained(os.path.join(output_directory, "lora"))
        tokenizer.save_pretrained(output_directory)
        with open(os.path.join(output_directory, "finished.json"), "w", encoding="utf-8") as f:
            json.dump({"status": "ok"}, f)
        return

    alphas = args.sweep_alpha if args.sweep_alpha else [int(args.sweep_alpha_factor * r) for r in args.sweep_r]
    combos = [
        (lr, r, int(args.sweep_alpha_factor * r if not args.sweep_alpha else a), d)
        for r in args.sweep_r
        for d in args.sweep_dropout
        for lr in args.sweep_lr
        for a in (alphas if args.sweep_alpha else [None])
    ]

    sweep_root = os.path.join(args.output_dir, f"{args.which}_{args.sweep_name}")
    os.makedirs(sweep_root, exist_ok=True)
    results_csv = os.path.join(sweep_root, "sweep_results.csv")
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "trial_dir",
            "lr",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "train_samples",
            "eval_samples",
            "final_train_loss",
            "final_eval_loss"
        ])

    for i, (lr, r, alpha, dropout) in enumerate(combos, 1):
        trial_dir = os.path.join(sweep_root, f"lr{lr}_r{r}_a{alpha}_d{dropout}")
        setup_logging(trial_dir)
        _, base = load_tokenizer_and_model(model_info["model_id"], use_qlora=not args.no_qlora, use_bfloat16=args.bf16,
                                           device_map=device_map)
        model = attach_lora(base, r=r, alpha=alpha, dropout=dropout, use_qlora=not args.no_qlora)
        trainer = build_trainer(args, tokenizer, model, dataset_processed, trial_dir, lr=lr,
                                max_steps=args.sweep_max_steps)
        train_result = trainer.train()
        metrics = train_result.metrics if train_result and hasattr(train_result, "metrics") else {}
        eval_metrics = trainer.evaluate()
        with open(os.path.join(trial_dir, "finished.json"), "w", encoding="utf-8") as f:
            json.dump({"status": "ok", "train_metrics": metrics, "eval_metrics": eval_metrics}, f)
        with open(results_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [trial_dir,
                 lr,
                 r,
                 alpha,
                 dropout,
                 len(dataset_processed["train"]),
                 len(dataset_processed["eval"]),
                 metrics.get("train_loss", None),
                 eval_metrics.get("eval_loss", None)]
            )


def run_cli(argv=None):
    args = parse_args(argv)
    train_or_sweep(args)


if __name__ == "__main__":
    run_cli()
