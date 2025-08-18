import os
from finetune import finetune_model

MODELS = {
    "m2m100_418m": {
        "batch_size": 12,
        "grad_accum": 2,
        "lr": 2e-4,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "save_steps": 1000,
        "eval_steps": 1000,
    },
    "mbart50_mmt_fr": {
        "batch_size": 8,
        "grad_accum": 2,
        "lr": 1.5e-4,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "save_steps": 1000,
        "eval_steps": 1000,
    },
    "mbart50_mmt_en": {
        "batch_size": 8,
        "grad_accum": 2,
        "lr": 1.5e-4,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "save_steps": 1000,
        "eval_steps": 1000,
    },
    "opus_mt_en_fr": {
        "batch_size": 16,
        "grad_accum": 2,
        "lr": 3e-4,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "save_steps": 1000,
        "eval_steps": 1000,
    },
    "opus_mt_fr_en": {
        "batch_size": 16,
        "grad_accum": 2,
        "lr": 3e-4,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "save_steps": 1000,
        "eval_steps": 1000,
    },
}

TRAINING_FILE = "training_data.jsonl"
OUTPUT_ROOT = "outputs"
EPOCHS = 1.0
LOGGING_STEPS = 50
SEED = 42
WARMUP_RATIO = 0.03
VAL_RATIO = 0.05
MAX_SOURCE_LEN = 512
MAX_TARGET_LEN = 512
BF16 = True
FP16 = False
NO_QLORA = True
DEVICE_MAP = "auto"
DISABLE_TQDM = True


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    for model_name, cfg in MODELS.items():
        output_dir = os.path.join(OUTPUT_ROOT, model_name)
        os.makedirs(output_dir, exist_ok=True)
        finetune_model(
            which=model_name,
            data_path=TRAINING_FILE,
            output_directory=output_dir,
            learning_rate=float(cfg["lr"]),
            batch_size=int(cfg["batch_size"]),
            grad_accum=int(cfg["grad_accum"]),
            epochs=EPOCHS,
            eval_steps=int(cfg["eval_steps"]),
            logging_steps=LOGGING_STEPS,
            save_steps=int(cfg["save_steps"]),
            seed=SEED,
            warmup_ratio=WARMUP_RATIO,
            val_ratio=VAL_RATIO,
            max_source_len=MAX_SOURCE_LEN,
            max_target_len=MAX_TARGET_LEN,
            bf16=BF16,
            fp16=FP16,
            no_qlora=NO_QLORA,
            device_map=DEVICE_MAP,
            disable_tqdm=DISABLE_TQDM,
            lora_r=int(cfg["lora_r"]),
            lora_alpha=int(cfg["lora_alpha"]),
            lora_dropout=float(cfg["lora_dropout"]),
        )


if __name__ == "__main__":
    main()
