import os, sys, subprocess

DEFAULTS = {
    "m2m100_418m": {
        "batch_size": 12,
        "grad_accum": 2,
        "lr_grid": ["7e-5", "1e-4", "2e-4"],
        "r_grid": ["8", "16", "32"],
        "dropout_grid": ["0.05"]
    },
    "mbart50_mmt_fr": {
        "batch_size": 8,
        "grad_accum": 2,
        "lr_grid": ["7e-5", "1e-4", "1.5e-4"],
        "r_grid": ["16", "32"],
        "dropout_grid": ["0.05"]
    },
    "mbart50_mmt_en": {
        "batch_size": 8,
        "grad_accum": 2,
        "lr_grid": ["7e-5", "1e-4", "1.5e-4"],
        "r_grid": ["16", "32"],
        "dropout_grid": ["0.05"]
    },
    "opus_mt_en_fr": {
        "batch_size": 16,
        "grad_accum": 2,
        "lr_grid": ["1e-4", "2e-4", "3e-4"],
        "r_grid": ["8", "16", "32"],
        "dropout_grid": ["0.05"]
    },
    "opus_mt_fr_en": {
        "batch_size": 16,
        "grad_accum": 2,
        "lr_grid": ["1e-4", "2e-4", "3e-4"],
        "r_grid": ["8", "16", "32"],
        "dropout_grid": ["0.05"]
    },
}

def main():
    if len(sys.argv) < 2:
        print("usage: python launch_sweep.py <model_name> [runs_dir] [steps]")
        sys.exit(1)
    model = sys.argv[1]
    runs = sys.argv[2] if len(sys.argv) > 2 else "runs"
    steps = sys.argv[3] if len(sys.argv) > 3 else "4000"
    cfg = DEFAULTS[model]
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node=2", "finetune.py",
        "--which", model, "--sweep",
        "--bf16", "--no_qlora", "--disable_tqdm",
        "--sweep_name", "auto",
        "--sweep_lr", *cfg["lr_grid"],
        "--sweep_r", *cfg["r_grid"],
        "--sweep_dropout", *cfg["dropout_grid"],
        "--sweep_max_steps", steps,
        "--sweep_train_samples", "200000",
        "--sweep_eval_samples", "10000",
        "--batch_size", str(cfg["batch_size"]),
        "--grad_accum", str(cfg["grad_accum"]),
        "--eval_steps", "500",
        "--logging_steps", "50",
        "--output_dir", runs,
    ]
    os.makedirs(runs, exist_ok=True)
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
