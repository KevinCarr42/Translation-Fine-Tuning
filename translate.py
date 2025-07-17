import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformers import BitsAndBytesConfig


BASE_MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
LORA_PATH = "mixtral-finetuned-enfr/checkpoint-48604"
DEVICE_MAP = "auto"
DTYPE = torch.float16

_tokenizer = None
_base_model = None
_finetuned_model = None

DEBUG = False


def _load_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
        _tokenizer.pad_token = _tokenizer.eos_token
        if "<sep>" not in _tokenizer.get_vocab():
            _tokenizer.add_tokens(["<sep>"])
    return _tokenizer


def _load_base_model():
    global _base_model
    if _base_model is None:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf4",
            bnb_8bit_compute_dtype=DTYPE,
        )
        _base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=quant_cfg,
            device_map=DEVICE_MAP,
            trust_remote_code=True,
            local_files_only=True,
        )
        tokenizer = _load_tokenizer()
        if len(tokenizer) > _base_model.config.vocab_size:
            _base_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    return _base_model


def _load_finetuned_model():
    global _finetuned_model
    if _finetuned_model is None:
        tokenizer = _load_tokenizer()
        base = _load_base_model()

        if len(tokenizer) > base.config.vocab_size:
            base.resize_token_embeddings(len(tokenizer), mean_resizing=False)

        _finetuned_model = PeftModel.from_pretrained(base, LORA_PATH)
    return _finetuned_model


def _build_prompt(text, src_lang):
    sep = "<sep>"
    if src_lang.lower().startswith("en"):
        return f"Translate to French: {sep} {text}"
    return f"Traduire en anglais : {sep} {text}"


@torch.inference_mode()
def translate_text(input_text, input_language="en", finetuned=True):
    tokenizer = _load_tokenizer()
    model = _load_finetuned_model() if finetuned else _load_base_model()
    prompt = _build_prompt(input_text, input_language)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        num_beams=4,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )[0]
    generated = tokenizer.decode(output_ids, skip_special_tokens=True)

    if DEBUG:
        print(f'\tinput text: {input_text}')
        print(f'\t\tinput lang: {input_language}')
        print(f'\t\t\tgenerated {generated}')

    if "<sep>" in generated:
        return generated.split("<sep>", 1)[1].strip()
    return generated.strip()
