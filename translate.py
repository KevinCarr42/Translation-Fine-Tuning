import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformers import BitsAndBytesConfig

BASE_MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
LORA_PATH = "mixtral-finetuned-enfr/checkpoint-30499"
DEVICE_MAP = "auto"
DTYPE = torch.float16

_tokenizer = None
_base_model = None
_finetuned_model = None
DEBUG = False


def load_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
        _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer


def load_base_model():
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
            revision="41bd4c9e7e4fb318ca40e721131d4933966c2cc1",  # not sure why this needs to be explicit
        )
        tokenizer = load_tokenizer()
        if len(tokenizer) > _base_model.config.vocab_size:
            _base_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    return _base_model


def load_finetuned_model():
    global _finetuned_model
    if _finetuned_model is None:
        tokenizer = load_tokenizer()
        base = load_base_model()
        if len(tokenizer) > base.config.vocab_size:
            base.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        _finetuned_model = PeftModel.from_pretrained(base, LORA_PATH)
    return _finetuned_model


def build_prompt(text, src_lang):
    if src_lang.lower().startswith("en"):
        instruction = "Translate the following English text to French. Output only the French translation, nothing else."
        prompt = f"[INST] {instruction}\n\nEnglish: {text}\n\nFrench: [/INST]"
    else:
        instruction = "Traduisez le texte français suivant en anglais. Ne donnez que la traduction anglaise, rien d'autre."
        prompt = f"[INST] {instruction}\n\nFrançais: {text}\n\nAnglais: [/INST]"

    return prompt


@torch.inference_mode()
def translate_text(input_text, input_language="en", finetuned=True):
    tokenizer = load_tokenizer()
    model = load_finetuned_model() if finetuned else load_base_model()

    prompt = build_prompt(input_text, input_language)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    stop_token_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("."),
        tokenizer.convert_tokens_to_ids("\n"),
    ]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        min_new_tokens=5,
        do_sample=False,
        num_beams=4,
        repetition_penalty=1.2,
        early_stopping=True,
        eos_token_id=stop_token_ids,  # Multiple stop tokens
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=3,  # Prevent 3-gram repetitions
    )[0]

    generated = tokenizer.decode(output_ids, skip_special_tokens=True)

    if DEBUG:
        print(f'Input text: {input_text}')
        print(f'Input language: {input_language}')
        print(f'Full generated: {generated}')

    # Extract only the translation part (after [/INST])
    if "[/INST]" in generated:
        translation = generated.split("[/INST]", 1)[1].strip()
    else:
        # Fallback if format is different
        translation = generated.replace(prompt, "").strip()

    # Clean up any remaining artifacts
    translation = translation.split("\n")[0].strip()  # Take only first line

    return translation


def main():
    parser = argparse.ArgumentParser(description="Translation tool")
    parser.add_argument("text", help="Text to translate")
    parser.add_argument("--lang", default="en", help="Source language (en or fr)")
    parser.add_argument("--base", action="store_true", help="Use base model instead of finetuned")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    global DEBUG
    DEBUG = args.debug

    translation = translate_text(args.text, args.lang, not args.base)
    print(translation)


if __name__ == "__main__":
    main()
