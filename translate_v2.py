import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformers import BitsAndBytesConfig
import logging


DEFAULT_CONFIG = {
    "base_model_id": "facebook/nllb-200-3.3B",
    "lora_path": None,
    "device_map": "auto",
    "dtype": torch.bfloat16,
    "local_files_only": False,
    "revision": None,
    "use_quantization": False,
    "debug": False,
    "max_memory": {0: "30GB", 1: "30GB"},
    "model_type": "seq2seq"
}


class TranslationModel:
    def __init__(self, config=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self._tokenizer = None
        self._base_model = None
        self._finetuned_model = None

        self.logger = logging.getLogger(__name__)
        if self.config["debug"]:
            logging.basicConfig(level=logging.DEBUG)

    def load_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config["base_model_id"],
                use_fast=True,
                local_files_only=self.config["local_files_only"]
            )
            if hasattr(self._tokenizer, 'pad_token') and self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def load_base_model(self):
        if self._base_model is None:
            model_kwargs = {
                "device_map": self.config["device_map"],
                "trust_remote_code": True,
                "local_files_only": self.config["local_files_only"],
                "torch_dtype": self.config["dtype"]
            }

            if self.config["max_memory"]:
                model_kwargs["max_memory"] = self.config["max_memory"]

            if self.config["use_quantization"]:
                quant_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self.config["dtype"],
                )
                model_kwargs["quantization_config"] = quant_cfg

            if self.config["revision"]:
                model_kwargs["revision"] = self.config["revision"]

            if self.config["model_type"] == "seq2seq":
                self._base_model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.config["base_model_id"], **model_kwargs
                )
            else:
                self._base_model = AutoModelForCausalLM.from_pretrained(
                    self.config["base_model_id"], **model_kwargs
                )

            tokenizer = self.load_tokenizer()
            if hasattr(self._base_model.config, 'vocab_size') and len(tokenizer) > self._base_model.config.vocab_size:
                self._base_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

        return self._base_model

    def load_finetuned_model(self):
        if self._finetuned_model is None:
            if not self.config["lora_path"]:
                raise ValueError("lora_path must be specified to load finetuned model")

            base_model = self.load_base_model()
            self._finetuned_model = PeftModel.from_pretrained(base_model, self.config["lora_path"])

        return self._finetuned_model

    def build_nllb_prompt(self, text, src_lang, tgt_lang):
        tokenizer = self.load_tokenizer()

        if "nllb" in self.config["base_model_id"].lower():
            src_code = "eng_Latn" if src_lang.lower().startswith("en") else "fra_Latn"
            tgt_code = "fra_Latn" if src_lang.lower().startswith("en") else "eng_Latn"

            tokenizer.src_lang = src_code
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            inputs["forced_bos_token_id"] = tokenizer.lang_code_to_id[tgt_code]
            return inputs
        else:
            return tokenizer(text, return_tensors="pt", padding=True)

    def build_causal_prompt(self, text, src_lang) -> str:
        if src_lang.lower().startswith("en"):
            prompt = "You are a professional scientific translator. Translate this English text to French, preserving all technical terminology and maintaining academic tone. Output only the translation:\n\n"
            prompt += f"English: {text}\n\nTranslation:"
        else:
            prompt = "Vous êtes un traducteur scientifique professionnel. Traduisez ce texte français en anglais, en préservant toute la terminologie technique et en maintenant le ton académique. Ne donnez que la traduction:\n\n"
            prompt += f"Français: {text}\n\nTraduction:"
        return prompt

    @torch.inference_mode()
    def translate_text(self, input_text, input_language="en",
                       target_language="fr", use_finetuned: bool = False,
                       generation_kwargs=None) -> str:
        tokenizer = self.load_tokenizer()

        if use_finetuned and self.config["lora_path"]:
            model = self.load_finetuned_model()
        else:
            model = self.load_base_model()

        if self.config["model_type"] == "seq2seq":
            inputs = self.build_nllb_prompt(input_text, input_language, target_language)
            inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            default_gen_kwargs = {
                "max_new_tokens": 512,
                "min_new_tokens": 5,
                "num_beams": 4,
                "temperature": 0.1,
                "top_p": 0.8,
                "repetition_penalty": 1.1,
                "early_stopping": True,
                "pad_token_id": tokenizer.pad_token_id,
                "no_repeat_ngram_size": 3,
            }
        else:
            prompt = self.build_causal_prompt(input_text, input_language)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            stop_token_ids = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("."),
                tokenizer.convert_tokens_to_ids("\n"),
            ]

            default_gen_kwargs = {
                "max_new_tokens": 512,
                "min_new_tokens": 5,
                "do_sample": False,
                "num_beams": 4,
                "repetition_penalty": 1.1,
                "early_stopping": True,
                "eos_token_id": stop_token_ids,
                "pad_token_id": tokenizer.pad_token_id,
                "no_repeat_ngram_size": 3,
            }

        if generation_kwargs:
            default_gen_kwargs.update(generation_kwargs)

        output_ids = model.generate(**inputs, **default_gen_kwargs)

        if self.config["model_type"] == "seq2seq":
            generated = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            translation = generated.strip()
        else:
            generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            if "Translation:" in generated:
                translation = generated.split("Translation:", 1)[1].strip()
            elif "Traduction:" in generated:
                translation = generated.split("Traduction:", 1)[1].strip()
            else:
                prompt_text = self.build_causal_prompt(input_text, input_language)
                translation = generated.replace(prompt_text, "").strip()

            translation = translation.split("\n")[0].strip()

        translation = self._clean_output(translation)

        if self.config["debug"]:
            self.logger.debug(f'Input: {input_text}')
            self.logger.debug(f'Generated: {generated}')
            self.logger.debug(f'Translation: {translation}')

        return translation

    def _clean_output(self, text: str) -> str:
        import re

        patterns_to_remove = [
            r'^(Here is the translation|Voici la traduction)[:\s]*',
            r'^(Translation|Traduction)[:\s]*',
            r'^(The translation is|La traduction est)[:\s]*',
            r'\s*\([^)]*translation[^)]*\)\s*$',
        ]

        cleaned = text
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()

        return cleaned

    def clear_cache(self):
        self._base_model = None
        self._finetuned_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_translator(base_model_id: str, lora_path=None,
                      local_files_only: bool = False, model_type: str = "seq2seq", **kwargs):
    config = {
        "base_model_id": base_model_id,
        "lora_path": lora_path,
        "local_files_only": local_files_only,
        "model_type": model_type,
        **kwargs
    }
    return TranslationModel(config)


