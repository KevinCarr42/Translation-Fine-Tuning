import logging
import re
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel


class BaseTranslationModel:
    def __init__(self, base_model_id, model_type="seq2seq", **parameters):
        self.base_model_id = base_model_id
        self.model_type = model_type
        self.parameters = parameters
        self.model = None
        self.tokenizer = None
        self.finetuned_model = None
        if self.parameters.get("debug"):
            logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

        self.keep_pool_size = self.parameters.get("keep_pool_size", 1024)
        self.tag_open_prefix = self.parameters.get("tag_open_prefix", "<KE")
        self.tag_close_prefix = self.parameters.get("tag_close_prefix", "</KE")
        self.tag_suffix = self.parameters.get("tag_suffix", ">")
        self.force_tags = self.parameters.get("force_tags", True)
        self.max_forced_tags = self.parameters.get("max_forced_tags", 128)

        self._tag_token_re = re.compile(r"(</?KE\d+>)")

    def _tokenizer_kwargs(self):
        return {
            "use_fast": True,
            "local_files_only": self.parameters.get("local_files_only", False),
        }

    def _model_kwargs(self, allow_device_map=True):
        kwargs = {
            "trust_remote_code": True,
            "local_files_only": self.parameters.get("local_files_only", False),
            "torch_dtype": self.parameters.get("dtype", torch.bfloat16),
        }
        if allow_device_map:
            kwargs["device_map"] = self.parameters.get("device_map", "auto")
            if self.parameters.get("max_memory"):
                kwargs["max_memory"] = self.parameters["max_memory"]
        if self.parameters.get("revision"):
            kwargs["revision"] = self.parameters["revision"]
        if self.parameters.get("use_quantization"):
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.parameters.get("dtype", torch.bfloat16),
            )
        return kwargs

    def _wrap_tokens_pool(self):
        opens = [f"{self.tag_open_prefix}{i}{self.tag_suffix}" for i in range(1, self.keep_pool_size + 1)]
        closes = [f"{self.tag_close_prefix}{i}{self.tag_suffix}" for i in range(1, self.keep_pool_size + 1)]
        return opens + closes

    def _ensure_wrap_tokens(self, tokenizer):
        pool = self._wrap_tokens_pool()
        have = set(tokenizer.get_vocab().keys())
        need = [t for t in pool if t not in have]
        if need:
            tokenizer.add_tokens(need, special_tokens=False)

    def _collect_force_and_badlists(self, tokenizer, text):
        if not self.force_tags:
            return None, None
        present = []
        for match in self._tag_token_re.finditer(text):
            token_str = match.group(1)
            ids = tokenizer(token_str, add_special_tokens=False).input_ids
            if ids:
                present.append(ids)
                if len(present) >= self.max_forced_tags:
                    break
        all_tags = [
            f"{prefix}{i}{self.tag_suffix}"
            for i in range(1, self.keep_pool_size + 1)
            for prefix in (self.tag_open_prefix, self.tag_close_prefix)
        ]
        allowed = {m.group(1) for m in self._tag_token_re.finditer(text)}
        bad = []
        for t in all_tags:
            if t not in allowed:
                tid = tokenizer(t, add_special_tokens=False).input_ids
                if tid:
                    bad.append(tid)
        return (present or None), (bad or None)

    def load_tokenizer(self):
        if self.tokenizer is None:
            tokenizer_path = self.parameters.get("merged_model_path", self.base_model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **self._tokenizer_kwargs())
            if getattr(self.tokenizer, "pad_token", None) is None and getattr(self.tokenizer, "eos_token", None):
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self._ensure_wrap_tokens(self.tokenizer)
        return self.tokenizer

    def load_model(self):
        if self.model is None:
            loader = AutoModelForSeq2SeqLM if self.model_type == "seq2seq" else AutoModelForCausalLM
            model_path = self.parameters.get("merged_model_path", self.base_model_id)
            self.model = loader.from_pretrained(model_path, **self._model_kwargs(allow_device_map=True))
            tokenizer = self.load_tokenizer()
            if hasattr(self.model.config, "vocab_size") and len(tokenizer) > self.model.config.vocab_size:
                self.model.resize_token_embeddings(len(tokenizer))
        return self.model

    def translate_text(self, input_text, input_language="en", target_language="fr", generation_kwargs=None):
        tokenizer = self.load_tokenizer()
        model = self.load_model()

    def clear_cache(self):
        self.model = None
        self.finetuned_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class NLLBTranslationModel(BaseTranslationModel):
    LANGUAGE_CODES = {"en": "eng_Latn", "fr": "fra_Latn"}

    def translate_text(
        self,
        input_text,
        input_language="en",
        target_language="fr",
        use_finetuned=False,
        generation_kwargs=None,
    ):
        tokenizer = self.load_tokenizer()
        model = self.load_model()
        tokenizer.src_lang = self.LANGUAGE_CODES[input_language]

        model_inputs = tokenizer(input_text, return_tensors="pt", padding=True)
        model_inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in model_inputs.items()}

        target_token_id = tokenizer.convert_tokens_to_ids(self.LANGUAGE_CODES[target_language])

        generation_arguments = {
            "max_new_tokens": 256,
            "num_beams": 4,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.05,
            "remove_invalid_values": True,
        }
        if target_token_id is not None:
            generation_arguments["forced_bos_token_id"] = target_token_id

        force_ids, bad_ids = self._collect_force_and_badlists(tokenizer, input_text)
        if force_ids:
            generation_arguments["force_words_ids"] = force_ids
        if bad_ids:
            generation_arguments["bad_words_ids"] = bad_ids
        if generation_kwargs:
            generation_arguments.update(generation_kwargs)

        output_token_ids = model.generate(**model_inputs, **generation_arguments)
        text_output = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0].strip()
        return text_output


class OpusTranslationModel(BaseTranslationModel):
    LANGUAGE_ALIASES = {"en": "en", "fr": "fr"}

    def __init__(self, base_model_id, model_type="seq2seq", **parameters):
        super().__init__(base_model_id, model_type, **parameters)
        self.directional_cache = {}

    def _root_model_id(self):
        parts = self.base_model_id.split("-")
        if parts[-2:] in (["en", "fr"], ["fr", "en"]):
            return "-".join(parts[:-2])
        return self.base_model_id

    def _directional_model_id(self, source_language, target_language):
        root_id = self._root_model_id()
        source_alias = self.LANGUAGE_ALIASES[source_language]
        target_alias = self.LANGUAGE_ALIASES[target_language]
        return f"{root_id}-{source_alias}-{target_alias}"

    def _load_directional(self, source_language, target_language):
        cache_key = f"{source_language}-{target_language}"
        if cache_key in self.directional_cache:
            return self.directional_cache[cache_key]

        merged_path = self.parameters.get(f"merged_model_path_{source_language}_{target_language}")
        model_id = merged_path if merged_path else self._directional_model_id(source_language, target_language)

        tokenizer = AutoTokenizer.from_pretrained(model_id, **self._tokenizer_kwargs())
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
            tokenizer.pad_token = tokenizer.eos_token
        self._ensure_wrap_tokens(tokenizer)

        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **self._model_kwargs(allow_device_map=False))
        if torch.cuda.is_available():
            model = model.cuda()
        if hasattr(model.config, "vocab_size") and len(tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))

        self.directional_cache[cache_key] = (tokenizer, model)
        return tokenizer, model

    def translate_text(
        self,
        input_text,
        input_language="en",
        target_language="fr",
        use_finetuned=False,
        generation_kwargs=None,
    ):
        tokenizer, model = self._load_directional(input_language, target_language)

        model_inputs = tokenizer(input_text, return_tensors="pt", padding=True)
        model_inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in model_inputs.items()}

        generation_arguments = {
            "max_new_tokens": 256,
            "num_beams": 4,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.05,
            "remove_invalid_values": True,
        }

        force_ids, bad_ids = self._collect_force_and_badlists(tokenizer, input_text)
        if force_ids:
            generation_arguments["force_words_ids"] = force_ids
        if bad_ids:
            generation_arguments["bad_words_ids"] = bad_ids
        if generation_kwargs:
            generation_arguments.update(generation_kwargs)

        output_token_ids = model.generate(**model_inputs, **generation_arguments)
        text_output = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0].strip()
        return text_output


class M2M100TranslationModel(BaseTranslationModel):
    LANGUAGE_CODES = {"en": "en", "fr": "fr"}

    def translate_text(self, input_text, input_language="en", target_language="fr", use_finetuned=False, generation_kwargs=None):
        tokenizer = self.load_tokenizer()
        model = self.load_model()
        tokenizer.src_lang = self.LANGUAGE_CODES[input_language]

        model_inputs = tokenizer(input_text, return_tensors="pt", padding=True)
        model_inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in model_inputs.items()}

        generation_arguments = {
            "max_new_tokens": 256,
            "num_beams": 4,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
            "forced_bos_token_id": tokenizer.get_lang_id(self.LANGUAGE_CODES[target_language]),
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.05,
            "remove_invalid_values": True,
        }

        force_ids, bad_ids = self._collect_force_and_badlists(tokenizer, input_text)
        if force_ids:
            generation_arguments["force_words_ids"] = force_ids
        if bad_ids:
            generation_arguments["bad_words_ids"] = bad_ids
        if generation_kwargs:
            generation_arguments.update(generation_kwargs)

        output_token_ids = model.generate(**model_inputs, **generation_arguments)
        text_output = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0].strip()
        return text_output


class MBART50TranslationModel(BaseTranslationModel):
    LANGUAGE_CODES = {"en": "en_XX", "fr": "fr_XX"}

    def __init__(self, base_model_id, model_type="seq2seq", **parameters):
        super().__init__(base_model_id, model_type, **parameters)
        self.directional_cache = {}

    def _get_directional_model_path(self, source_language, target_language):
        direction_key = f"merged_model_path_{source_language}_{target_language}"
        if direction_key in self.parameters:
            return self.parameters[direction_key]
        return self.base_model_id

    def _load_directional(self, source_language, target_language):
        cache_key = f"{source_language}-{target_language}"
        if cache_key in self.directional_cache:
            return self.directional_cache[cache_key]

        model_path = self._get_directional_model_path(source_language, target_language)

        tokenizer = AutoTokenizer.from_pretrained(model_path, **self._tokenizer_kwargs())
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
            tokenizer.pad_token = tokenizer.eos_token
        self._ensure_wrap_tokens(tokenizer)

        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **self._model_kwargs(allow_device_map=False))
        if torch.cuda.is_available():
            model = model.cuda()
        if hasattr(model.config, "vocab_size") and len(tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))

        self.directional_cache[cache_key] = (tokenizer, model)
        return tokenizer, model

    def translate_text(self, input_text, input_language="en", target_language="fr", generation_kwargs=None):
        tokenizer, model = self._load_directional(input_language, target_language)
        tokenizer.src_lang = self.LANGUAGE_CODES[input_language]

        model_inputs = tokenizer(input_text, return_tensors="pt", padding=True)
        model_inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in model_inputs.items()}

        target_id = getattr(tokenizer, "lang_code_to_id", {}).get(self.LANGUAGE_CODES[target_language]) if hasattr(tokenizer, "lang_code_to_id") else None
        if target_id is None:
            target_id = tokenizer.convert_tokens_to_ids(self.LANGUAGE_CODES[target_language])

        generation_arguments = {
            "max_new_tokens": 256,
            "num_beams": 4,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
            "forced_bos_token_id": target_id,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.05,
            "remove_invalid_values": True,
        }

        force_ids, bad_ids = self._collect_force_and_badlists(tokenizer, input_text)
        if force_ids:
            generation_arguments["force_words_ids"] = force_ids
        if bad_ids:
            generation_arguments["bad_words_ids"] = bad_ids
        if generation_kwargs:
            generation_arguments.update(generation_kwargs)

        output_token_ids = model.generate(**model_inputs, **generation_arguments)
        text_output = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0].strip()
        return text_output

    def clear_cache(self):
        self.directional_cache.clear()
        super().clear_cache()


def create_translator(translator_class, **config):
    return translator_class(**config)
