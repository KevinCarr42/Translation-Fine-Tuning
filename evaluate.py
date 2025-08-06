import random
from datetime import datetime
import json
from translate import translate_text
from translate_v2 import create_translator, NLLBTranslationModel, OpusTranslationModel


language_codes = {
    "en": "English",
    "fr": "French",
}


def sample_jsonl(path, n_samples=10, source_lang=None):
    with open(path, 'r') as f:
        filtered = (line for line in f if not source_lang or json.loads(line)['source_lang'] == source_lang)
        data = [json.loads(line) for line in filtered]
    return random.sample(data, min(n_samples, len(data)))


def compare_finetuning(n=10, run_name="compare_finetuning"):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = f"{run_name}_{ts}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for i, d in enumerate(sample_jsonl("training_data.jsonl", n), start=1):
            source = d.get("source")
            target = d.get("target")
            source_lang = d.get("source_lang")
            translated_base_model = translate_text(source, source_lang, False)
            translated_finetuned = translate_text(source, source_lang, True)

            chunk = (
                f"\n[{i}/{n}] {source_lang}\ttext in\n"
                f"\t{source}\n"
                f"text out (expected)\n"
                f"\t{target}\n"
                f"text out (predicted with base model)\n"
                f"\t{translated_base_model}\n"
                f"text out (predicted with finetuned model)\n"
                f"\t{translated_finetuned}\n"
            )
            print(chunk, end="", flush=True)
            f.write(chunk)
    print(f"\nSaved to {out_path}", flush=True)


def test_translations(dict_of_models, n_samples=10, source_lang=None, debug=False):
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    INDENT = 50
    out_path = f"translation_comparison_{ts}.txt"
    all_models = dict_of_models.copy()

    for name, data in all_models.items():
        dict_of_models[name]['translator'] = create_translator(
            data['cls'],
            base_model_id=data['base_model_id'],
            model_type=data['model_type'],
            local_files_only=False,
            use_quantization=False,
            debug=debug,
        )
        dict_of_models[name]['translator'].translate_text("Load the shards!", "en", "fr", False)

    with open(out_path, "w", encoding="utf-8") as f:
        def print_and_write(file, text):
            file.write(text + "\n")
            print(text)
        for i, d in enumerate(sample_jsonl("training_data.jsonl", n_samples, source_lang), start=1):
            source = d.get("source") + "."
            target = d.get("target") + "."
            source_lang = d.get("source_lang")
            other_lang = "en" if source_lang == "fr" else "fr"
            print_and_write(
                f,
                f"\n[sample {i}/{n_samples}] {language_codes[source_lang]}"
                f"\n{f'text in ({language_codes[source_lang]}):':<{INDENT}}{source}"
                f"\n{f'text out ({language_codes[other_lang]}), expected:':<{INDENT}}{target}"
            )
            for name, data in all_models.items():
                translated_text = data['translator'].translate_text(
                    source,
                    input_language=source_lang,
                    target_language=other_lang,
                    use_finetuned=False
                )
                print_and_write(
                    f,
                    f"{f'text out ({language_codes[other_lang]}), predicted with {name}:':<{INDENT}}{translated_text}"
                )
                # data['translator'].clear_cache() # TODO only if out of memory


if __name__ == "__main__":
    all_models = {
        "nllb_3b": {
            "cls": NLLBTranslationModel,
            "base_model_id": "facebook/nllb-200-3.3B",
            "model_type": "seq2seq",
        },
        "opus_mt": {
            "cls": OpusTranslationModel,
            "base_model_id": "Helsinki-NLP/opus-mt-tc-big-en-fr",  # fr->en auto-swap
            "model_type": "seq2seq",
        },
    }

    test_translations(all_models, n_samples=100, source_lang="fr")
