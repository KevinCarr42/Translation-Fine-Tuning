import random
from datetime import datetime
import json
from translate import translate_text
from translate_v2 import create_translator


def sample_jsonl(path, n_samples=10):
    with open(path, 'r') as f:
        lines = random.sample(list(f), n_samples)
    return [json.loads(line) for line in lines]


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


def test_translations(dict_of_models, n_samples=10, debug=False):
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    out_path = f"translation_comparison_{ts}.txt"
    all_models = dict_of_models.copy()

    for name, data in all_models.items():
        dict_of_models[name]['translator'] = create_translator(
            base_model_id=data['base_model_id'],
            model_type=data['model_type'],
            local_files_only=False,
            use_quantization=False,
            debug=debug
        )
        dict_of_models[name]['translator'].translate_text("Load the shards!", "en", "fr", False)

    print()
    with open(out_path, "w", encoding="utf-8") as f:
        def print_and_write(file, text):
            file.write(text)
            print(text)
        for i, d in enumerate(sample_jsonl("training_data.jsonl", n_samples), start=1):
            source = d.get("source")
            target = d.get("target")
            source_lang = d.get("source_lang")
            print_and_write(
                f,
                f"\n[sample {i}/{n_samples}] {source_lang}\n"
                f"text in\n"
                f"\t{source}\n"
                f"text out (expected)\n"
                f"\t{target}"
            )
            for name, data in all_models.items():
                translated_text = data['translator'].translate_text(
                    source,
                    input_language="en",
                    target_language="fr",
                    use_finetuned=False
                )
                print_and_write(
                    f,
                    f"text out, predicted with:\t\t{name}\n"
                    f"\t{translated_text}"
                )
                # data['translator'].clear_cache() # TODO only if out of memory

    print("\n\nCOMPLETE!!!\n\n")


if __name__ == "__main__":
    all_models = {
        # "mixtral_8x7b": {
        #     "base_model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        #     "model_type": "causal",
        # },
        "nllb_3b": {
            "base_model_id": "facebook/nllb-200-3.3B",
            "model_type": "seq2seq",
        },
        "mt5_large": {
            "base_model_id": "google/mt5-large",
            "model_type": "seq2seq",
        },
        # "medical_mt5": {
        #     "base_model_id": "ixa-ehu/Medical-mT5-large",
        #     "model_type": "seq2seq",
        # },
        "opus_mt": {
            "base_model_id": "Helsinki-NLP/opus-mt-tc-big-en-fr",
            "model_type": "seq2seq",
        },
    }
    test_translations(all_models)