import random
from datetime import datetime
import json
import csv
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
from translate import (create_translator, NLLBTranslationModel, OpusTranslationModel,
                       M2M100TranslationModel, MBART50TranslationModel)

language_codes = {
    "en": "English",
    "fr": "French",
}


def sample_jsonl(path, n_samples=10, source_lang=None):
    with open(path, 'r') as f:
        filtered = (line for line in f if not source_lang or json.loads(line)['source_lang'] == source_lang)
        data = [json.loads(line) for line in filtered]
    return random.sample(data, min(n_samples, len(data)))


def test_translations(dict_of_models, n_samples=10, source_lang=None, debug=False):
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    INDENT = 60
    csv_path = f"translation_comparison_{ts}.csv"

    print("\nLoading embedder...\n")
    embedder = SentenceTransformer('sentence-transformers/LaBSE')

    all_models = dict_of_models.copy()
    for name, data in all_models.items():
        config_params = {
            'base_model_id': data['base_model_id'],
            'model_type': data['model_type'],
            'local_files_only': False,
            'use_quantization': False,
            'debug': debug,
        }

        if 'merged_model_path' in data:
            config_params['merged_model_path'] = data['merged_model_path']

        if 'merged_model_path_en_fr' in data:
            config_params['merged_model_path_en_fr'] = data['merged_model_path_en_fr']
        if 'merged_model_path_fr_en' in data:
            config_params['merged_model_path_fr_en'] = data['merged_model_path_fr_en']

        dict_of_models[name]['translator'] = create_translator(
            data['cls'],
            **config_params
        )
        dict_of_models[name]['translator'].translate_text("Load the shards!", "en", "fr")

    csv_data = []

    for i, d in enumerate(sample_jsonl("training_data.jsonl", n_samples, source_lang), start=1):
        source = d.get("source") + "."
        target = d.get("target") + "."
        source_lang = d.get("source_lang")
        other_lang = "en" if source_lang == "fr" else "fr"

        print(
            f"\n[sample {i}/{n_samples}] {language_codes[source_lang]}"
            f"\n{f'text in ({language_codes[source_lang]}):':<{INDENT}}{source}"
            f"\n{f'text out ({language_codes[other_lang]}), expected:':<{INDENT}}{target}"
        )

        source_embedding = embedder.encode(source, convert_to_tensor=True)
        target_embedding = embedder.encode(target, convert_to_tensor=True)
        cos_sim_original = pytorch_cos_sim(source_embedding, target_embedding).item()

        for name, data in all_models.items():
            translated_text = data['translator'].translate_text(
                source,
                input_language=source_lang,
                target_language=other_lang
            )

            translated_embedding = embedder.encode(translated_text, convert_to_tensor=True)

            cos_sim_source = pytorch_cos_sim(source_embedding, translated_embedding).item()
            cos_sim_target = pytorch_cos_sim(target_embedding, translated_embedding).item()

            csv_data.append({
                'source': source,
                'target': target,
                'source_lang': source_lang,
                'other_lang': other_lang,
                'translator_name': name,
                'translated_text': translated_text,
                'cosine_similarity_original_translation': cos_sim_original,
                'cosine_similarity_vs_source': cos_sim_source,
                'cosine_similarity_vs_target': cos_sim_target,
            })

            print(
                f"{f'text out ({language_codes[other_lang]}), predicted with {name}:':<{INDENT}}{translated_text}"
            )

            # data['translator'].clear_cache()  # TODO only if out of memory

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'source', 'target', 'source_lang', 'other_lang', 'translator_name', 'translated_text',
            'cosine_similarity_original_translation', 'cosine_similarity_vs_source', 'cosine_similarity_vs_target'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)


if __name__ == "__main__":
    all_models = {
        "nllb_3b_base_researchonly": {
            "cls": NLLBTranslationModel,
            "base_model_id": "facebook/nllb-200-3.3B",
            "model_type": "seq2seq",
        },

        "opus_mt_base": {
            "cls": OpusTranslationModel,
            "base_model_id": "Helsinki-NLP/opus-mt-tc-big-en-fr",
            "model_type": "seq2seq",
        },
        "opus_mt_finetuned": {
            "cls": OpusTranslationModel,
            "base_model_id": "Helsinki-NLP/opus-mt-tc-big-en-fr",
            "model_type": "seq2seq",
            "merged_model_path_en_fr": "merged/opus_mt_en_fr",
            "merged_model_path_fr_en": "merged/opus_mt_fr_en",
        },

        "m2m100_418m_base": {
            "cls": M2M100TranslationModel,
            "base_model_id": "facebook/m2m100_418M",
            "model_type": "seq2seq",
        },
        "m2m100_418m_finetuned": {
            "cls": M2M100TranslationModel,
            "base_model_id": "facebook/m2m100_418M",
            "model_type": "seq2seq",
            "merged_model_path": "merged/m2m100_418m",
        },

        "mbart50_mmt_base": {
            "cls": MBART50TranslationModel,
            "base_model_id": "facebook/mbart-large-50-many-to-many-mmt",
            "model_type": "seq2seq",
        },
        "mbart50_mmt_finetuned": {
            "cls": MBART50TranslationModel,
            "base_model_id": "facebook/mbart-large-50-many-to-many-mmt",
            "model_type": "seq2seq",
            "merged_model_path_en_fr": "merged/mbart50_mmt_fr",
            "merged_model_path_fr_en": "merged/mbart50_mmt_en",
        },
    }

    test_translations(all_models, n_samples=10)
