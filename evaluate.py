import random
import json
from translate import translate_text


def sample_jsonl(path, n=100):
    with open(path, 'r') as f:
        lines = random.sample(list(f), n)
    return [json.loads(line) for line in lines]


sample_translations = sample_jsonl("training_data.jsonl", 10)

for d in sample_translations:
    source = d.get('source')
    target = d.get('target')
    source_lang = d.get('source_lang')
    translated_base_model = translate_text(source, source_lang, False)
    translated_finetuned = translate_text(source, source_lang, True)

    print()
    print(source_lang, '\ttext in')
    print('\t', source)
    print('text out (expected)')
    print('\t', target)
    print('text out (predicted with base model)')
    print('\t', translated_base_model)
    print('text out (predicted with finetuned model)')
    print('\t', translated_finetuned)
    print()
