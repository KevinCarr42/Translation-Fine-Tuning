import json
import re
import time
import torch

import pandas as pd

from functools import lru_cache
from sentence_transformers import SentenceTransformer, util


# ======================================================================
# NOTE: this wasn't very effective as a final cleaning strategy
#   it didn't fix most problems (mostly OCR sentence fragment mismatch)
# ======================================================================


# encoder
sentence_encoder = SentenceTransformer(
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').to(
    'cuda' if torch.cuda.is_available() else 'cpu')

# best dataset (by inspection, they're all statistically similar)
df = pd.read_pickle("../create_training_data/matched_data_wo_linebreaks.pickle")


def clean_text(text):
    text = str(text)

    # common patterns
    text = re.sub(r"^\s*(\d{1,3}[A-Za-z]?[\.\):]?)\s+", "", text)
    text = re.sub(r"\(?\b(Figure|Table|Fig(?:ure)?|Tableau)\b\s*\d+[A-Za-z0-9\-\.]*\)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(DFO|MPO)\s+\d{4}[a-z]?", "", text)
    text = re.sub(r"[’‘`´]", "'", text)
    text = re.sub(r"[–—−]", "-", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


@lru_cache(maxsize=None)
def cached_encode(text):
    return sentence_encoder.encode(text, convert_to_tensor=True)


def print_time_estimate(start_time, n, n_total):
    if n == 0:
        print(f"\n{n}/{n_total} complete.", end="... ")
        return

    time_elapsed = int(time.time() - start_time)
    time_remaining = int((time_elapsed / n) * (n_total - n))

    time_elapsed_text = f"{time_elapsed // 3600}h:{(time_elapsed % 3600) // 60:02d}m"
    time_remaining_text = f"{time_remaining // 3600}h:{(time_remaining % 3600) // 60:02d}m"

    print(f"\n{n}/{n_total} complete at {time_elapsed_text}. Estimated {time_remaining_text} remaining.", end="... ")


def print_status(start_time, n, n_total):
    small_update = 50
    large_update = 500

    if n % small_update == 0:
        if n % large_update == 0:
            print_time_estimate(start_time, n, n_total)
        else:
            print(f"{n}", end="... ")


def maybe_improve(row):
    fr_orig, en_orig = row.fr, row.en
    fr_clean, en_clean = clean_text(fr_orig), clean_text(en_orig)

    fr_variants = [fr_orig, fr_clean]
    en_variants = [en_orig, en_clean]
    pairs = [(f, e) for f in fr_variants for e in en_variants]

    fr_texts = list({f for f, _ in pairs})
    en_texts = list({e for _, e in pairs})

    fr_embeds = {t: cached_encode(t) for t in fr_texts}
    en_embeds = {t: cached_encode(t) for t in en_texts}

    sims = [util.pytorch_cos_sim(fr_embeds[f], en_embeds[e]).item() for f, e in pairs]
    best_idx = sims.index(max(sims))
    best_fr, best_en = pairs[best_idx]

    return pd.Series({
        "fr": best_fr,
        "en": best_en,
        "similarity": sims[best_idx]
    })


def clean_data(dataframe):
    df_out = dataframe.copy()
    start_time = time.time()
    n_rows = df_out.shape[0]

    for i, row in enumerate(df_out.itertuples(index=False)):
        new_row = maybe_improve(row)
        before, after = df_out.at[i, 'similarity'], new_row['similarity']
        if before < after and (row.fr != new_row['fr'] or row.en != new_row['en']):
            print(i, before, after)
            df_out.loc[i, ["fr", "en", "similarity"]] = new_row
        print_status(start_time, i, n_rows)

    return df_out


def save_jsonl(dataframe, filename):
    start_time = time.time()
    n_rows = dataframe.shape[0]

    with open(filename, "w", encoding="utf-8") as f:
        for i, row in enumerate(dataframe.itertuples(index=False)):
            f.write(json.dumps({
                "text": f"translate French to English: {row.fr} <sep> {row.en}"
            }, ensure_ascii=False) + "\n")
            f.write(json.dumps({
                "text": f"translate English to French: {row.en} <sep> {row.fr}"
            }, ensure_ascii=False) + "\n")
            print_status(start_time, i, n_rows)


if __name__ == '__main__':
    df_clean = clean_data(df)
    df_clean.to_pickle("df_clean.pickle")
    save_jsonl(df_clean, "training_data.jsonl")

