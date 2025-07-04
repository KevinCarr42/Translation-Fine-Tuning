import pandas as pd
import spacy
import numpy as np
import time


def add_features(dataframe):
    t_total = time.perf_counter()

    print("loading nlp language models")
    t0 = time.perf_counter()
    nlp_fr = spacy.load("fr_core_news_lg")
    nlp_en = spacy.load("en_core_web_lg")
    print(f"→ done in {time.perf_counter() - t0:.2f}s")
    print(f"TOTAL time elapsed so far: {time.perf_counter() - t_total:.2f}s")

    print("creating fr nlp pipe")
    t0 = time.perf_counter()
    docs_fr = list(nlp_fr.pipe(dataframe["fr"].astype(str)))
    print(f"→ done in {time.perf_counter() - t0:.2f}s")
    print(f"TOTAL time elapsed so far: {time.perf_counter() - t_total:.2f}s")

    print("creating en nlp pipe")
    t0 = time.perf_counter()
    docs_en = list(nlp_en.pipe(dataframe["en"].astype(str)))
    print(f"→ done in {time.perf_counter() - t0:.2f}s")
    print(f"TOTAL time elapsed so far: {time.perf_counter() - t_total:.2f}s")

    print('appending len_ratio')
    t0 = time.perf_counter()
    dataframe["len_ratio"] = np.maximum(dataframe["fr"].str.len() / dataframe["en"].str.len(), dataframe["en"].str.len() / dataframe["fr"].str.len())
    print(f"→ done in {time.perf_counter() - t0:.2f}s")
    print(f"TOTAL time elapsed so far: {time.perf_counter() - t_total:.2f}s")

    print('appending verb_ratio')
    t0 = time.perf_counter()
    verb_counts_fr = [doc.count_by(spacy.attrs.POS).get(spacy.symbols.VERB, 0) + 1 for doc in docs_fr]
    verb_counts_en = [doc.count_by(spacy.attrs.POS).get(spacy.symbols.VERB, 0) + 1 for doc in docs_en]
    dataframe["verb_ratio"] = np.maximum(
        np.array(verb_counts_fr) / np.array(verb_counts_en),
        np.array(verb_counts_en) / np.array(verb_counts_fr)
    )
    print(f"→ done in {time.perf_counter() - t0:.2f}s")
    print(f"TOTAL time elapsed so far: {time.perf_counter() - t_total:.2f}s")

    print('appending noun_ratio')
    t0 = time.perf_counter()
    noun_counts_fr = [doc.count_by(spacy.attrs.POS).get(spacy.symbols.NOUN, 0) + 1 for doc in docs_fr]
    noun_counts_en = [doc.count_by(spacy.attrs.POS).get(spacy.symbols.NOUN, 0) + 1 for doc in docs_en]
    dataframe["noun_ratio"] = np.maximum(
        np.array(noun_counts_fr) / np.array(noun_counts_en),
        np.array(noun_counts_en) / np.array(noun_counts_fr)
    )
    print(f"→ done in {time.perf_counter() - t0:.2f}s")
    print(f"TOTAL time elapsed so far: {time.perf_counter() - t_total:.2f}s")

    print('appending entity_ratio')
    t0 = time.perf_counter()
    ent_counts_fr = [len(doc.ents) + 1 for doc in docs_fr]
    ent_counts_en = [len(doc.ents) + 1 for doc in docs_en]
    dataframe["entity_ratio"] = np.maximum(
        np.array(ent_counts_fr) / np.array(ent_counts_en),
        np.array(ent_counts_en) / np.array(ent_counts_fr)
    )
    print(f"→ done in {time.perf_counter() - t0:.2f}s")
    print(f"TOTAL time elapsed so far: {time.perf_counter() - t_total:.2f}s")

    print('appending clause_ratio')
    t0 = time.perf_counter()
    clauses_fr = dataframe["fr"].str.count(",") + dataframe["fr"].str.count(";") + 1
    clauses_en = dataframe["en"].str.count(",") + dataframe["en"].str.count(";") + 1
    dataframe["clause_ratio"] = np.maximum(
        clauses_fr / clauses_en,
        clauses_en / clauses_fr
    )
    print(f"→ done in {time.perf_counter() - t0:.2f}s")
    print(f"TOTAL time: {time.perf_counter() - t_total:.2f}s")

    return df


if __name__ == '__main__':
    print("reading dataframe")
    df = pd.read_pickle("../create_training_data/matched_data_wo_linebreaks.pickle")
    df = add_features(df)
    df.to_pickle("df_with_features.pickle")
