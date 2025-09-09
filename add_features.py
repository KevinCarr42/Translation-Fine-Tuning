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
    nlp_en.disable_pipes("parser")
    nlp_fr.disable_pipes("parser")
    print(f"→ done in {(time.perf_counter() - t0)/60:.2f} min")
    print(f"TOTAL time elapsed so far: {(time.perf_counter() - t_total)/60:.2f} min")

    print("creating fr nlp pipe")
    t0 = time.perf_counter()
    docs_fr = list(nlp_fr.pipe(dataframe["fr"].astype(str), n_process=6, batch_size=1000))
    print(f"→ done in {(time.perf_counter() - t0)/60:.2f} min")
    print(f"TOTAL time elapsed so far: {(time.perf_counter() - t_total)/60:.2f} min")

    print("creating en nlp pipe")
    t0 = time.perf_counter()
    docs_en = list(nlp_en.pipe(dataframe["en"].astype(str), n_process=6, batch_size=1000))
    print(f"→ done in {(time.perf_counter() - t0)/60:.2f} min")
    print(f"TOTAL time elapsed so far: {(time.perf_counter() - t_total)/60:.2f} min")

    print('appending len_ratio')
    t0 = time.perf_counter()
    dataframe["len_ratio"] = dataframe["fr"].str.len() / dataframe["en"].str.len()
    print(f"→ done in {(time.perf_counter() - t0)/60:.2f} min")
    print(f"TOTAL time elapsed so far: {(time.perf_counter() - t_total)/60:.2f} min")

    print('appending verb_ratio')
    t0 = time.perf_counter()
    verb_counts_fr = [doc.count_by(spacy.attrs.POS).get(spacy.symbols.VERB, 0) + 1 for doc in docs_fr]
    verb_counts_en = [doc.count_by(spacy.attrs.POS).get(spacy.symbols.VERB, 0) + 1 for doc in docs_en]
    dataframe["verb_ratio"] = np.array(verb_counts_fr) / np.array(verb_counts_en)
    print(f"→ done in {(time.perf_counter() - t0)/60:.2f} min")
    print(f"TOTAL time elapsed so far: {(time.perf_counter() - t_total)/60:.2f} min")

    print('appending noun_ratio')
    t0 = time.perf_counter()
    noun_counts_fr = [doc.count_by(spacy.attrs.POS).get(spacy.symbols.NOUN, 0) + 1 for doc in docs_fr]
    noun_counts_en = [doc.count_by(spacy.attrs.POS).get(spacy.symbols.NOUN, 0) + 1 for doc in docs_en]
    dataframe["noun_ratio"] = np.array(noun_counts_fr) / np.array(noun_counts_en)
    print(f"→ done in {(time.perf_counter() - t0)/60:.2f} min")
    print(f"TOTAL time elapsed so far: {(time.perf_counter() - t_total)/60:.2f} min")

    print('appending entity_ratio')
    t0 = time.perf_counter()
    ent_counts_fr = [len(doc.ents) + 1 for doc in docs_fr]
    ent_counts_en = [len(doc.ents) + 1 for doc in docs_en]
    dataframe["entity_ratio"] = np.array(ent_counts_fr) / np.array(ent_counts_en)
    print(f"→ done in {(time.perf_counter() - t0)/60:.2f} min")
    print(f"TOTAL time elapsed so far: {(time.perf_counter() - t_total)/60:.2f} min")

    print('appending clause_ratio')
    t0 = time.perf_counter()
    clauses_fr = dataframe["fr"].str.count(",") + dataframe["fr"].str.count(";") + 1
    clauses_en = dataframe["en"].str.count(",") + dataframe["en"].str.count(";") + 1
    dataframe["clause_ratio"] = clauses_fr / clauses_en
    print(f"→ done in {(time.perf_counter() - t0)/60:.2f} min")
    print(f"TOTAL time: {(time.perf_counter() - t_total)/60:.2f} min")
    print("Saving file...")

    return dataframe


if __name__ == '__main__':
    print("reading dataframe")
    df = pd.read_pickle("../Data/matched_data_wo_linebreaks.pickle")
    df = add_features(df)

    df.to_pickle("../Data/df_with_features.pickle")
    print("Done!")

