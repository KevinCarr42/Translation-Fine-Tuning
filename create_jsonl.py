import json


def save_jsonl(dataframe, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for i, row in enumerate(dataframe.itertuples(index=False)):
            f.write(json.dumps({
                "source": f"{row.en}",
                "target": f"{row.fr}",
                "source_lang": "en",
            }, ensure_ascii=False) + "\n")
            f.write(json.dumps({
                "source": f"{row.fr}",
                "target": f"{row.en}",
                "source_lang": "fr",
            }, ensure_ascii=False) + "\n")
