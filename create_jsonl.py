import json


def save_jsonl(dataframe, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for i, row in enumerate(dataframe.itertuples(index=False)):
            f.write(json.dumps({
                "text": f"translate French to English: {row.fr} <sep> {row.en}"
            }, ensure_ascii=False) + "\n")
            f.write(json.dumps({
                "text": f"translate English to French: {row.en} <sep> {row.fr}"
            }, ensure_ascii=False) + "\n")
