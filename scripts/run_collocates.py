from pathlib import Path
import pandas as pd
from opencc import OpenCC
from qhchina.helpers import load_stopwords
from qhchina.preprocessing.segmentation import create_segmenter
from qhchina.analytics.collocations import find_collocates

CORPUS_PATH = Path("data/红楼梦.txt")
TARGET = "林黛玉" 

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    print("1. Loading & converting to simplified Chinese...")
    raw = CORPUS_PATH.read_text(encoding="utf-8")
    text = OpenCC("t2s").convert(raw)
    print(f"   Corpus chars: {len(text)}")

    print("2. Loading stopwords & segmenting...")
    stopwords = load_stopwords("zh_sim")

    segmenter = create_segmenter(
        backend="jieba",
        strategy="sentence",
        filters={
            "stopwords": stopwords,
            "min_word_length": 2,
        },
    )
    sentences = segmenter.segment(text)

    print(f"   Sentences: {len(sentences)}")


    collocate_filters = {
        "min_word_length": 2,
        "max_p": 0.05,
        "stopwords": stopwords,
    }

    runs = [
        {"name": "window5",  "method": "window",   "horizon": 5},
        {"name": "window10", "method": "window",   "horizon": 10},
        {"name": "sentence", "method": "sentence", "horizon": None},
    ]

    all_tables = {}

    for run in runs:
        print(f"\n3. Running find_collocates → {run['name']} ...")
        kwargs = {
            "sentences": sentences,
            "target_words": TARGET,
            "method": run["method"],
            "filters": collocate_filters,
            "alternative": "greater",   # 課堂常用：只看顯著「吸引」的搭配
            "return_type": "dataframe",
        }
        if run["horizon"] is not None:
            kwargs["horizon"] = run["horizon"]

        df = find_collocates(**kwargs)

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        out_csv = OUTPUT_DIR / f"collocates_{run['name']}.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"   Saved {out_csv}  ({len(df)} rows)")
        all_tables[run["name"]] = df

    html = [
        "<html><head><meta charset='utf-8'>",
        f"<title>Collocates of {TARGET}</title>",
        "<style>body{font-family:sans-serif;margin:2em}",
        "table{border-collapse:collapse;margin-bottom:2em}",
        "th,td{border:1px solid #ccc;padding:4px 8px}</style></head><body>",
        f"<h1>搭配詞結果：{TARGET}</h1>",
    ]
    for name, df in all_tables.items():
        html.append(f"<h2>{name}</h2>")
        html.append(df.head(40).to_html(index=False))
    html.append("</body></html>")

    (OUTPUT_DIR / "results.html").write_text("\n".join(html), encoding="utf-8")
    print("\nDone → output/results.html")

if __name__ == "__main__":
    main()
