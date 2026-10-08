# Operationalizing Character-Space with Collocations

This project uses statistical collocation analysis to operationalize Alex Woloch’s concept of *character-space*.

## Corpus
- **Title**: 紅樓夢 (Dream of the Red Chamber)
- **Author**: 曹雪芹
- **Source**: `data/红楼梦.txt`
- **Target character**: 林黛玉
- **Size**: ~896,526 characters; 35,059 sentences

## Method
- OpenCC (t2s) for simplified Chinese
- jieba segmentation via qhchina
- `find_collocates` with window5, window10, sentence
- Filters: min_word_length=2, max_p=0.05

## Reproduce
```bash
pip install -r requirements.txt
python scripts/run_collocates.py
