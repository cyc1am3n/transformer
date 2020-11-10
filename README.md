# Transformer
Re-implementation of [Attention Is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) (NIPS 2017)

## Requirements

- python=3.8.2
- requests=2.23.0
- pandas=1.1.4
- pytorch=1.4.0
- torchtext=0.6.0
- sentencepiece=0.1.85

## Dataset

Use [IWSLT17 FR-EN](https://wit3.fbk.eu/archive/2017-01-trnted/texts/fr/en/fr-en.tgz).  You can download datasets and pre-process them with a following script.  

```bash
$ python ./data/download.py
```

## Build Sentence Piece tokenizer

Train sentencepiece tokenizer using a following script. You can also change the configuration of sentencepiece tokenizer at [./data/config.json](./data/config.json)

```bash
$ python build_vocab.py
```



## Project Structure

```
transformer
│  build_vocab.py
│  evaluate.py
│  train.py
│  utils.py
│
├─data
│  │  config.json
│  │  download.py
│  │
│  ├─ORIG
│  │      ...
│  │
│  └─PREP
│          ...
│
├─experiments
│  ├─base_model
│  │      config.json
│  │
│  └─large_model
│          config.json
│
└─model
        data.py
        metric.py
        net.py
        utils.py
```

