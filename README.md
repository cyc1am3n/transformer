# Transformer
Re-implementation of [Attention Is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) (NIPS 2017)

## Dataset

Use [IWSLT17 FR-EN](https://wit3.fbk.eu/archive/2017-01-trnted/texts/fr/en/fr-en.tgz).

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
│  │
│  └─fr-en
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

