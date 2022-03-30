# State-of-the-art Music ZeroShot Models
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PyTorch implementation of state-of-the-art music zeroshot model

**TL;DR**
- Zeroshot learning utilizes the relationship of word embedding for large vocab.
- The music database for this repository is GTZAN (very small!)

## Quick Start
1. Download Word Model ([glove](https://nlp.stanford.edu/data/glove.42B.300d.zip), [MusicalWordEmbedding-WIP](#))
2. Download Pretrained Zeroshot Model with `mkdir /dataset/pretrained` ([zenodo](https://zenodo.org/record/6395456))
3. Run `Query by Tag` Notebook File with GTZAN Dataset(1.34G)([notebook](https://github.com/SeungHeonDoh/music_zeroshot_models/blob/master/notebook/Query_by_Tag.ipynb))

## Available Models 

with `Zeroshot Tag (1126) Supervision`

- CNN1D/GLOVE : Zero-shot Learning for Audio-based Music Classification and Tagging, Choi et al., 2019 [[arxiv](https://arxiv.org/abs/1907.02670)]
- TaggingTransformer/GLOVE : Implementation by this repo
- CNN1D/MusicalWordEmbedding : Will be Updated
- TaggingTransformer/MusicalWordEmbedding : Will be Updated

with `Tagging Tag (50) Supervision`
ㄴ
- CNN1D/GLOVE : Implementation by this repo
- TaggingTransformer/GLOVE : Implementation by this repo
- CNN1D/MusicalWordEmbedding : Will be Updated
- TaggingTransformer/MusicalWordEmbedding : Will be Updated
    
with `Zeroshot Tag/Artist/Track (1126, ~30K, ~0.5M) Supervision`

- CNN1D/MusicalWordEmbedding : Will be Updated
- TaggingTransformer/MusicalWordEmbedding : Will be Updated

## Requirements
- pytorch-lightning==1.5.5 (important!)
- torch==1.7.1 (Please install it according to your [CUDA version](https://pytorch.org/get-started/previous-versions/#linux-and-windows-4).)
```
conda create -n YOUR_ENV_NAME python=3.7
conda activate YOUR_ENV_NAME
pip install -r requirements.txt
```

## Train your self
WIP
