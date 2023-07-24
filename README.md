# MSIntuit - Chowder implementation

This is an official implementation of Chowder model used in MSIntuit CRC<sup>TM</sup>. 
MSIntuit CRC<sup>TM</sup> is a tool for pre-screening of microsatellite instability (MSI) using only and H&E slide of a patient with colorectal cancer (CRC).


The model is a variant of Chowder model, originally published by Courtiol et al., 2018: [Classification and Disease Localization 
in Histopathology Using Only Global Labels: A Weakly-Supervised Approach](https://arxiv.org/abs/1802.02212).

!["Chowder"](./assets/chowder.png)

## Install
Python: 3.7+

```bash
pip install .
```

## How to use
```python
import torch
from msintuit.chowder import MSIntuitChowder

dummy_features = torch.ones((16, 100, 2048))  # (batch_size, n_tiles, dimension)
model = MSIntuitChowder(in_features=2048, n_extreme=10)
logits, extreme_scores = model(dummy_features)
```
