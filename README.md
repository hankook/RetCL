# Retrosynthesis

## Requirements

- Pytorch (requires `nn.Transformer`)
- RDKit

#### RDKit

- https://www.rdkit.org/
- https://www.rdkit.org/docs/Install.html
- https://www.rdkit.org/docs/GettingStartedInPython.html

Installation with Conda is highly recommened.

## Preprocessing

```bash
python preprocess.py
```

## Training

```bash
python train.py --logdir logs2/single --lr 1e-4
```
