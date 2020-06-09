# RetCL: A Selection-based Approach for Retrosynthesis via Contrastive Learning

## Prepare dependencies

We use [Pytorch](https://pytorch.org/) for ML framework, [RDKit](https://www.rdkit.org/) for cheminformatics tool, and [DGL](https://www.dgl.ai/) for easy implementation of graph neural networks. We highly recommend to use [Anaconda](https://www.anaconda.com/products/individual) for managing package dependencies.

```bash
conda create -n RetCL python=3.7
conda activate RetCL
conda install pytorch=1.5.0 cudatoolkit=10.1 -c pytorch
conda install rdkit=2019.09 -c rdkit
conda install dgl=0.4.3 dgllife=0.2.1 -c dglteam
```

## Preprocessing data

```bash
python preprocess.py --dataset uspto_50k --datadir data/uspto_50k/
python preprocess.py --dataset uspto_50k --datadir data/uspto_50k_modified/
python preprocess.py --dataset uspto_candidates --datadir data/uspto_candidates/
```

Currently the USPTO-full dataset is not provided in this supplementray metrial due to the size limit. You can download the dataset from the [GLN](https://github.com/Hanjun-Dai/GLN) repository.

## Evaluation

We provide pretrained models in `checkpoints/`.

You can obtain the same results reported in Table 1 using the following scripts:
```bash
python evaluate.py --num-layers 5 --use-sum --best 50 --beam 50 --ckpt checkpoints/uspto50k_unknown.pth
python evaluate.py --num-layers 5 --use-sum --best 50 --beam 50 --ckpt checkpoints/uspto50k_given.pth --use-label
```
To obtain the results in Table 2, replace `--best 50 --beam 50` with `--best 200 --beam 200`.

For the generalization experiment, run the following script:
```bash
python evaluate.py --num-layers 5 --use-sum --best 50 --beam 50 --ckpt checkpoints/uspto50k_modified_unknown.pth --datadir data/uspto_50k_modified/ --classwise
```

## Training

You can train our RetCL framework using `train.py`. For example, one can learn with 4 nearest neighbors and sum pooling using the following script:
```bash
python train.py --use-sum --num-neighbors 4 --logdir logs/uspto_50k_unknown_N4
```

