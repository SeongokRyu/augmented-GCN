# augmented-GCN

***

This repository provides implementations introduced in 
"Deeply learning molecular structure property relationships using attention- and gate-augmented graph convolutional network".

***

# Usage
We used several scripts and a 'Harvard Clean Energy Project (CEP)' dataset in https://github.com/HIPS/neural-fingerprint.

## 1. First, make results and save directories to save output files and save files, respectively.

> mkdir results 
> mkdir save

## 2. Convert smiles files to graph inputs at a database folder.

> cd database
> python smilesToGraph.py ZINC 10000 1
> python smilesToGraph.py CEP 1000 1

## 3. Also, enter below command to obtain logP, TPSA, QED and SAS.

> python calcProperty.py

## 4. Training

python train.py model property #layers #epoch initial_learning_rate decay_rate 

> python train.py GCN logP 3 100 0.001 0.95

models : GCN, GCN+a, GCN+g, GCN+a+g, GGNN 

property : logP, TPSA, QED, SAS (ZINC dataset) and pve (CEP dataset)

