# CL-Bayesian-SNN-PFP

Research code accompanying the paper:

**Protein Function Prediction Using a Bayesian Spiking Neural Network Ensemble with Contrastive Knowledge Transfer**

## Overview

This project investigates protein function prediction using an ensemble of spiking neural networks (SNNs) with knowledge transfer from contrastively pretrained protein representations.

A teacher classifier is trained on fixed CPC-Prot representations, while three heterogeneous SNN students operate on fixed ESM-2 embeddings. Student training consists of knowledge distillation followed by weighted binary cross-entropy and Soft-F1 fine-tuning. Predictions are combined using validation-based generalized Bayesian ensemble weights.

## Proposed Method

- Protein embeddings: ESM-2 (1280 dimensions)
- Teacher representations: CPC-Prot (512 dimensions)
- Student models: Three heterogeneous SNNs with LIF neurons
- Simulation steps: 10
- Training: Knowledge distillation followed by Soft-F1 fine-tuning
- Ensemble: Generalized Bayesian weighting using validation BCE losses

## Benchmark

The experiments use the *Saccharomyces cerevisiae(yeast)* random-split benchmark distributed with GOProFormer.

The Biological Process (BP) setting contains 284 GO terms, with 3,424 training proteins, 605 validation proteins, and 712 test proteins.

The reported BP ensemble results are:

| Metric | Test |
|---|---:|
| Fmax | 0.6002 |
| Smin | 15.4171 |
| AUPRC | 0.6389 |

## Code Availability

This repository is an initial research-code release focused on the BP experiment.

The main training script is `PFP_Bayesian_SNN_BP_MF_CC_IDE.py`.

The initial release does not yet contain every supporting module, pretrained representation, dataset artifact, or checkpoint required to reproduce the published experiments independently.

Additional components may be released in subsequent updates.

## Environment

The experiments were conducted using Python 3.12, PyTorch 2.4.1 with CUDA 11.8, and snnTorch 1.0.0.

See `requirements.txt` for the core Python dependencies.

## Data and Reproducibility

The original GOProFormer benchmark splits, aligned protein embeddings, trained CPC-Prot teacher, Gene Ontology files, and supporting evaluation utilities are required for full reproduction.

The GO ontology release used in the experiments is dated January 13, 2022.

Absolute file paths and local data locations in the research script may need adjustment.

## Citation

If you use this work, please cite the accompanying conference paper. Full bibliographic information will be added after publication.
