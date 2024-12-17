# Self-Supervised-Contrastive-Learning-for-Time-Series-Anomaly-Detection
Source code of the research paper "Self-Supervised Contrastive Learning with Autoencoder Augmentations for Time Series Anomaly Detection". 
\n

This repository contains the implementation of the TRACE (**T**ime series
**R**epresentation learning with **A**utoencoder-based **C**ontrastive **E**mbeddings) framework proposed in the paper.

**Overview**
This paper introduces a novel framework for multivariate time series anomaly detection using:

**Autoencoder-based augmentations** for generating positive and negative samples. \n
**Contrastive learning** via a Siamese network to learn robust embeddings. \n
**Clustering methods** (KMeans, DBSCAN) applied to the learned embeddings for anomaly detection. \n

Key features of the framework:\n
- Incorporates hard negative augmentation strategies: Random, Distance-based, and Reconstruction Error-based.\n
- Uses augmentations such as jittering, masking, time warping, and more to create diverse data views.\n
- Evaluated across multiple datasets with performance metrics like precision, recall, and F1-score.
