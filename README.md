# Self-Supervised-Contrastive-Learning-for-Time-Series-Anomaly-Detection
Source code of the research paper "Self-Supervised Contrastive Learning with Autoencoder Augmentations for Time Series Anomaly Detection". 
\n

This repository contains the implementation of the TRACE (**T**ime series
**R**epresentation learning with **A**utoencoder-based **C**ontrastive **E**mbeddings) framework proposed in the paper.

**Overview**
This paper introduces a novel framework for multivariate time series anomaly detection using:

**Autoencoder-based augmentations** for generating positive and negative samples.
**Contrastive learning** via a Siamese network to learn robust embeddings.
**Clustering methods** (KMeans, DBSCAN) applied to the learned embeddings for anomaly detection.

Key features of the framework:
Incorporates hard negative augmentation strategies: Random, Distance-based, and Reconstruction Error-based.
Uses augmentations such as jittering, masking, time warping, and more to create diverse data views.
Evaluated across multiple datasets with performance metrics like precision, recall, and F1-score.
