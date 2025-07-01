# ml-email-search

This project provides a comprehensive experimental comparison of vector database indexing methods for email search applications, analyzing the performance trade-offs between different approaches to dense embedding retrieval.

## Overview

As semantic search using dense embeddings becomes more prevalent, this research systematically evaluates various vector indexing algorithms for email retrieval tasks. Through controlled experiments, we examine computational efficiency, search quality, and scalability characteristics of different indexing approaches.

## Research Questions

- How does embedder size affect corpus vectorization time?
- How does the choice of index affect build time?
- What is the search speed of different indexing & searching algorithms?
- What are the trade-offs between search time and retrieval accuracy?
- Does index quantization really help? What are the pros & cons of this approach?

## Dataset

**Enron Email Dataset** from Kaggle containing 517,401 emails, postprocessed using regex to parse IDs, dates, and content while cleaning corrupted HTML and metadata.

## Experiments

Our experimental evaluation covers:

- **Embedding Performance**: Testing embedders of different sizes for document vectorization time and computational requirements
- **Index Benchmarking**: Evaluating indexing algorithms across corpus sizes for build time and scalability assessment
- **Search Quality**: Comparing index-based vs. brute-force search using Recall@k to quantify accuracy trade-offs between different indexing methods
- **Performance Analysis**: Measuring search latency and queries-per-second across different vector index types
- **Quantization Study**: Analyzing quantization impact on IVF index speed vs. quality trade-offs

## Key Findings

- Performance comparison between different vector indexing methods (HNSW, IVF, LSH, PQ)
- Trade-off analysis between search accuracy and computational efficiency
- Quantization effects on retrieval speed and quality
- Scalability characteristics across different corpus sizes

## Technologies

PyTorch Lightning, HuggingFace Transformers, FAISS, various vector indexing algorithms (HNSW, IVF, LSH, Product Quantization)

## Team

- **Egor Chernobrovkin** (_e.chernobrovkin@innopolis.university_)
- **Alexandra Starikova** (_a.nasibullina@innopolis.university_)
