# Email Search Engine

This document outlines the machine learning system design for Email Search Engine. The design follows a structured 9-step approach, covering all critical aspects from problem definition to deployment and monitoring.

**Authors:**

- Egor Chernobrovkin (_e.chernobrovkin@innopolis.university_)
- Alexandra Starikova (_a.nasibullina@innopolis.university_)

## **1. Problem Definition**

There are many situations where users want to find a specific email in their mailbox. Recently, one of the authors encountered the same situation where they had to find an email with X-ray images of their teeth and send them directly to the dentist. Unfortunately, they spent a lot of time trying to find it and ended up asking the clinic to resend it again.

Despite the remarkable advances in modern ML systems (Large Reasoning Models, Agentic AI), fundamental tasks like email search remain surprisingly challenging. Current email search engines rely primarily on basic keyword matching and simple filters, failing to understand semantic meaning. This creates a significant gap between the sophisticated AI capabilities available today and the practical tools users interact with daily for essential tasks like email retrieval.

To formulate the problem more specifically, we aim to build a system that:

- accepts queries in arbitrary language (english, russian, etc.) and returns the most semantically relevant emails from a user's mailbox;
- performs semantic search beyond simple keyword matching to find contextually relevant emails;
- provides fast and accurate retrieval from large email datasets.

## **2. Metrics and losses**

### Metrics

For metrics we consider both online and offline approaches to evaluate system performance. Online metrics measure real user behavior and satisfaction, while offline metrics assess retrieval quality using labeled datasets.

#### Online metrics

- Time to First Email Click
  - Time spent on the search results page before clicking an email or abandoning.
- Post Query Clicks
  - The number of distinct emails clicked by a user after submitting a search query.

#### Offline metrics

- Precision@k, Recall@k;
- HitRate@k;
- NDCG@k.

### Loss functions

For loss functions we will experiment with:

- Triplet loss;
- Info-NCE loss;
- Positive/Negative Cross-Entropy loss.

These loss functions aim to bring semantically similar queries and emails closer together in the embedding space while pushing dissimilar pairs apart.

## **3. Dataset**

### Data Sources

Since we build new search engine we don't have any intrinsic data. However, we have an access to open-source datasets, so we will consider [Enron Email Dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset) as our baseline. It contains 500,000+ unlabeled emails in English. That means that we have no information about queries and relevant emails for them.

The dataset also have some issues including:

- Corrupted HTML tags;
- Corrupted metadata linked to emails;
- Emails belong to only Enror Company employees.

### Data Labeling

Since our dataset doesn't have any labels to learn from, we will generate synthetic data. For each email, we will generate 3 query examples and use them as positive pairs. To generate synthetic queries, we will use the DeepSeek R1 model from the API.

To craft negative pairs, we will use a random sampling strategy, where random emails (except the ones chosen as positives) are treated as negatives.

## **4. Validation Schema**

To validate our model building process and prevent data leakage:

- We split the dataset into training (80%), validation (10%), and test (10%) subsets by unique email Message-IDs;
- Additionally, we implement a temporal split where training data comes from earlier time periods and test data from later periods to simulate real-world deployment conditions;
- During training, emails and their corresponding synthetic queries are paired, and cross-validation is used to tune hyperparameters;
- For each experiment, we ensure no query or email overlap across train and val/test sets;

## **5. Baseline Solution**

For our baseline solution, we will test two model architectures:

- BM25 (sparse retrieval)
- Dense Passage Retriever (dense retrieval, transformer-based architecture, pretrained on popular Information Retrieval tasks)

For the baseline, we do not anticipate strong results. However, since we aim to replace traditional lexical systems like BM25, the quality of our post-baseline models should be significantly better.

## **6. Error Analysis**

- **Learning Curves**: Track training/validation loss and metrics (Recall@k, NDCG@k) over time;
- **Overfitting/Underfitting**: Use early stopping and dropout;
- **Manual Review**: Create a dashboard to inspect hard incorrectly predicted cases.

## **7. Training Pipeline**

- **Frameworks**: PyTorch Lightning, HuggingFace Transformers, Faiss for vector indexing;
- **Hyperparameters**: Tune learning rate, batch size, dropout, cross-entropy weights and margin size;
- **Hardware**: 1 GPU, 32GB RAM, 500GB SSD;
- **Metrics**: Log train/val loss, Recall@k, Precision@k, HitRate@k, NDCG@k, mean embedding norm, grads norm;
- **Experiment Tracking**: Use Weights & Biases (WandB) or TensorBoard for experiment logging and hyperparameter sweeps.

## **8. Measuring & Reporting**

- **Testing Strategy**: A/B test comparing current keyword-based system (BM-25 retrieval) vs. our new semantic engine.
- **Traffic Split**: 80% control (keyword), 20% treatment (semantic).
- **Success Criteria**:
  - \>10% improvement in Recall@5, >8% improvement in HitRate@5, and >12% improvement in Precision@5
  - 15% reduction in Time to First Email Click
  - \>5% increase in Post Query Clicks
- **Reports**:
  - P-value plots over time showing statistical significance of key metrics
  - Confidence intervals for treatment vs control comparisons
  - Time series plots of Time to First Click and Post Query Clicks
  - Conversion funnel analysis showing user engagement at each step
  - Performance degradation alerts if metrics fall below thresholds

## **9. Integration**

- **APIs**:
  - `POST /search`: Accepts query and returns top-k emails.
  - `POST /change_embedding_model`: Updates embedding model for documents and queries separately.
  - `POST /create_index`: Creates new index with predefined emails.
  - `POST /add_email`: Adds a single email to the index.
  - `GET /status`: Healthcheck and versioning.
- **Interfaces**:
  - FastAPI Backend with Swagger documentation for REST endpoints;
  - FAISS for vector storage and similarity search;
  - PostgreSQL for document metadata and private data storage;
  - PyTorch Lightning for model inference;
  - Grafana for performance tracking and monitoring;
  - Docker for containerization;
- **SLAs**: Response time < 200ms, 99.9% uptime.
- **Fallback**:
  - Revert to simpler models (BM-25) if semantic engine fails;
