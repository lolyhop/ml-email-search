# Email Search Engine

This document outlines the machine learning system design for Email Search Engine. The design follows a structured 9-step approach, covering all critical aspects from problem definition to deployment and monitoring.

**Authors:**
* Egor Chernobrovkin (_e.chernobrovkin@innopolis.university_)
* Alexandra Starikova (_a.nasibullina@innopolis.university_)

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
- Describe which steps we took to validate the correctness of model building (how we prevented data leakage for example)

## **5. Baseline Solution**
- Which baselines will we use (think of BM-25 and pretrained retriever models from open-source)
- What are acceptable baseline results?
- What are the model architectures we will try?
- What features will we use in baseline models?

##  **6. Error Analysis**
- How will we analyze learning curves?
- How will we handle overfitting/underfitting?


## **7. Training Pipeline**
- What frameworks will we use to train models?
- How will we tune the hyperparams?
- What are the hardware requirements?
- What metrics will we log?
- How will we track the experiments results?

## **8. Measuring & Reporting**
- What is the testing strategy? (A/B test)
- How will we split traffic?
- What are the success criteria?
- What reports will be generated?

## **9. Integration**
- What APIs will we expose?
- What are the interfaces?
- What are the SLAs?
- What are the fallback plans?

TODO: Put & Visualise UML Diagram from Assignment 3 second part here...