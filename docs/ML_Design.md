# Email Search Engine

This document outlines the machine learning system design for Email Search Engine. The design follows a structured 9-step approach, covering all critical aspects from problem definition to deployment and monitoring.

**Authors:**
* Egor Chernobrovkin (_e.chernobrovkin@innopolis.university_)
* Alexandra Starikova (_a.nasibullina@innopolis.university_)

## **1. Problem Definition**

There are many situations where users want to find a specific email in their mailbox. Recently, one of the authors encountered the same situation where they had to find an email with X-ray images of their teeth and send them directly to the dentist. Unfortunately, they spent a lot of time trying to find it and ended up asking the clinic to resend it again.

Despite the remarkable advances in modern ML systems (Large Reasoning Models, Agentic AI), fundamental tasks like email search remain surprisingly challenging. Current email search engines rely primarily on basic keyword matching and simple filters, failing to understand semantic meaning. This creates a significant gap between the sophisticated AI capabilities available today and the practical tools users interact with daily for essential tasks like email retrieval.

To formulate the problem more specifically, we aim to build a system that:
- performs semantic search beyond simple keyword matching to find contextually relevant emails;
- provides fast and accurate retrieval from large email datasets.

## **2. Metrics and losses**

### Offline & Online metrics
- What are the key business metrics?
- What are the model performance metrics?


### Loss functions
- What loss functions will be used?
- How do they relate to business metrics?

## **3. Dataset**

### Data Sources
- What data can be used to build system?
- Do data have any issues? What are the data quality issues? Are data fresh?

### Data Labeling
- How the train/test dataset will be crafted?


### ETL Pipeline
- Describe how the data will be processed


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