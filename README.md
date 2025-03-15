# ProCIS_Tensorflow_Implementation
ProCIS is a large-scale benchmark dataset designed for proactive document retrieval in multi-party conversations. It consists of over 2.8 million conversations linked to Wikipedia articles as relevance targets. This project aims to evaluate and improve proactive conversational AI systems that can monitor ongoing dialogues and provide timely, relevant information without explicit user requests.

# Project Overview
Dataset: 2.8 million multi-party conversations from Reddit threads.

Relevance Targets: Linked to 5.3 million Wikipedia articles.

Evaluation Metric: Normalized Proactive Discounted Cumulative Gain (npDCG).

# Key Features
Proactive Retrieval: Systems can engage without explicit user queries.

Temporal Generalization: Evaluated using a future-dev split with newer conversations.

Hybrid Pipelines: Combining BM25 with transformer models for improved performance.

# Implementation
Framework: Originally implemented in PyTorch, our version uses TensorFlow for production-oriented efficiency.

Models Evaluated: TF-IDF, BM25, Dense Passage Retriever (DPR), BERT, T5, GPT-3.5.

# Results
Performance Comparison: T5-base achieved the highest npDCG@10 but faced computational inefficiencies.

Temporal Generalization: All models showed a performance drop on newer conversations.

# Data Sources
Conversations: Sourced from Reddit.

Relevance Targets: Wikipedia articles.

# Future Work
Temporal Adaptation: Improve model performance on emerging topics.

Multi-modal Integration: Expand to include images or videos.

Acknowledgments
Original Authors: Chris Samarinas(csamarinas@cs.umass.edu) and Hamed Zamani(zamani@cs.umass.edu)
