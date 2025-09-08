# Task 1: Identifying Attack and Support Argumentative Relations in Social Media Discussion Threads

This repository contains our work for **NTCIR-17 FinArg-1 Task**, focused on classifying **argumentative relations** (Support, Attack, None) between pairs of posts in financial social media discussions.

---

## 📌 Project Overview
The task involves identifying whether a second post in a discussion thread:

- **Support (1)**: Endorses the first post’s argument.  
- **Attack (2)**: Challenges or refutes the first post.  
- **None (0)**: Shows no clear relation.  

Our objective is to improve the fine-grained classification of arguments in financial texts, which contributes to understanding how arguments are constructed and debated in online discussions:contentReference[oaicite:0]{index=0}.

---

## 🛠️ Methods
We explored three main approaches:

### 1. Feature-based Approach
- Used `text-embedding-3-large` from OpenAI to encode text.  
- Trained a Random Forest classifier with 5-fold cross-validation and grid search.  
- Evaluated using **Weighted F1-score**.  

### 2. Supervised Fine-tuning (GPT-3.5)
- Converted training and development data into **chat-style JSON format** with system, user, and assistant roles.  
- Each instance contained two posts as input and the relation label (`0`, `1`, or `2`) as output.  
- Fine-tuned GPT-3.5 with these structured examples to adapt it for the classification task.  
- Produced prediction outputs on the test set through the fine-tuned model.  

### 3. Fine-tuning BERT-base-Chinese
- Tokenized post pairs with `[SEP]` separator.  
- Fine-tuned using HuggingFace `transformers` library inside a Jupyter Notebook.  
- Optimizer: **AdamW** (lr=2e-5, epsilon=1e-8).  
- Batch size = 16, Epochs = 3.  
- Achieved the **best Weighted F1-score** among methods:contentReference[oaicite:1]{index=1}.  

---

## 📊 Results
We evaluated three main approaches on the Kaggle leaderboard.  
The performance is reported in terms of **Weighted F1 Score** (higher is better).

| Method                           | Weighted F1 Score |
|----------------------------------|---------------|
| **BERT-base-Chinese (Fine-tuning)** | **0.71291**   | 
| Feature-based (OpenAI Embedding + RF) | 0.58593      | 
| Fine-tuned GPT-3.5 (Supervised)   | 0.55511       | 

📌 **Conclusion:**  
- Fine-tuning **BERT-base-Chinese** achieved the best performance suparssing strong baseline.
---

## 📂 Repository Structure
```plaintext
Task1-FinArg-Classification/
│
├── README.md                # Project documentation
│
├── report/                  # Reports and documents
│   ├── Task_1_report.pdf
│   └── Team_Project.pdf
│
├── data/                    # Data (only instructions or samples)
│   └── README.md
│
├── notebooks/               # Jupyter notebooks for each approach
│   ├── feature_based.ipynb
│   ├── gpt_finetune.ipynb
│   └── bert_finetune.ipynb
