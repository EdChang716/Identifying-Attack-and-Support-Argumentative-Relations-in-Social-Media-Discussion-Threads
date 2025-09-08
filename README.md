# Identifying-Attack-and-Support-Argumentative-Relations-in-Social-Media-Discussion-Threads
This repository contains our work for **NTCIR-17 FinArg-1 Task**, focused on classifying **argumentative relations** (Support, Attack, None) between pairs of posts in financial social media discussions.

## 📌 Project Overview
The task involves identifying whether a second post in a discussion thread:
- **Support (1)**: Endorses the first post’s argument.
- **Attack (2)**: Challenges or refutes the first post.
- **None (0)**: Shows no clear relation.

Our objective is to improve the fine-grained classification of arguments in financial texts, which contributes to understanding how arguments are constructed and debated in online discussions:contentReference[oaicite:0]{index=0}.

## 🛠️ Methods
We explored three main approaches:

1. **Feature-based Approach**
   - Used `text-embedding-3-large` from OpenAI to encode text.
   - Trained a Random Forest classifier with 5-fold CV and grid search.
   - Evaluated using Weighted F1-score.

2. **In-Context Learning (ICL)**
   - Applied GPT-4o with system instructions and 6-shot prompting (2 examples per class).
   - Queried the API with customized prompts.
   - Model outputs: `0`, `1`, or `2`.

3. **Fine-tuning BERT-base-Chinese**
   - Tokenized post pairs with `[SEP]` separator.
   - Fine-tuned using HuggingFace `transformers` library.
   - Optimizer: AdamW (lr=2e-5, epsilon=1e-8).
   - Batch size = 16, Epochs = 3.
   - Achieved best **Weighted F1-score** among methods:contentReference[oaicite:1]{index=1}.

## 📊 Results
- Fine-tuning **BERT-base-Chinese** outperformed other methods.
- Final Kaggle submission (`Task_1.csv`) scored highest on the leaderboard.

## 📂 Repository Structure
