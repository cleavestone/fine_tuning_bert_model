# Fine-Tuning BERT for Multiclass Text Classification

## 📌 Project Overview

This project focuses on fine-tuning the BERT (base-uncased) model for a multiclass text classification task. The goal is to predict the **sentiment** of a review as one of the following classes:
- `Negative`
- `Neutral`
- `Positive`

The task is framed as a **3-class classification problem**, and Hugging Face Transformers were used to perform training and evaluation. The performance of the fine-tuned model is compared against the zero-shot performance of the pretrained model.

---

## 📊 Dataset Analysis

The initial dataset provided contained approximately **9,999 rows** of review text and corresponding sentiment labels. However, during preprocessing, a **critical issue** was discovered:

### ⚠️ Major Finding: Extreme Duplication

Out of the 9,999 rows:
- **9,573 entries** were **duplicates** of each other based on the `review_text` and `sentiment` column.
- Only **426 unique reviews** remained after removing duplicates.

### 🔍 Implication

This duplication would cause the model to:
- **See the same samples repeatedly during training**, making it trivial to memorize.
- **Likely overfit** if evaluated on the same (or duplicated) examples during validation or testing.
- Produce **unrealistically high performance metrics** due to data leakage.

### ✅ Solution

To ensure fair evaluation and effective fine-tuning:
- **All duplicates were removed** from the dataset.
- The remaining **426 unique examples** were used for training and validation.

---

## 🚫 Zero-Shot Performance (Pretrained BERT without Fine-Tuning)

Before fine-tuning, I evaluated the pretrained `bert-base-uncased` model in a zero-shot setting — meaning, it was directly used to classify review sentiments without any task-specific training.

### 📉 Classification Report (Zero-Shot)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Negative) | 0.00 | 0.00 | 0.00 | 29 |
| 1 (Neutral)  | 0.35 | 1.00 | 0.52 | 30 |
| 2 (Positive) | 0.00 | 0.00 | 0.00 | 27 |

**Overall:**
- **Accuracy:** 0.35  
- **Macro Avg F1-Score:** 0.17  
- **Weighted Avg F1-Score:** 0.18  

### ⚠️ Observations

- The model **predicted all samples as 'Neutral'**, completely ignoring the 'Negative' and 'Positive' classes.
- This led to **zero precision, recall, and F1-score** for two out of the three classes.
- The **macro and weighted F1-scores were extremely low**, reflecting the poor per-class performance.

### 💡 Why Did This Happen?

The `bert-base-uncased` model was **not fine-tuned for sentiment classification**, so:
- It lacked task-specific understanding of what makes a review "positive", "neutral", or "negative".
- In the absence of guidance, it **defaulted to predicting the majority class**, or the one with the most "neutral" embeddings.

This result **highlights the importance of fine-tuning** even powerful language models like BERT on domain-specific and task-specific data.



