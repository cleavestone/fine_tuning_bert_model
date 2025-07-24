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
- **9,573 entries** were **duplicates** of each other based on the `review_text` column.
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
- A stratified split was performed to maintain class balance across train/test.

---

*(Next: Fine-tuning setup, metrics, and insights... let me know when you're ready to continue)*

