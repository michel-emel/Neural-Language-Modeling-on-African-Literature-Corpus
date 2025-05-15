# Neural Language Modeling on African Literature Corpus

This project explores statistical and neural approaches to language modeling using a custom African Literature Corpus based on the works of Chinua Achebe. It includes implementations of n-gram models with Maximum Likelihood Estimation (MLE) and LSTM-based deep learning models for next-word prediction and text generation.

---

## 📚 Dataset: African Literature Corpus

The corpus includes four literary works by Chinua Achebe:

* *Things Fall Apart*
* *Arrow of God*
* *A Man of the People*
* *Anthills of the Savannah*

The texts were cleaned, lowercased, and concatenated into a single corpus (\~190,000 tokens) for model training and evaluation.

---

## 🧠 Models

### 🔹 N-Gram Language Model (MLE)

* Used NLTK to build trigram models with sentence padding.
* Fitted an MLE model to estimate n-gram probabilities.
* Evaluated word likelihoods and tested model perplexity.

### 🔹 Neural Language Models (LSTM)

#### One-Word Input Model

* Architecture: Embedding → LSTM → Dense (softmax).
* Trained on single-word input, predicting the next word.
* Accuracy improved over 15 epochs (final: \~15.6%).

#### Multi-Word Input Model

* Trained on 50-word sequences using stacked LSTM layers.
* Generated more coherent and contextually rich sentences.
* Demonstrated improved fluency over short-context models.

---

## 📊 Sample Results

**Generated Text:**
`"worked, as it were better than any young man"`

**N-Gram Stats:**
P("a" | "man") = 0.0021
Perplexity on test set (MLE): `inf` (indicative of unseen tokens)

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/african-lit-language-model.git
cd african-lit-language-model
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Training

```bash
python train_ngram_model.py     # Trains trigram MLE model
python train_lstm_model.py      # Trains one-word and multi-word LSTM models
```

---

## 🛠️ Tools & Libraries

* Python, TensorFlow, Keras
* NLTK, NumPy, pandas, Scikit-learn
* Custom tokenization, padding, and text generation utilities

---

## 📌 Notes

* The corpus is custom-built and not a standard benchmark dataset.
* Designed for educational use in NLP experimentation and sequence modeling.