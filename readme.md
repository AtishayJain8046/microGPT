# 🧠 nanoGPT from Scratch (Jupyter Notebook)

A minimal implementation of a GPT-style language model built entirely from scratch in a single Jupyter Notebook. This project focuses on understanding the internal mechanics of transformer-based language models.

---

# 🚀 Overview

This project walks through building a **Generative Pre-trained Transformer (GPT)** step-by-step using PyTorch, without relying on high-level libraries like Hugging Face Transformers.

All components—from tokenization to training and text generation—are implemented manually inside a single notebook.

---

# 📓 Notebook Structure

The notebook (`gpt.ipynb`) is organized into the following sections:

### 1. Data Loading & Preprocessing

* Load Tiny Shakespeare dataset
* Character-level tokenization
* Vocabulary creation
* Encode/decode functions

---

### 2. Input Pipeline

* Batch generation
* Context window (block size)
* Train/validation split

---

### 3. Model Components

#### 🔹 Embedding Layer

* Token embeddings
* Positional embeddings

#### 🔹 Self-Attention Mechanism

* Query, Key, Value projections
* Scaled dot-product attention
* Masking for autoregressive behavior

#### 🔹 Transformer Block

* Multi-head attention
* Feedforward network (MLP)
* Residual connections
* Layer normalization

---

### 4. Full GPT Model

* Stack of transformer blocks
* Final linear layer for token prediction

---

### 5. Training Loop

* Forward pass
* Cross-entropy loss
* Backpropagation
* Optimization (AdamW)

---

### 6. Text Generation

* Autoregressive token generation
* Sampling with temperature

---

# 🧠 Key Concepts Implemented

### Tokenization

Character-level encoding of text into integers.

### Embeddings

Mapping tokens to dense vector representations.

### Self-Attention

```python
Attention(Q, K, V) = softmax(QKᵀ / √d) V
```

### Transformer Architecture

* Multi-head attention
* Feedforward layers
* Residual connections

### Language Modeling Objective

Predict the next token in a sequence.

---

# ⚙️ Requirements

Install dependencies:

```bash
pip install torch numpy matplotlib
```

---

# ▶️ How to Run

1. Open the notebook:

```bash
jupyter notebook gpt.ipynb
```

2. Run cells sequentially:

   * Preprocess data
   * Build model
   * Train model
   * Generate text

---

# ✨ Sample Output

After training, the model generates Shakespeare-like text:

```text
To be, or not to be: that is the question:
Whether 'tis nobler in the mind to suffer...
```

---

# 📊 What I Learned

* How transformer architectures work internally
* How attention enables contextual understanding
* How tokens are converted into meaningful representations
* How autoregressive text generation works

---

# ⚠️ Limitations

* Trained on a small dataset (Tiny Shakespeare)
* Character-level model (limited expressiveness)
* Not optimized for performance or scale

---

# 🔮 Future Improvements

* Implement subword tokenization (BPE)
* Train on larger datasets
* Add GPU acceleration
* Experiment with deeper/larger models

---

# 📚 References

* Attention Is All You Need (Vaswani et al., 2017)
* nanoGPT by Andrej Karpathy





