# BiDAF: Bi-Directional Attention Flow network for SQuAD 2.0 question-answering
In this repo we present a PyTorch implementation of the *Bidirectional Attention Flow network* described in [[Minjoon Seo et al., Bidirectional Attention Flow for Machine Comprehension]](https://arxiv.org/abs/1611.01603).

### DATASET
The model presented here was designed for question-answering on the updated version of the Stanford Question Answering Dataset (SQuAD 2.0) [[Rajpurkar et al.,2018]](https://arxiv.org/abs/1806.03822). 
<p align="center">
  <img src="https://github.com/NLP-course-project-2023/BiDAF/blob/main/images/squad_logo.png">
</p>

The dataset comprises pairs of questions and context paragraphs from which the answer should be retrieved. Differently from the original version, SQuAD 2.0 includes also unanswerable questions, i.e., questions whose answer is not directly provided in the associated context paragraph. The dataset consists of approximately 150k questions, of which ~50% are unanswerable.

The data is available [here](https://rajpurkar.github.io/SQuAD-explorer/).

### DATA PREPARATION
We performed different data processing steps, that can be summarized as:
1. *Text pre-processing* (lowercasing, special char removal, ...)
2. *Tokenization and feature extraction* (lemmas, POS tags, and ENT extraction, vocabulary, ...)

For further information about data preparation please have a look at the detailed description in *SQUAD_main.ipynb*.  

The implementation of the *Data Preparation* pipeline is available in the directory:
```
src/data_preparation
```

### MODEL
In this section, we present the question-answering model implemented for the SQuAD 2.0 dataset. We considered the *Bi-Directional Attention Flow (BIDAF) model* as our baseline.
<img src="https://github.com/NLP-course-project-2023/BiDAF/blob/main/images/Screenshot%202023-08-12%20163853.png">
Then we introduced various enhancements to capture finer-grained linguistic information, improve contextual understanding, and, hence, generate more accurate answers. These improvements include:
- information extraction from *token features embeddings* (part-of-speech tags and entity recognition), and an 
- *Iterative Re-Attention* mechanism

### RESULTS



### ADDITIONAL
#### Requirements
