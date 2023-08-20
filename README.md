# BiDAF: Bi-Directional Attention Flow network for SQuAD 2.0
In this repo we present a PyTorch implementation of the *Bidirectional Attention Flow network* described in [[Minjoon Seo et al., Bidirectional Attention Flow for Machine Comprehension]](https://arxiv.org/abs/1611.01603).

For further information and a more detailed explanation of the data pre-processing, model design, and results, please refer to the relative sections in the "summary" notebook *SQUAD_main.ipynb*. 

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

The implementation of the *Data Preparation* pipeline is available in the directory ```src/data_preparation```

### MODEL
In this section, we present the question-answering model implemented for the SQuAD 2.0 dataset. We considered the *Bi-Directional Attention Flow (BIDAF) model* as our baseline.
<img src="https://github.com/NLP-course-project-2023/BiDAF/blob/main/images/Screenshot%202023-08-12%20163853.png">
Then we introduced various enhancements to capture finer-grained linguistic information, improve contextual understanding, and, hence, generate more accurate answers. These improvements include:
- Information extraction from *part-of-speech and entity recognition embeddings* [[Chen et al., 2017]](https://aclanthology.org/P17-1171.pdf),
- *Iterative Re-Attention* mechanism [[Hu et al., 2018]](https://arxiv.org/pdf/1705.02798.pdf)

The implementation of the different model architectures and associated tools is available in the directory ```src/model```.

### MODEL EVALUATION 
The performance of each model has been evaluated using different metrics and tools.
In particular we 
- retrieve the raw predictions from a trained model, given as input a pre-defined data loader. The prediction at this stage consists directly in the output of the model, namely the 2 arrays obtained after the log_softmax regarding respectively the start and end position probability of the answer. This function provides also additional information regarding the start and end location of each token at character level on the original context, obtained by mapping the original target start and end location over our tokenization;
- process predictions to extract from the raw predictions the actual predicted span and retrieve the answers in textual format. The predicted span is defined by the maximum joint probability of the start-end position pair, identified as the answer score. On the other hand, is computed the score of the question to be unanswerable as the difference between the 0-index pair probability and the score of the prediction. In case the answer score is lower than the unanswerable score, then the prediction is "impossible answer" and the possible answer is provided. This logic has been inspired by [[Zhou et al.]](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1224/reports/default_116657437.pdf);
- evaluate the models using the official function provided by the challenge;
- visualize the results with intuitive graphics.

The implementation of the different model architectures and associated tools is available in the directory ```src/model_evaluation/```.

### RESULTS



### ADDITIONAL
#### Requirements
