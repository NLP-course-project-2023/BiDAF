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
In this section, we present the question-answering model implemented for the SQuAD 2.0 dataset. We considered the *Bi-Directional Attention Flow (BIDAF) model* as our baseline(**BiDAF without character embedding model**).
<img src="https://github.com/NLP-course-project-2023/BiDAF/blob/main/images/Screenshot%202023-08-12%20163853.png">

Then we introduced various enhancements to capture finer-grained linguistic information, improve contextual understanding, and, hence, generate more accurate answers. These improvements include:
- Information extraction from *part-of-speech and entity recognition embeddings* [[Chen et al., 2017]](https://aclanthology.org/P17-1171.pdf) (**BiDAF Original Model**)
- *Iterative Re-Attention* mechanism [[Hu et al., 2018]](https://arxiv.org/pdf/1705.02798.pdf) (**BiDAF Pro Model**)

The implementation of the different model architectures and associated tools is available in the directory ```src/model```.


### EVALUATION
In all its variants, the BiDAF model returns as output a probability distribution over the position of the *start* and *end* tokens for the answer in the text.
Thus, the evaluation of the model is made by comparing the predicted positions with the ground truth. That comparison can be made in an *exact* fashion, or in more flexible ways (e.g., overlapping between predicted and ground truth span), as for a given question there might be multiple reasonable and slightly different answers inside the context paragraph.
Moreover, the probability distribution of start and end positions can be handled in different ways. For example, one can simply consider the predicted position as the one with the highest probability, or combine the most likely position in more sophisticated ways.

The output of the model is the values of $\log(p_{\textbf{start}})$ and $\log(p_{\textbf{end}})$ obtained after the $\text{logsoftmax}$, computed for each model's vocabulary word.
Additionally, for prediction purposes, the start and end char positions of each context token are retrieved. Additionally, the true answer start token and answer end token are retrieved.

Then the actual predicted span is extracted from the raw predictions. This includes the following steps:

1.   Process the $\log(p_{\textbf{start}})$ and $\log(p_{\textbf{end}})$ to get the actual probabilities $p_{\textbf{start}}$ and $p_{\textbf{end}})$
2.   The retrieval of the most probable span over the probabilities.

The 2nd step actually implements the core of the answer retrieval. It takes as input the probability vectors $p_{\textbf{start}}$ and $p_{\textbf{end}})$ and computes the answer span by:

1.   Compute the probability of *impossible answer* as the joint probability $p_{\textbf{start}}0)$ and $p_{\textbf{end}}(0)$ (which is how the model has been trained to encode the impossible answers) and removing those 0-index probabilities from the following evaluations.
2.   Compute  the joint probability of start-end pairs, resulting in a matrix of dimension (number of context tokens, number of context tokens) in which the $(i, j)$ element is the joint probability 
$p_{\textbf{start}}(i) \cdot p_{\textbf{end}}(j)$ 
3. Define the answer span as the pair that maximizes the joint probability, excluding invalid pairs. A pair is considered invalid if its end precedes its start and if the start-end span is greater than the imposed maximum length of the answer.

Finally, the textual answer given the span, is retrieved. To do so, the start and end char positions of each context token retrieved in the prediction step are employed.

The rationale at this point is the following:

1.   Given the answer span from $\boldsymbol{start} = i$  to  $\boldsymbol{end} = j$ (which means that the model identified as an answer from the i-th token to the j-th token on the processed & tokenized context) retrieve the start of the original word on the original context in characters of the i-th token and the end of the j-th one.
2.   Retrieve the answer, extracting the text included in the identified span from the original context.
3. Compute the score of the prediction as the maximum of the joint probability identified before and as score of the question to be unanswerable the difference between the 0-index pair probability and the score of the prediction. Then consider the sample as predicted as impossible if the unanswerable score is above the score of the prediction. (Zhou et al., Neural Question Answering on SQuAD 2.0)
4. Provide the prediction. If the question is predicted as impossible, consider the retrieved answer as plausible.  

The implementation of the evaluation tools and metrics is available in the directory ```src/model_evaluation/```.

### RESULTS

#### BiDAF without character embedding

| **Metric**                  | Value                 |
|-------------------------|-----------------------|
| **Exact Match**             | 46.593110418596815    |
| **F1 Score**                | 51.93552841837668     |
| **Total**                   | 11873                 |
| **Has Answer Exact Match**  | 45.86707152496626     |
| **Has Answer F1 Score**     | 56.56722822391746     |
| **Has Answer Total**        | 5928                  |
| **No Answer Exact Match**   | 47.31707317073171     |
| **No Answer F1 Score**      | 47.31707317073171     |
| **No Answer Total**         | 5945                  |


#### BiDAF Original

| **Metric**                  | Value                 |
|-------------------------|-----------------------|
| **Exact Match**             | 50.24004042786154     |
| **F1 Score**                | 55.67969814269031     |
| **Total**                   | 11873                 |
| **Has Answer Exact Match**  | 46.912955465587046    |
| **Has Answer F1 Score**     | 57.807870453467906    |
| **Has Answer Total**        | 5928                  |
| **No Answer Exact Match**   | 53.55761143818335     |
| **No Answer F1 Score**      | 53.55761143818335     |
| **No Answer Total**         | 5945                  |

#### BiDAF Pro


| **Metric**                  | Value                 |
|-------------------------|-----------------------|
| **Exact Match**             | 51.02333024509391     |
| **F1 Score**                | 55.85737482217886     |
| **Total**                   | 11873                 |
| **Has Answer Exact Match**  | 48.431174089068826    |
| **Has Answer F1 Score**     | 58.113126056634066    |
| **Has Answer Total**        | 5928                  |
| **No Answer Exact Match**   | 53.6080740117746      |
| **No Answer F1 Score**      | 53.6080740117746      |
| **No Answer Total**         | 5945                  |

