# BiDAF: Bi-Directional Attention Flow network for SQuAD 2.0 question-answering
In this repo we present a PyTorch implementation of the *Bidirectional Attention Flow network* described in [[Minjoon Seo et al., Bidirectional Attention Flow for Machine Comprehension]](https://arxiv.org/abs/1611.01603).

### DATA
The model presented here was designed for question-answering on the updated version of the Stanford Question Answering Dataset (SQuAD 2.0) [[Rajpurkar et al.,2018]](https://arxiv.org/abs/1806.03822). 
![SQUAD2.0 Logo](https://www.google.com/search?sca_esv=556313000&rlz=1C1VDKB_itIT949IT949&sxsrf=AB5stBghdSrFqGhW39dAdZ_oMPursckCRg:1691848635615&q=squad2.0+dataset&tbm=isch&source=lnms&sa=X&ved=2ahUKEwjZisGno9eAAxWegf0HHTgyBBgQ0pQJegQICxAB&biw=767&bih=707&dpr=1.25#imgrc=Partenqdrn0FMM)
The dataset comprises pairs of questions and context paragraphs from which the answer should be retrieved. Differently from the original version, SQuAD 2.0 includes also unanswerable questions, i.e., questions whose answer is not directly provided in the associated context paragraph. The dataset consists of approximately 150k questions, of which ~50% are unanswerable.

The data is available [here](https://rajpurkar.github.io/SQuAD-explorer/).

### DATA PREPARATION

For more  used for *Data Preparation* is available in the directory:
```
src/data_preparation
```

### MODEL

### RESULTS

### ADDITIONAL
#### Requirements
