"""

In this section, we present the design and architecture of our question-answering models for the SQuAD 2.0 dataset.
We started with the Bi-Directional Attention Flow (BIDAF) model without character embeddings as our baseline.
We then systematically introduced various improvements to enhance the model's capabilities.

These improvements include the incorporation of character embeddings, token features embeddings (part-of-speech tags
and entity recognition), and an iterative reattention mechanism.

By integrating these enhancements, we aimed to capture finer-grained linguistic information, improve contextual understanding,
and generate more accurate answers. In the following sections, we provide a detailed explanation of the baseline model's components
and functionality, followed by a comprehensive description of each improvement we made to further enhance its performance.

"""

from typing import Tuple, Optional
import torch
from torch import nn
from layers import *


"""


***BIDAF BASE***
The baseline model is a based on Bidirectional Attention Flow (BiDAF) , a closed-domain, extractive, factoid Q&A model. 

The original BiDAF model uses learned character-level word embeddings in addition to the word-level embeddings. 
Unlike the original BiDAF model, our baseline implementation does not include a character-level embedding layer.

The model is composed by 5 layers: Word Embed Layer, Phrase Embed Layer, Attention Flow Layer, Modeling Layer and Output Layer.
It uses both context-to-query and query-to-context attention
Output Layer predicts start and end positions within the context where the answer lies

***BIDAF ORIGINAL***
To enhance the baseline BIDAF model, we introduced the integration of character embeddings into the model architecture.

By considering character-level representations alongside word embeddings, the model gains the ability to leverage fine-grained 
information encoded in the characters themselves, due to the fact that character-level embeddings allow us to condition 
on the internal structure of word. This improvement enables the model to capture subtle linguistic patterns, 
handle out-of-vocabulary words more effectively, and enhance its understanding of the input text.

***BIDAF PRO***
Starting from the previous improvement of incorporating character embeddings, we further enhanced the BIDAF model 
by integrating additional contextual information, following the ideas ideas from Chen et al (https://aclanthology.org/P17-1171.pdf) 
through the inclusion of POS embeddings and ENT embeddings. 

These embeddings capture syntactic and semantic features of the input text, providing the model with a richer understanding of the language. 
In addition, we introduced two Iterative Reattention Blocks, following the idea of reattention mechanism from 
Hu et al (https://arxiv.org/pdf/1705.02798.pdf) to enable the model to iteratively refine its attention mechanism and 
gather more precise and relevant information from the context and question.
"""


class BiDAF(nn.Module):
    """
    BiDAF (Bidirectional Attention Flow) model for machine comprehension.

    Attributes:
        emb (nn.Module): Word embedding module.
        enc (RNNEncoder): RNN encoder module.
        att (BiDAFAttention): Bidirectional attention module.
        mod (RNNEncoder): RNN encoder module for modeling.
        out (BiDAFOutput): Output layer module.
        model_type (str): Type of model ('base', 'original', or 'pro').
    """

    def __init__(self,
                 word_vectors: torch.Tensor,
                 hidden_size: int,
                 model_type: str = 'base',
                 alphabet_size: Optional[int] = None,
                 char_embed_dim: Optional[int] = None,
                 char_channel_width: Optional[int] = None,
                 drop_prob: float = 0.0):

        """
        Initialize the BiDAF model.

        Parameters:
            word_vectors (torch.Tensor): Pre-trained word vectors.
            hidden_size (int): Size of hidden activations.
            model_type (str, optional): Model type to use. Can be 'base', 'original', or 'pro'.
            alphabet_size (int, optional): Size of character alphabet.
            char_embed_dim (int, optional): Dimension of character embeddings.
            char_channel_width (int, optional): Width of character channels.
            drop_prob (float, optional): Dropout probability.
        """

        super(BiDAF, self).__init__()

        if model_type == 'base':
            self.emb = WordEmbedding(word_vectors = word_vectors,
                                     hidden_size = hidden_size,
                                     drop_prob = drop_prob)
        elif model_type == 'original':
            self.emb = WCEmbedding(word_vectors = word_vectors,
                                   hidden_size = hidden_size,
                                   alphabet_size = alphabet_size,
                                   in_embedding_dim = char_embed_dim,
                                   char_channel_width = char_channel_width,
                                   drop_prob = drop_prob)

        else:
            self.emb = ProEmbedding(word_vectors = word_vectors,
                                    hidden_size = hidden_size,
                                    alphabet_size = alphabet_size,
                                    in_embedding_dim = char_embed_dim,
                                    char_channel_width = char_channel_width,
                                    drop_prob = drop_prob)

        i_size = hidden_size
        if model_type == 'original':
            i_size = 2 * hidden_size
        elif model_type == 'pro':
            i_size = 4 * hidden_size

        self.enc = RNNEncoder(input_size = i_size,
                              hidden_size = hidden_size,
                              num_layers = 1,
                              drop_prob = drop_prob)

        self.att = BiDAFAttention(hidden_size = 2 * hidden_size,
                                  drop_prob = drop_prob)

        if model_type == 'pro':
            self.irb1 = IterativeReattentionBlock(embedding_dim = hidden_size, hidden_dim = hidden_size)
            self.irb2 = IterativeReattentionBlock(embedding_dim = hidden_size, hidden_dim = hidden_size)

        self.mod = RNNEncoder(input_size = 8 * hidden_size,
                              hidden_size = hidden_size,
                              num_layers = 2,
                              drop_prob = drop_prob)

        self.out = BiDAFOutput(hidden_size = hidden_size,
                               drop_prob = drop_prob)

        self.model_type = model_type

    def forward(self,
                cw_idxs: torch.Tensor,
                qw_idxs: torch.Tensor,
                cc_idxs: Optional[torch.Tensor] = None,
                qc_idxs: Optional[torch.Tensor] = None,
                c_pos_vectors: Optional[torch.Tensor] = None,
                q_pos_vectors: Optional[torch.Tensor] = None,
                c_ent_idxs: Optional[torch.Tensor] = None,
                q_ent_idxs: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Define the forward pass for the BiDAF model.

        Parameters:
            cw_idxs (torch.Tensor): Word indices for the context.
            qw_idxs (torch.Tensor): Word indices for the question.
            cc_idxs (torch.Tensor, optional): Character indices for the context.
            qc_idxs (torch.Tensor, optional): Character indices for the question.
            c_pos_vectors (torch.Tensor, optional): POS vectors for the context.
            q_pos_vectors (torch.Tensor, optional): POS vectors for the question.
            c_ent_idxs (torch.Tensor, optional): Entity indices for the context.
            q_ent_idxs (torch.Tensor, optional): Entity indices for the question.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of two tensors representing start and end logits.
        """

        # c_mask and q_mask will be boolean tensors with the same shape as cw_idxs and qw_idxs, respectively.
        # The values in these tensors will be True where the corresponding elements in cw_idxs and qw_idxs are
        # non-zero, and False where the corresponding elements in cw_idxs and qw_idxs are zero.
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        if self.model_type == 'base':
            c_emb = self.emb(cw_idxs)  # (batch_size, c_len, hidden_size)
            q_emb = self.emb(qw_idxs)  # (batch_size, q_len, hidden_size)

        elif self.model_type == 'original':
            c_emb = self.emb(cw_idxs, cc_idxs)  # (batch_size, c_len, hidden_size)
            q_emb = self.emb(qw_idxs, qc_idxs)  # (batch_size, q_len, hidden_size)

        else:
            c_ent_idxs, q_ent_idxs = torch.unsqueeze(c_ent_idxs, -1), torch.unsqueeze(q_ent_idxs, -1)

            c_emb = self.emb(cw_idxs, cc_idxs, c_pos_vectors, c_ent_idxs)  # (batch_size, c_len, hidden_size)
            q_emb = self.emb(qw_idxs, qc_idxs, q_pos_vectors, q_ent_idxs)  # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)  # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)  # (batch_size, q_len, 2 * hidden_size)

        if self.model_type == 'pro':
            c_irenc = self.irb1(c_enc, q_enc)
            c_enc = self.irb2(c_irenc, q_enc)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)  # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)  # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
