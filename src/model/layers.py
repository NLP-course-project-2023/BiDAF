from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# --------------------------------------------------------------------------------------------------------------------------------------------
def masked_softmax(logits: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   log_softmax: bool = False) -> torch.Tensor:
    """Take the softmax of `logits` over given dimension, and set
    entries to 0 wherever `mask` is 0.

    Args:
        logits (torch.Tensor): Inputs to the softmax function.
        mask (torch.Tensor): Same shape as `logits`, with 0 indicating
            positions that should be assigned 0 probability in the output.
        dim (int): Dimension over which to take softmax.
        log_softmax (bool): Take log-softmax rather than regular softmax.
            E.g., some PyTorch functions such as `F.nll_loss` expect log-softmax.

    Returns:
        probs (torch.Tensor): Result of taking masked softmax over the logits.
    """

    mask = mask.type(torch.float32)
    masked_logits = mask * logits + (1 - mask) * -1e30
    softmax_fn = F.log_softmax if log_softmax else F.softmax
    probs = softmax_fn(masked_logits, dim)

    return probs
# --------------------------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------------------------------

"""
We implemented the character embed layer as in the original BiDAF paper, following the original CNN embeding structure 
in the paper by Kim(https://aclanthology.org/D14-1181.pdf). Each character are first embeded as a trainable vector. 
The character embeddings are convoluted. This convoluted
matrix is maxpooled along the word span in each channel.

"""

class CharEmbedding(nn.Module):
    """
    Character embedding module based on a CNN.

    This module first embeds characters into dense vectors. These embeddings are then passed through a
    CNN layer, and the result is max-pooled over the word dimension.

    Args:
        alphabet_size (int): Size of the character vocabulary.
        in_embedding_dim (int, optional): Size of initial character embeddings. Default is 8.
        out_embedding_dim (int, optional): Size of the output embeddings after the CNN. Default is 100.
        char_channel_width (int, optional): Width of the CNN filter. Default is 5.
        dropout_prob (float, optional): Dropout probability. Default is 0.5.
    """

    def __init__(self,
                 alphabet_size: int,
                 in_embedding_dim: int = 8,
                 out_embedding_dim: int = 100,
                 char_channel_width: int = 5,
                 dropout_prob: float = 0.5) -> None:
        super(CharEmbedding, self).__init__()
        # store parameters
        self.alphabet_size = alphabet_size
        self.in_embedding_dim = in_embedding_dim
        self.out_embedding_dim = out_embedding_dim
        self.char_channel_width = char_channel_width

        # embedding layer -> associate to each character a random embedding vector
        # of size embedding_dim
        self.char_emb = nn.Embedding(alphabet_size, in_embedding_dim, padding_idx = 0)
        nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)

        # convolution layer
        self.char_conv = nn.Sequential(
                nn.Conv2d(in_channels = 1, out_channels = out_embedding_dim,
                          kernel_size = (in_embedding_dim, char_channel_width)),
                nn.ReLU()
        )

        self.dropout = nn.Dropout(p = dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        -----
        x: (torch.tensor)
            The current batch, tensor of size (batch, seq_len, word_len)
        """
        batch_size = x.size(0)
        # curr_size: (batch, seq_len, word_len)
        x = self.char_emb(x)
        x = self.dropout(x)
        # curr_size: (batch, seq_len, word_len, embedding_dim)
        x = x.transpose(2, 3)
        # curr_size: (batch, seq_len, in_embedding_dim, word_len) -> put character vectors in columns
        x = x.view(-1, self.in_embedding_dim, x.size(3)).unsqueeze(1)
        # curr_size: (batch * seq_len, 1, in_embedding_dim, word_len) -> split by words and not by batch first
        x = self.char_conv(x).squeeze()
        # curr_size: (batch * seq_len, out_embedding_dim, 1, conv_len) -> (batch * seq_len, out_embedding_dim, conv_len)
        x = F.max_pool1d(x, x.size(2)).squeeze()
        # curr_size: (batch * seq_len, _embedding_dim, 1) -> (batch * seq_len, out_embedding_dim)
        x = x.view(batch_size, -1, self.out_embedding_dim)
        # curr_size: (batch, seq_len, char_channel_size)

        return x

class WCEmbedding(nn.Module):
    """
    Word and Character Embedding module.

    This module combines word embeddings with character embeddings for each word.

    Args:
        word_vectors (torch.Tensor): Pre-trained word embeddings.
        hidden_size (int): Size of the output embeddings.
        alphabet_size (int): Size of the character vocabulary.
        in_embedding_dim (int, optional): Size of initial character embeddings. Default is 8.
        char_channel_width (int, optional): Width of the CNN filter for character embedding. Default is 5.
        drop_prob (float, optional): Dropout probability. Default is 0.5.
    """

    def __init__(self, word_vectors, hidden_size, alphabet_size, in_embedding_dim = 8, char_channel_width = 5,
                 drop_prob = 0.5):
        super(WCEmbedding, self).__init__()

        self.char_emb = CharEmbedding(alphabet_size = alphabet_size,
                                      in_embedding_dim = in_embedding_dim,
                                      out_embedding_dim = hidden_size,
                                      char_channel_width = char_channel_width,
                                      dropout_prob = drop_prob
                                      )
        self.word_emb = WordEmbedding(word_vectors = word_vectors,
                                      hidden_size = hidden_size,
                                      drop_prob = drop_prob)

        self.hwy = HighwayEncoder(2, 2 * hidden_size)

    def forward(self, x: torch.Tensor, ch: torch.Tensor) -> torch.Tensor:
        w_emb = self.word_emb(x, hwy = False)  # (batch_size, seq_len, embed_size)
        c_emb = self.char_emb(ch)
        concat_emb = torch.cat((c_emb, w_emb), 2)
        emb = self.hwy(concat_emb)
        return emb
# --------------------------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------------------------------

"""
Given some input word indices (which are essentially integers that indicate where in the embedding matrix you can find 
the word's embeddings), the embedding layer uses a GLOVE embedding lookup. This turns these indices into word embeddings. 
Both the context and the question undergo this process, producing embeddings for each.

In the embedding layer, these embeddings are further refined in two steps:

1. Every embedding is projected to have a dimensionality, commonly referred to as H. There's a learnable matrix, W_proj, 
used for this. So, for an embedding vector v_i, its new representation becomes h_i = W_proj * v_i.

2. Next, a Highway Network (as described in a provided paper) is used to refine these embeddings even more. 
For an input vector h_i, a one-layer highway network computes:

   - Gate vector g = sigmoid(W_g * h_i + b_g)
   - Transform vector t = ReLU(W_t * h_i + b_t)
   - The refined embedding h_i' = g * t + (1 - g) * h_i

Here, W_g and W_t are learnable matrices, while b_g and b_t are learnable vectors. The symbols * represent 
matrix multiplication. The term "g" stands for "gate" and "t" is for "transform". 
This highway transformation is applied twice (making it a two-layer network) using different sets of learnable parameters for each round.

"""

class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """

    def __init__(self,
                 num_layers: int,
                 hidden_size: int) -> None:
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x

class WordEmbedding(nn.Module):
    """
    Embedding layer used by BiDAF, without the character-level component.
    This module transforms pre-trained word embeddings using a projection layer and
    optionally refines them with a Highway network.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self,
                 word_vectors: torch.Tensor,
                 hidden_size: int,
                 drop_prob: float) -> None:
        super(WordEmbedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias = False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x: torch.Tensor, hwy: bool = True) -> torch.Tensor:
        emb = self.embed(x)  # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        if hwy:
            emb = self.hwy(emb)  # (batch_size, seq_len, hidden_size)
        return emb
# --------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------------

"""


**RNN Encoder (Phrase Embed Layer)**
The encoder layer receives the output from the embedding layer and employs a bidirectional LSTM to factor in temporal 
dependencies between the steps of the embedding layer's output. The encoded result is the RNN's hidden state at each position.

For each step i:
- The forward hidden state, denoted as h_i_fwd', is determined using: LSTM of (h_(i-1)', h_i)
  
- The reverse hidden state, termed h_i_rev', is calculated using: LSTM of (h_(i+1)', h_i)
  
- The overall hidden state at step i, labeled h_i', is the combination of the forward and reverse hidden states: 
  h_i' = [h_i_fwd' ; h_i_rev']

It's crucial to realize that the size of h_i' is twice that of the original size H, as it combines both the forward 
and reverse hidden states at step i.

**RNNEncoder (Modeling Layer)**

The modeling layer focuses on refining the vector sequence post-attention. Since the modeling layer is sequenced after 
the attention layer, by the time the context representations get to the modeling layer, they are conditioned based on the question. 
This means the modeling layer incorporates temporal data between context representations based on the question. 
The approach is similar to the Encoder layer, utilizing a bidirectional LSTM. 

For every step i:
- The forward modeling state, labeled as m_i_fwd, is determined using: LSTM of (m_(i-1), g_i)
  
- The reverse modeling state, termed m_i_rev, is computed using: LSTM of (m_(i+1), g_i)
  
- The overall modeling state at step i, denoted as m_i, combines both the forward and reverse modeling states: m_i = [m_i_fwd ; m_i_rev]

It's essential to note the difference between the modeling and encoder layers. While a single-layer LSTM is used in the 
encoder layer, a two-layer LSTM is employed in the modeling layer. 


"""
class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    This module encodes a sequence using a bidirectional RNN, which considers both past
    and future context for each timestep in the sequence.
    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 drop_prob: float = 0.0) -> None:
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first = True,
                           bidirectional = True,
                           dropout = drop_prob if num_layers > 1 else 0.)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending = True)
        x = x[sort_idx]  # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths.to('cpu'), batch_first = True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first = True, total_length = orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]  # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x
# --------------------------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------------------------------

"""

The layer takes in POS (Part Of Speech) embeddings, represented as p_1 to p_N, each of size D.
In the embedding layer, the embeddings undergo further refinement through a two-step procedure:
Every embedding is projected to have a dimensionality of H. To achieve this, there's a learnable matrix, W_proj, of size H x D. 
Each embedding vector, denoted as v_i, is transformed using W_proj to produce a new vector, h_i. This new vector h_i is of size H.

"""

class PosEmbedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 drop_prob: float) -> None:
        super(PosEmbedding, self).__init__()
        self.drop_prob = drop_prob
        self.proj = nn.Linear(input_size, hidden_size, bias = False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine the embeddings and return the transformed embeddings."""
        emb = F.dropout(x, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        return emb
# --------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------------
class ProEmbedding(nn.Module):
    """
    Combines character, word, and POS embeddings.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Hidden size for the embedding layers.
        alphabet_size (int): The size of the character alphabet.
        in_embedding_dim (int, optional): Input embedding dimensionality. Default is 8.
        char_channel_width (int, optional): Width of the character channel. Default is 5.
        drop_prob (float, optional): Dropout probability for regularization. Default is 0.5.
    """

    def __init__(self,
                 word_vectors: torch.Tensor,
                 hidden_size: int,
                 alphabet_size: int,
                 in_embedding_dim: int = 8,
                 char_channel_width: int = 5,
                 drop_prob: float = 0.5) -> None:
        super(ProEmbedding, self).__init__()

        self.char_emb = CharEmbedding(alphabet_size = alphabet_size,
                                      in_embedding_dim = in_embedding_dim,
                                      out_embedding_dim = hidden_size,
                                      char_channel_width = char_channel_width,
                                      dropout_prob = drop_prob
                                      )
        self.word_emb = WordEmbedding(word_vectors = word_vectors,
                                      hidden_size = hidden_size,
                                      drop_prob = drop_prob)

        self.pos_emb = PosEmbedding(input_size = word_vectors.size(1),
                                    hidden_size = hidden_size,
                                    drop_prob = drop_prob)

        self.ent_proj = nn.Linear(1, hidden_size, bias = False)

        self.hwy = HighwayEncoder(2, 4 * hidden_size)  #

    def forward(self, w: torch.Tensor, ch: torch.Tensor, pos: torch.Tensor, ent: torch.Tensor) -> torch.Tensor:
        w_emb = self.word_emb(w, hwy = False)  # (batch_size, seq_len, embed_size)
        c_emb = self.char_emb(ch)
        p_emb = self.pos_emb(pos)
        e_emb = self.ent_proj(ent)
        concat_emb = torch.cat((e_emb, p_emb, c_emb, w_emb), 2)
        emb = self.hwy(concat_emb)
        return emb
# --------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------------

"""

The Query and Context representations then enter the attention and modeling layers. 
These layers use several matrix operations to fuse the information contained in the Query and the Context. 
The output of these steps is another representation of the Context that contains information from the Query. 
This output is referred as the “Query-aware Context representation.

"""
class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self,
                 hidden_size: int,
                 drop_prob: float = 0.1) -> None:
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c: torch.Tensor, q: torch.Tensor, c_mask: torch.Tensor, q_mask: torch.Tensor) -> torch.Tensor:
        """Compute the bidirectional attention between context and query."""
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)  # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim = 2)  # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim = 1)  # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim = 2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603

        Args:
            c (torch.Tensor): Context tensor.
            q (torch.Tensor): Query tensor.

        Returns:
            torch.Tensor: The similarity matrix.
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob,
                      self.training)  # (bs, c_len, hid_size) #Real shape is torch.Size([1825, 64, 200])
        q = F.dropout(q, self.drop_prob,
                      self.training)  # (bs, q_len, hid_size) #Real shape is torch.Size([11, 64, 200])

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2).expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s
# --------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------------

"""

The output layer's primary function is to generate a probability vector for each position in the context. 
This results in two vectors: p_start and p_end, both having a length of N. 
As indicated by their names, p_start(i) predicts the probability that the answer begins at the i-th position, 
and p_end(i) predicts the probability that the answer concludes at the same position. 
(For predictions where no answer is given, refer to the 'Predicting no-answer' section).

To get into specifics:

1. The output layer receives its input from two sources:
   - Outputs from the attention layer, which are vectors g_1 to g_N, each of size 8H.
   - Outputs from the modeling layer, which are vectors m_1 to m_N, each of size 2H.
   
2. The output layer then processes the modeling layer outputs using a bidirectional LSTM. For each vector m_i:
   - The forward state m_i_fwd is calculated as: LSTM(m_(i-1)_prime, m_i)
   - The reverse state m_i_rev is derived from: LSTM(m_(i+1)_prime, m_i)
   - The combined state at i, m_i_prime, is a combination of m_i_fwd and m_i_rev.

3. Further processing involves organizing the vectors into matrices:
   - G is a matrix sized 8H x N with columns from g_1 to g_N.
   - M and M_prime are matrices sized 2H x N. M contains columns from m_1 to m_N, while M_prime contains columns from m_1_prime to m_N_prime.

4. The final step to produce p_start and p_end involves:
   - For p_start: Apply softmax to the result of multiplying W_start with the combination of G and M.
   - For p_end: Apply softmax to the result of multiplying W_end with the combination of G and M_prime.
   
   Here, W_start and W_end (both of size 1 x 10H) are parameters that the model learns over time.

In the implementation, it's crucial to mention that the softmax function uses a context mask. 
All probabilities are computed in logarithmic space to ensure numerical stability. 
This is also because the F.nll_loss function expects log-probabilities.

"""

class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self,
                 hidden_size: int,
                 drop_prob: float) -> None:
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size = 2 * hidden_size,
                              hidden_size = hidden_size,
                              num_layers = 1,
                              drop_prob = drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att: torch.Tensor, mod: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the start and end positions for the answer."""
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax = True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax = True)

        return log_p1, log_p2
# --------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------------

"""
In this layer, to integrate the attention from the question into the context, a similarity matrix between C_align^(i-1)
 and Q_emb is initially computed as:

S_CQ^i = Conv1d(C_align^(i-1)) * Conv1d(Q_emb^T)

Here, S_CQ^i has dimensions T x J. The attended query vector is then derived by:

A_Q^i = softmax(S_CQ^i) * Q_emb

In this case, A_Q^i has dimensions T x 2d. Following this, a fusion function blends the information from C_align^(i-1) 
and the attended query vector A_Q^i. Let's call this function Fusion(x, y) and its output as o. Then:

g = ReLU(Conv1d[x; y; x * y; x-y])
h = sigmoid(Conv1d[x; y; x * y; x-y])
o = h * g + (1-h) * x

Here:
- "sigmoid" refers to the sigmoid activation function.
- "*" denotes element-wise multiplication (or the Hadamard product).
- "h" acts as a gate to regulate the contribution from the two different vectors.

Finally, the output of the fusion function is represented as:

H_CQ^i = Fusion(C_align^(i-1), A_Q^i).

"""

class InteractiveAlignment(nn.Module):
    """
    Integrates attention from the question into the context.

    It computes a similarity matrix between the aligned context and the embedded question.
    This matrix is then used to obtain an attended query vector, which is fused with the
    aligned context to produce the final output.

    Args:
        input_channels1 (int): Number of channels in the input tensor for context alignment.
        input_channels2 (int): Number of channels in the input tensor for question embedding.
        output_channels (int, optional): Number of channels in the output tensor. Default is 128.
    """

    def __init__(self,
                 input_channels1: int,
                 input_channels2: int,
                 output_channels: int = 128) -> None:
        super(InteractiveAlignment, self).__init__()
        self.conv_c = nn.Conv1d(input_channels1, output_channels, stride = 1, kernel_size = 1)
        self.conv_q = nn.Conv1d(input_channels2, output_channels, stride = 1, kernel_size = 1)
        self.fusion = FusionBlock(input_channels1 * 4, input_channels1)

    def forward(self,
                c: torch.Tensor,  # [bs, len_c, dim]
                q: torch.Tensor) -> torch.Tensor:  # [bs, len_q, dim]
        """Compute the attended query and blend with context for final output."""
        # attention
        c_ = F.relu(self.conv_c(c.permute(0, 2, 1)))  # [bs, dim, len_c]
        c_ = c_.permute(0, 2, 1)  # [bs, len_c, dim]
        q_ = F.relu(self.conv_q(q.permute(0, 2, 1)))  # [bs, dim, len_q]
        q_ = q_.permute(0, 2, 1)  # [bs, len_q, dim]
        S_cq = torch.bmm(c_, q_.permute(0, 2, 1))  # [bs, len_q, len_c]
        A_q = F.softmax(S_cq, dim = 2)  # [bs, len_q, len_c]
        A_q = torch.bmm(A_q, q)  # [bs, len_c, dim]

        # fusion
        return self.fusion(c, A_q)  # [bs, len_c, dim]

class FusionBlock(nn.Module):
    """
    A block that fuses two vectors using a gating mechanism.

    The fusion is achieved using element-wise multiplication and a combination of ReLU and
    sigmoid activation functions. The resulting output is a blend of the input vectors.

    Args:
        input_channels (int): Number of channels in the input tensor.
        output_channels (int): Number of channels in the output tensor.
    """

    def __init__(self,
                 input_channels: int,
                 output_channels: int) -> None:
        super(FusionBlock, self).__init__()
        self.conv_x = nn.Conv1d(input_channels, output_channels, stride = 1, kernel_size = 1)
        self.conv_h = nn.Conv1d(input_channels, output_channels, stride = 1, kernel_size = 1)

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Fuse the input vectors and produce the blended output."""
        # fusion

        uv = torch.cat([u, v, u * v, u - v], dim = -1)
        x = F.relu(self.conv_x(uv.permute(0, 2, 1)))
        x = x.permute(0, 2, 1)
        g = F.sigmoid(self.conv_h(uv.permute(0, 2, 1)))
        g = g.permute(0, 2, 1)
        h = g * x + (1 - g) * u  # [bs, len_c, dim]
        return h
# ---------------------------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------------------------------

"""

In this layer, to get the context vector that's attentive to itself, similar to the Interactive Alignment Layer, 
a self-similarity matrix is created as:

S_self^i = Conv1d(H_CQ^i) * Conv1d(H_CQ^(T i))

Here, S_self^i has dimensions T x T. The attended context vector is then derived by:

A_self^i = softmax(S_self^i) * H_CQ^i

In this case, A_self^i has dimensions T x 2d. Following this, the fusion function is used once more to obtain 
the self-aware context vector:

H_self^i = Fusion(H_CQ^i, A_self^i).

"""

class SelfAlignment(nn.Module):
    """
    A layer that gets the context vector attentive to itself.

    It computes a self-similarity matrix between the context vectors and uses it
    to derive a self-attended context vector. This vector is then fused with the
    original context to produce the self-aware context vector.

    Args:
        input_channels (int): Number of channels in the input tensor.
        output_channels (int, optional): Number of channels in the output tensor. Default is 128.
    """

    def __init__(self,
                 input_channels: int,
                 output_channels: int = 128) -> None:
        super(SelfAlignment, self).__init__()
        self.conv_h_1 = nn.Conv1d(input_channels, output_channels, stride = 1, kernel_size = 1)
        self.conv_h_2 = nn.Conv1d(input_channels, output_channels, stride = 1, kernel_size = 1)
        self.fusion = FusionBlock(input_channels * 4, input_channels)

    def forward(self,
                h: torch.Tensor) -> torch.Tensor:
        """Compute the self-attended context vector and produce the final output."""
        # attention
        h_1 = F.relu(self.conv_h_1(h.permute(0, 2, 1)))  # [bs, dim, len_h]
        h_1 = h_1.permute(0, 2, 1)  # [bs, len_h, dim]
        h_2 = F.relu(self.conv_h_2(h.permute(0, 2, 1)))  # [bs, dim, len_h]
        h_2 = h_2.permute(0, 2, 1)  # [bs, len_h, dim]

        S = torch.bmm(h_2, h_1.permute(0, 2, 1))  # [bs, len_h, len_h]
        A_s = F.softmax(S, dim = 2)  # [bs, len_h, len_h]
        A_s = torch.bmm(A_s, h)  # [bs, len_h, dim]

        # fusion
        return self.fusion(h, A_s)  # [bs, len_h, dim]
# --------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------------

"""

In this layer, a two-layer bidirectional LSTM is applied to gather information from the self-aware context vector H_self^i. 

This is done as:

C_align^i = BiLSTM(H_self^i).

"""

class EvidenceCollection(nn.Module):
    """
    Applies a two-layer bidirectional LSTM to gather information from the input tensor.

    This module is responsible for aligning the information in the input tensor.
    Specifically, the transformation is given by:

    C_align^i = BiLSTM(H_self^i).

    Args:
        embedding_dim (int): Dimensionality of the input tensor.
        hidden_dim (int): Hidden size used in the BiLSTM.
        dropout (float, optional): Dropout probability for regularization. Default is 0.0.
    """

    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 dropout: float = 0.0) -> None:
        super(EvidenceCollection, self).__init__()
        self.BiLSTM = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_dim, num_layers = 2, dropout = dropout,
                              bidirectional = True)

    def forward(self,
                c: torch.Tensor) -> torch.Tensor:  # [bs, len_c, dim]
        """Process the input tensor using the bidirectional LSTM."""

        # bilstm wants  (len_c,bs,dim)
        c = c.permute(1, 0, 2)
        output, (hn, cn) = self.BiLSTM(c)  # [len_c, bs, dim]

        return output.permute(1, 0, 2)  # [bs, len_c, dim]
# --------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------------
class IterativeReattentionBlock(nn.Module):
    """
    Iteratively reapplies attention mechanisms to refine context representations.

    This block integrates attention from the question into the context, applies self-attention
    to the context, and then gathers evidence from the self-aware context vector.

    Args:
        embedding_dim (int): Dimensionality of the input tensor.
        hidden_dim (int): Hidden size used in the attention mechanisms.
    """

    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int) -> None:
        super(IterativeReattentionBlock, self).__init__()

        self.interactiveAlignment = InteractiveAlignment(2 * embedding_dim, 2 * embedding_dim, hidden_dim)
        self.selfAlignment = SelfAlignment(2 * embedding_dim, hidden_dim)
        self.evidenceCollection = EvidenceCollection(2 * embedding_dim, embedding_dim)

    def forward(self, c: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Compute the refined context representations using iterative reattention."""
        H_cq = self.interactiveAlignment(c, q)
        H_s = self.selfAlignment(H_cq)
        c_new = self.evidenceCollection(H_s)

        return c_new
# --------------------------------------------------------------------------------------------------------------------------------------------
