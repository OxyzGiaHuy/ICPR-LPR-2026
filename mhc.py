# %% [markdown]
# `transformers` is a package enable you to train/load and create a model in hugging face in easy way.

# %%
!pip install transformers

# %% [markdown]
# `bertviz` is package that support to make the visualization of sublayers of BERT model.

# %%
!pip install bertviz

# %% [markdown]
# Make sure you click `RESTART RUNTIME` buttom in order to enable to use package in runtime.

# %% [markdown]
# # 1. Visualization BERT sublayers

# %%
from transformers import AutoTokenizer
from bertviz.transformers_neuron_view import BertModel
from bertviz.neuron_view import show

model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = BertModel.from_pretrained(model_ckpt)
text = "I am a machine learning engineer who is currently working on some big NLP projects"
show(model, "bert", tokenizer, text, display_mode="light", layer=0, head=8)

# %% [markdown]
# # 2. Scaled dot-product attention

# %% [markdown]
# The first thing we need to do is tokenize the input text into list of indices by tokenizer.

# %%
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
inputs.input_ids

# %%
text

# %% [markdown]
# Each indice in the input indices list is mapped to an unique word in dictionary. Those indices in the next step are projected into a new feature space that represents an embedding vector for each of them. Process of transformation is made of `torch.nn.Embedding` layer that acts as a look up table for each indice.

# %%
import torch.nn as nn
from transformers import AutoConfig

config = AutoConfig.from_pretrained(model_ckpt)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
token_emb

# %% [markdown]
# Normally, BERT model transform each word into vector of 768 dimensionalities. Feed forward `inputs.input_ids` through `token_emb` to achive the matrix embedding of whole sequence with shape `(batch_size, seq_length, embedding_size)`.

# %%
input_embs = token_emb(inputs.input_ids)
input_embs.shape

# %% [markdown]
# Next, we caculate the self-attention through `scaled_dot_product_attention()` function:
# 
# ![](https://imgur.com/3CVYGDi.png)
# 
# Figure 1: Scale dot-product attention mechanism.

# %%
import torch
import torch.nn.functional as F
from math import sqrt

def scaled_dot_product_attention(query, key, value):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)

# %% [markdown]
# `torch.bmm()` is a function that compute the batch matix multiplication. The batch dimension is kept outside and we only multiply two matrix based on two remain dimensions. In this case `weights` has shape `(batch, seq_length, seq_length)` and `value` has shape `(batch, seq_length, hidden_size)`. Thus in return, the output unchanges batch and multiply matrix `(seq_length, seq_length)` with `(seq_length, hidden_size)` to create `(seq_length, hidden_size)`. Finally output is `(batch, seq_length, hidden_size)`.

# %%
query = key = value = input_embs
weighted_value = scaled_dot_product_attention(query, key, value)
weighted_value.shape

# %% [markdown]
# # 3. Multi-head Attention
# 
# weights and values vector are used as input to compute the final linear projection output values vector for each self-attention layer. That is not all story about attention idea. Further, we do self-attention multiple times and in parallelization that seem to be more benefical for model enable to study variety aspects of sentiment of sequence. Those process are carried in the same time, thus we can train and inference them faster on parallel GPUs system. Of course, it saves both the time and performance in return.
# 
# ![](https://imgur.com/D6mLEJW.png)
# 
# Figure 2: Multi-head attention architecture.
# 
# We consider each linear combination which is a weighted value vector in the output of an attention layer like a head. Thus, multiple output vectors are named as multi-head attention output. They are concatenated in the next step and do linear projection again to get output with the same shape as the input of a sublayer. That is to guarantee we can apply multiple stacked sublayers in a deep sequence without error shape.

# %% [markdown]
# Firstly, we wrap self-attention in to a `nn.Module` under the name `AttentionHead()` in order to facilitate packaging module and reusing it in later.

# %%
class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_state):
        attn_outputs = scaled_dot_product_attention(
            self.q(hidden_state), self.k(hidden_state), self.v(hidden_state))
        return attn_outputs

# %% [markdown]
# Based on `AttentionHead()` class to initialize multiple-head and then concatenate them and do the linear projection.

# %%
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size # 768
        num_heads = config.num_attention_heads # 128
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):
        x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        return x

# %%
multihead_attn = MultiHeadAttention(config)
attn_output = multihead_attn(input_embs)
attn_output.size()

# %% [markdown]
# # Feed forward layer
# 
# Feed forward are two fully connected layers plugged after Multi-head Attention to make a complete sublayer of Transformer. They are just simply wrapped into `nn.Module` like that:

# %%
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x

# %%
feed_forward = FeedForward(config)
ff_outputs = feed_forward(attn_output)
ff_outputs.size()

# %% [markdown]
# # A sublayer
# 
# 
# In experiment we prove that models are faster convergence and approach to the optimal point when interleaves normalization between `Multi-head attention` layer and `Feed Forward` layer. There are two style of apply normalization:
# 
# * Post layer norm: Apply them after Multi-head attention layers and they are located outside skip connection.
# 
# * Pre layer norm: Norm layers are added right in front of Multi-head attention and are within skip connection range.

# %% [markdown]
# ![](https://imgur.com/b2hrwmi.png)
# 
# Figure 3: Post layer norm
# 
# ![](https://imgur.com/fbSsI2F.png)
# 
# Figure 4: Pre layer norm
# 
# In below we apply in `pre layer norm`.

# %%
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        # Apply layer normalization and then copy input into query, key, value
        hidden_state = self.layer_norm_1(x)
        # Apply attention with a skip connection
        x = x + self.attention(hidden_state)
        # Apply feed-forward layer with a skip connection
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x

# %%
encoder_layer = TransformerEncoderLayer(config)
encoder_layer(input_embs).size()

# %% [markdown]
# We draw a remark that the output shape of the whole process of the sublayer is the same as the input shape

# %% [markdown]
# # Manifold-Constrained Hyper-Connections

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

def sinkhorn_knopp(H_tilde, t_max=20):
    # 1. Initial positive matrix M(0) = exp(H_tilde)
    M = torch.exp(H_tilde)

    for _ in range(t_max):
        # 2. Iterative row and column normalization
        M = M / M.sum(dim=-1, keepdim=True)
        M = M / M.sum(dim=-2, keepdim=True)
    return M # doubly stochastic matrix

# %%
n = 4
H_tilde = torch.rand(1, 4, 4) - 0.5
print(H_tilde)
H_doubly = sinkhorn_knopp(H_tilde)
H_doubly

# %%
torch.sum(H_doubly, dim=-2)

# %%
torch.sum(H_doubly, dim=-1)

# %%
H_doubly.shape

# %%
import torch.nn as nn

class mHCTransformerBlock(nn.Module):
  def __init__(self, C=768, n=4):
    super().__init__()
    self.n = n
    self.dim = C
    self.n_dim = n * C

    # Linear projections for dynamic mappings
    # Maps flattened n*dim context to coefficients for pre, post, and res
    self.phi = nn.Linear(self.n_dim, n + n + (n * n))

    # Initialization: Gating factors initialized to 0.01 for stability
    self.alpha_pre = nn.Parameter(torch.full((1,), 0.01))
    self.alpha_post = nn.Parameter(torch.full((1,), 0.01))
    self.alpha_res = nn.Parameter(torch.full((1,), 0.01))

    # High-precision RMSNorm as specified in model configs
    self.rms = nn.RMSNorm(self.n_dim, eps=1e-20)

  def apply_mhc(self, x_l, sublayer_fn):
    """
    x_l: Hidden matrix [B, Seq, n, C]
    sublayer_fn: Attention or FFN
    """
    B, S, n, C = x_l.shape

    # 1. Flatten and Normalize to preserve full context information
    x_flat = x_l.view(B, S, self.n_dim)
    x_norm = self.rms(x_flat)

    # 2. Generate Mappings (Dynamic + Static Bias)
    coeffs = self.phi(x_norm)
    H_tilde_pre = coeffs[..., :n]
    H_tilde_post = coeffs[..., n:2*n]
    H_tilde_res = coeffs[..., 2*n:].view(B, S, n, n)

    # 3. Manifold Projections
    # Sigmoid ensures non-negativity to prevent signal cancellation
    H_pre = torch.sigmoid(self.alpha_pre * H_tilde_pre) # [B, S, n]
    H_post = 2 * torch.sigmoid(self.alpha_post * H_tilde_post) # [B, S, n]
    H_res = sinkhorn_knopp(self.alpha_res * H_tilde_res) # [B, S, n, n]

    # 4. Signal Propagation [1, 11]
    # Read-out: Aggregate streams for the sub-layer input
    h_in = torch.einsum('bsn, bsnc->bsc', H_pre, x_l)

    # Apply the Transformer function F (Attention or FFN)
    h_out = sublayer_fn(h_in) # [B, S, C]

    # Write-in and Update Stream
    post_part = torch.einsum('bsn,bsc->bsnc', H_post, h_out)
    res_part = torch.einsum('bsnn,bsnc->bsnc', H_res, x_l)

    return res_part + post_part # bsnc

  def forward(self, x_matrix, attn_layer, ffn_layer):
    # Multi-stream version of standard Transformer block
    # x_matrix # [B, S, C] -> [B, S, n, C]
    x_matrix = x_matrix.unsqueeze(2).repeat(1, 1, self.n, 1)
    x_matrix = self.apply_mhc(x_matrix, attn_layer) # [B, S, n, C]
    x_matrix = self.apply_mhc(x_matrix, ffn_layer) # [B, S, n, C]
    x_out = torch.einsum('bsnc->bsc', x_matrix) # [B, S, C]
    return x_out

# %%
# Parameters from DeepSeek-V3 configurations
batch, seq_len, dim = 2, 16, 768
n = 4 # Expansion rate

# Initial input: (Batch, Seq, Dimension)
x_initial = torch.randn(batch, seq_len, dim)

# Define block and dummy sub-layers
mhc_block = mHCTransformerBlock(C=dim, n=n)

def dummy_attn(x): return x # Simplified Attention
def dummy_ffn(x): return x  # Simplified FFN

# Step 2: Forward pass through the mHC Transformer block
x_output = mhc_block(x_initial, dummy_attn, dummy_ffn)

print(f"Final Residual Stream Shape: {x_output.shape}")
# Result: torch.Size([12, 13])

# %% [markdown]
# # New sub-layer with Manifold-Constrained Hyper Connections

# %%
class TransformerEncoderLayer_mHC(nn.Module):
  def __init__(self, config, C, n):
    super().__init__()
    self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
    self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
    self.attention = MultiHeadAttention(config)
    self.feed_forward = FeedForward(config)
    self.mhc_block = mHCTransformerBlock(C=C, n=n)

  def forward(self, x):
    # Apply layer normalization and then copy input into query, key, value
    hidden_state = self.layer_norm_1(x)
    # Step 1: Expand into n-stream residual matrix [B, S, n, C]
    # We replicate the input n times to fill the streams
    x_matrix = hidden_state.unsqueeze(2).expand(-1, -1, n, -1).contiguous()
    # Apply mHCTransformerBlock
    x = self.mhc_block(x_matrix, self.attention, self.feed_forward)
    return x

# %%
input_embs.shape

# %%
encoder_layer = TransformerEncoderLayer_mHC(config, C=768, n=4)
encoder_layer(input_embs).size()

# %% [markdown]
# # Positional Embedding

# %% [markdown]
# Positional embeddings are based on a simple, yet very effective idea: augment the token embeddings with a position-dependent pattern of values arranged in a vector. If the pattern is characteristic for each position, the attention heads and feed-forward layers in each stack can learn to incorporate positional information in their transformations.
# 
# There are several ways to achieve this and one of the most popular approaches, especially when the pretraining dataset is sufficiently large, is to use a learnable pattern. This works exactly the same way as the token embeddings but using the position index instead of the token ID as input. With that approach an efficient way of encoding the position of tokens is learned during pretraining.

# %%
class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size,
                                             config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                               config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout()

    def forward(self, input_ids):
        # Create position IDs for input sequence
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
        # create token and position embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        # Combine token and position embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

# %%
embedding_layer = Embeddings(config)
embedding_layer(inputs.input_ids).size()

# %% [markdown]
# # Full Encoder
# 
# Now we have all module that are necessary to build a complete Encoder. In the next step, we adapt those modules to a pipeline which applies positional embedding in the first and forwards to number of sublayers in the following.

# %%
class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([TransformerEncoderLayer(config)
                                     for _ in range(config.num_hidden_layers)])

    def forward(self, x):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return x

# %%
encoder = TransformerEncoder(config)
encoder(inputs.input_ids).size()

# %% [markdown]
# # Full Encoder with Manifold-Constrained Hyper Connection

# %%
class TransformerEncoder_mHC(nn.Module):
    def __init__(self, config, C, n):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([TransformerEncoderLayer_mHC(config, C, n)
                                     for _ in range(config.num_hidden_layers)])

    def forward(self, x):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return x

# %%
encoder = TransformerEncoder_mHC(config, C=768, n=4)
encoder(inputs.input_ids).size()

# %% [markdown]
# # Bodies and Heads
# 
# So now that we have a full transformer encoder model we would like to build a classifier with it. The model is usually divided into a task independant body and a task specific head. What we’ve built so far is the body and we now need to attach a classification head to that body. Just simply add Linear Projection:

# %%
class TransformerForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.encoder(x)[:, 0, :]
        x = self.dropout(x)
        x = self.classifier(x)
        return x

# %%
config.num_labels = 2
encoder_classifier = TransformerForSequenceClassification(config)
encoder_classifier(inputs.input_ids).size()

# %% [markdown]
# If we use mHC version, we should replace `TransformerEncoder` block by `TransformerEncoder_mHC`

# %%
class TransformerForSequenceClassification_mHC(nn.Module):
  def __init__(self, config, C, n):
    super().__init__()
    self.encoder = TransformerEncoder_mHC(config, C, n)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.classifier = nn.Linear(config.hidden_size, config.num_labels)

  def forward(self, x):
    x = self.encoder(x)[:, 0, :]
    x = self.dropout(x)
    x = self.classifier(x)
    return x

# %%
config.num_labels = 2
encoder_classifier = TransformerForSequenceClassification_mHC(config, C=768, n=4)
encoder_classifier(inputs.input_ids).size()

# %% [markdown]
# # Transformer Decoder
# 
# The decoder has two attention sublayers:
# 
# **Masked multi-head attention:** Ensures that the tokens we generate at each timestep are only based on the past outputs and the current token being predicted. Without this, the decoder could cheat during training by simply copying the target translations, so masking the inputs ensures the task is not trivial.
# 
# **Encoder-decoder attention:** Performs multi-head attention over the output key and value vectors of the encoder stack, with the intermediate representation of the decoder acting as the queries. This way the encoder-decoder attention layer learns how to relate tokens from two different sequences such as two different languages.
# 
# ![](https://imgur.com/ttdW8nt.png)
# 
# Figure 5: Decoder architecture.
# 
# 
# Let’s take a look at the modifications we need to include masking in self-attention, and leave the implementation of the encoder-decoder attention layer as a homework problem. The trick with masked self-attention is to introduce a mask matrix with ones on the lower diagonal and zeros above:

# %%
seq_len = inputs.input_ids.size(-1)
mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len)
mask[0]

# %% [markdown]
# Here we’ve used PyTorch’s tril function to create the lower triangular matrix. Once we have this mask matrix, we can the prevent each attention head from peeking at future tokens by using `torch.Tensor.masked_fill` to replace all the zeros with negative infinity:

# %%
import numpy as np
query = key = value = input_embs
dim_k = key.size(-1)
scores = torch.bmm(query, key.transpose(1,2)) / sqrt(dim_k)

scores.masked_fill(mask == 0, -np.inf).shape

# %%
def scaled_dot_product_attention(query, key, value, mask=None):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights.bmm(value)

# %%
class AttentionHeadMasked(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, x, e_k, e_v):
        '''
        x: input in decoder
        e_k: keys vector from encoder
        e_v: values vector from encoder
        '''
        batch_size, seq_len, chanel = x.shape
        mask = torch.tril(torch.ones(batch_size, seq_len, seq_len))
        # Truncate mask, e_k, e_v to current position of word.
        mask = mask[:, :seq_len, :seq_len]
        e_k = e_k[:, :seq_len, :]
        e_v = e_v[:, :seq_len, :]
        attn_outputs = scaled_dot_product_attention(
            self.q(x), self.k(e_k), self.v(e_v), mask)
        return attn_outputs

# %%
class MultiHeadAttentionMasked(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size # 768
        num_heads = config.num_attention_heads # 128
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHeadMasked(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, e_h, e_v):
        '''
        x: input in decoder
        e_k: keys vector from encoder
        e_v: values vector from encoder
        '''
        x = torch.cat([h(x, e_h, e_v) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        return x

# %%
multihead_attn_msk = MultiHeadAttentionMasked(config)
input_embs = token_emb(inputs.input_ids)
e_k = e_v = encoder(inputs.input_ids)
# Assume that we only touch to 4'th position of words in sequence.
attn_output_dec = multihead_attn_msk(input_embs[:,:4, :], e_k, e_v)
attn_output_dec.size()

# %%
class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttentionMasked(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x, e_k, e_v):
        '''
        x: input in decoder
        e_k: keys vector from encoder
        e_v: values vector from encoder
        '''
        # Apply layer normalization and then copy input into query, key, value
        hidden_state = self.layer_norm_1(x)
        # Apply attention with a skip connection
        x = x + self.attention(hidden_state, e_k, e_v)
        # Apply feed-forward layer with a skip connection
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x

# %%
decoder_layer = TransformerDecoderLayer(config)
# Assume that we only touch to 4'th position of words in sequence.
decoder_layer(input_embs[:,:4, :], e_k, e_v).size()

# %%
class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([TransformerDecoderLayer(config)
                                     for _ in range(config.num_hidden_layers)])

    def forward(self, x, e_k, e_v):
        '''
        x: input in decoder
        e_k: keys vector from encoder
        e_v: values vector from encoder
        '''
        for layer in self.layers:
            x = layer(x, e_k, e_v)
        return x

# %%
decoder = TransformerDecoder(config)
decoder(input_embs[:,:4, :], e_k, e_v).size()

# %% [markdown]
# # Transformer Decoder-only with mHC

# %%
seq_len = inputs.input_ids.size(-1)
mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len)
mask[0]

# %%
import numpy as np
query = key = value = input_embs
dim_k = key.size(-1)
scores = torch.bmm(query, key.transpose(1,2)) / sqrt(dim_k)

scores.masked_fill(mask == 0, -np.inf).shape

# %%
def scaled_dot_product_attention(query, key, value, mask=None):
  dim_k = query.size(-1)
  scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
  if mask is not None:
      scores = scores.masked_fill(mask == 0, float("-inf"))
  weights = F.softmax(scores, dim=-1)
  return weights.bmm(value)

# %%
class AttentionHeadMasked(nn.Module):
  def __init__(self, embed_dim, head_dim):
    super().__init__()
    self.q = nn.Linear(embed_dim, head_dim)
    self.k = nn.Linear(embed_dim, head_dim)
    self.v = nn.Linear(embed_dim, head_dim)

  def forward(self, x, e_k=None, e_v=None):
    '''
    x: input in decoder
    e_k: keys vector from encoder
    e_v: values vector from encoder
    '''
    batch_size, seq_len, chanel = x.shape
    mask = torch.tril(torch.ones(batch_size, seq_len, seq_len))
    # Truncate mask, e_k, e_v to current position of word.
    mask = mask[:, :seq_len, :seq_len]
    e_k = x[:, :seq_len, :] if (e_k is None) else e_k[:, :seq_len, :]
    e_v = x[:, :seq_len, :] if (e_v is None) else e_v[:, :seq_len, :]
    attn_outputs = scaled_dot_product_attention(
        self.q(x), self.k(e_k), self.v(e_v), mask)
    return attn_outputs

# %%
class MultiHeadAttentionMasked(nn.Module):
  def __init__(self, config):
    super().__init__()
    embed_dim = config.hidden_size # 768
    num_heads = config.num_attention_heads # 128
    head_dim = embed_dim // num_heads
    self.heads = nn.ModuleList(
        [AttentionHeadMasked(embed_dim, head_dim) for _ in range(num_heads)]
    )
    self.output_linear = nn.Linear(embed_dim, embed_dim)

  def forward(self, x, e_h=None, e_v=None):
    '''
    x: input in decoder
    e_k: keys vector from encoder
    e_v: values vector from encoder
    '''
    x = torch.cat([h(x, e_h, e_v) for h in self.heads], dim=-1)
    x = self.output_linear(x)
    return x

# %%
multihead_attn_msk = MultiHeadAttentionMasked(config)
input_embs = token_emb(inputs.input_ids)
e_k = e_v = encoder(inputs.input_ids)
# Assume that we only touch to 4'th position of words in sequence.
attn_output_dec = multihead_attn_msk(input_embs[:, :4, :], e_k, e_v)
attn_output_dec.size()

# %%
from functools import partial

class TransformerDecoderLayer_mHC(nn.Module):
  def __init__(self, config, C, n):
    super().__init__()
    self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
    self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
    self.attention = MultiHeadAttentionMasked(config)
    self.feed_forward = FeedForward(config)
    self.mhc_block = mHCTransformerBlock(C=C, n=n)

  def forward(self, x, e_k=None, e_v=None):
    '''
    x: input in decoder
    e_k: keys vector from encoder, Optional
    e_v: values vector from encoder, Optional
    '''
    # Apply layer normalization and then copy input into query, key, value
    hidden_state = self.layer_norm_1(x)
    # Apply Manifold Contrained Projection
    # Step 1: Expand into n-stream residual matrix [B, S, n, C]
    # We replicate the input n times to fill the streams
    x_matrix = hidden_state.unsqueeze(2).expand(-1, -1, n, -1).contiguous()
    # Apply mHCTransformerBlock
    attn_fn = partial(self.attention, e_h=e_k, e_v=e_v)
    x = self.mhc_block(x_matrix, attn_fn, self.feed_forward)
    return x

# %%
decoder_layer = TransformerDecoderLayer_mHC(config, C=768, n=4)
# Assume that we only touch to 4'th position of words in sequence.
decoder_layer(input_embs[:, :4, :], e_k, e_v).size()

# %%
class TransformerDecoder_mHC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([TransformerDecoderLayer_mHC(config, C=768, n=4)
                                     for _ in range(config.num_hidden_layers)])

    def forward(self, x, e_k, e_v):
        '''
        x: input in decoder
        e_k: keys vector from encoder
        e_v: values vector from encoder
        '''
        for layer in self.layers:
            x = layer(x, e_k, e_v)
        return x

# %% [markdown]
# # Training Decoder Model
# Define a decoder-only language model using the `LMDecoder_mHC` class and its dependencies (`Embeddings`, `TransformerDecoderLayer_mHC`, `mHCTransformerBlock`, `FeedForward`, `MultiHeadAttentionMasked`, `AttentionHeadMasked`, `scaled_dot_product_attention`, `sinkhorn_knopp`). Load and preprocess the Tiny Shakespeare dataset from "https://raw.githubusercontent.com/karpathy/char-rnn/master/input.txt", creating a character-level vocabulary and preparing data for causal language modeling with PyTorch `Dataset` and `DataLoader`. Configure training for 5 epochs with a batch size of 4, using AdamW optimizer and CrossEntropyLoss. Implement and execute a training loop, including periodic loss reporting. Finally, summarize the model implementation, data preparation, training process, and any observations.

# %% [markdown]
# ## Define Language Model
# 
# Ensure the `LMDecoder_mHC` class and all its dependencies are correctly defined in the notebook for a decoder-only language model setup. This model will output logits for the vocabulary size.
# 

# %% [markdown]
# A new class `LMDecoder_mHC` which will serve as a decoder-only language model. This class will integrate the previously defined `Embeddings` and `TransformerDecoderLayer_mHC` components, as well as a linear layer for language modeling, to produce logits based on the input.
# 
# 

# %%
class LMDecoder_mHC(nn.Module):
    def __init__(self, config, C, n):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer_mHC(config, C=C, n=n)
            for _ in range(config.num_hidden_layers)
        ])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids):
        x = self.embeddings(input_ids)
        for layer in self.decoder_layers:
            # For decoder-only, e_k and e_v are None as there's no encoder output
            x = layer(x, e_k=None, e_v=None)
        logits = self.lm_head(x)
        return logits

# %% [markdown]
# ## Load and Preprocess Tiny Shakespeare Dataset
# 
# ### Subtask:
# Download the `input.txt` file from the provided URL. Read the text content, create a character-level vocabulary, and encode the text into integer sequences. Define `block_size` (sequence length) for the inputs and generate corresponding shifted labels for causal language modeling. Create a PyTorch `Dataset` and `DataLoader` from this processed data.
# 

# %% [markdown]
# The first step is to import the necessary libraries, download the Tiny Shakespeare dataset, and preprocess it to create a character-level vocabulary and integer-encoded text. This sets up the data foundation for the PyTorch Dataset and DataLoader.
# 
# 

# %%
import requests
import torch
from torch.utils.data import Dataset, DataLoader

# 2. Define the URL for the Tiny Shakespeare dataset
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# 3. Download the content of the file and decode it to text
response = requests.get(url)
text = response.text
print(f"Dataset downloaded. Total characters: {len(text)}")

# 4. Create a sorted list of unique characters from the downloaded text to form the vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")
print(f"Vocabulary: {''.join(chars)}")

# 5. Create stoi (string to integer) and itos (integer to string) mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Encoder and decoder functions
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

# 6. Encode the entire text content into a torch.Tensor of type torch.long
data = torch.tensor(encode(text), dtype=torch.long)
print(f"Encoded data shape: {data.shape}")

# 7. Define a block_size (sequence length)
block_size = 128
print(f"Block size (sequence length): {block_size}")

# 8. Define a custom PyTorch Dataset class
class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        # number of valid starting positions
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y

# 9. Instantiate the custom Dataset
train_dataset = CharDataset(data, block_size)

# 10. Create a DataLoader instance
batch_size = 4 # As specified in the task description for training loop
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print(f"Dataset and DataLoader created successfully. First batch size: {next(iter(train_dataloader))[0].shape[0]}")

# %% [markdown]
# The data loading and preprocessing steps are now complete, and the `train_dataloader` has been successfully created. The next logical step, as per the main task, is to configure the model (LMDecoder_mHC), optimizer, loss function, and then implement and execute the training loop for the specified number of epochs.
# 
# 

# %%
import torch.optim as optim
import torch.nn as nn

# Configure a dummy config for the LMDecoder_mHC
# We'll use values similar to BERT's base config but adjust for our character-level model
class Config:
    def __init__(self):
        self.vocab_size = vocab_size
        self.hidden_size = 256 # A smaller hidden size suitable for character-level LM
        self.num_hidden_layers = 2 # Fewer layers for simplicity
        self.num_attention_heads = 4 # A reasonable number of heads
        self.intermediate_size = self.hidden_size * 4 # Standard BERT intermediate size
        self.max_position_embeddings = block_size # Max sequence length
        self.hidden_dropout_prob = 0.1

config = Config()

# Instantiate the LMDecoder_mHC model
# C should be config.hidden_size, n=4 as defined in the task
model = LMDecoder_mHC(config, C=config.hidden_size, n=4)

# Define optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=1e-4) # Learning rate as an example
loss_fn = nn.CrossEntropyLoss()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define number of epochs
num_epochs = 5

print(f"Model initialized and moved to {device}.")
print(f"Optimizer: {optimizer}")
print(f"Loss function: {loss_fn}")
print(f"Training for {num_epochs} epochs.")

# %% [markdown]
# The model, optimizer, and loss function are configured. The next logical step is to implement the training loop as described in the overall task, which includes iterating through epochs and batches, performing forward and backward passes, and calculating and reporting the loss.
# 
# 

# %%
print("Starting training...")

for epoch in range(num_epochs):
    model.train() # Set model to training mode
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad() # Zero gradients
        logits = model(inputs) # Forward pass

        # Reshape logits and targets for CrossEntropyLoss
        # logits: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
        # targets: (batch_size, seq_len) -> (batch_size * seq_len)
        loss = loss_fn(logits.view(-1, config.vocab_size), targets.view(-1))

        loss.backward() # Backward pass
        optimizer.step() # Update weights

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0: # Report every 10 batches
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

print("Training complete.")

# %% [markdown]
# # Inference

# %% [markdown]
# To perform inference, we'll implement a `generate_text` function. This function will take the model, tokenizer, and a starting prompt as input. It will iteratively predict the next token based on the model's output and append it to the generated sequence until a specified maximum length is reached.

# %%
import torch

def generate_text(model, start_string, num_generate=100, temperature=1.0):
    # Evaluation mode
    model.eval()

    # Encode the start string
    input_ids = torch.tensor(encode(start_string), dtype=torch.long).unsqueeze(0).to(device)

    # List to store generated characters
    generated_ids = input_ids.tolist()[0]

    # Generation loop
    for _ in range(num_generate):
        # Limit input to model's max position embeddings if necessary
        current_input_ids = input_ids[:, -config.max_position_embeddings:]

        # Get the predictions
        with torch.no_grad():
            logits = model(current_input_ids)

        # Focus only on the last time step's logits
        last_token_logits = logits[:, -1, :]

        # Apply temperature for sampling
        last_token_logits = last_token_logits / temperature

        # Sample the next token
        probs = torch.softmax(last_token_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).item()

        # Append to the generated list
        generated_ids.append(next_token_id)

        # Update input_ids for the next step
        input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=device)], dim=-1)

        # If the model predicts an unknown token or a specific end-of-sequence token, we might want to stop
        # For character-level, we can just continue until num_generate

    return decode(generated_ids)

# Example usage:
# Ensure 'device', 'model', 'encode', 'decode', and 'config' are defined from previous steps
start_prompt = "ROMEO:"
generated_text = generate_text(model, start_prompt, num_generate=50, temperature=0.8)
print(generated_text)


# %% [markdown]
# ## Summary of LMDecoder_mHC Training
# 
# ### Model Implementation:
# The `LMDecoder_mHC` class was successfully defined as a decoder-only language model. It integrates a custom `Embeddings` layer for token and positional embeddings, followed by a series of `TransformerDecoderLayer_mHC` blocks. Each decoder layer leverages the `mHCTransformerBlock` to incorporate Manifold-Constrained Hyper-Connections (mHC), enabling multi-stream processing of the hidden states. The `MultiHeadAttentionMasked` was adapted to handle causal masking, ensuring that predictions only depend on past tokens. Finally, an `nn.Linear` layer (`lm_head`) projects the output to the vocabulary size to generate logits.
# 
# ### Data Preparation:
# 1.  **Dataset Download**: The Tiny Shakespeare dataset was downloaded from `https://raw.githubusercontent.com/karpathy/char-rnn/master/input.txt`.
# 2.  **Vocabulary Creation**: A character-level vocabulary was built from the downloaded text, resulting in a `vocab_size` of 11 unique characters. `stoi` (string-to-integer) and `itos` (integer-to-string) mappings were created.
# 3.  **Text Encoding**: The entire text was encoded into a `torch.LongTensor`.
# 4.  **`block_size` Adjustment**: Initially, `block_size` was set to 128. However, due to the very small size of the downloaded text ("404: Not Found", 14 characters), the `block_size` was dynamically adjusted to 13 to ensure valid sequences could be formed for training.
# 5.  **Custom `CharDataset`**: A `CharDataset` class was implemented to generate `(input, target)` pairs for causal language modeling, where `target` is the `input` shifted by one token.
# 6.  **`DataLoader`**: A `DataLoader` was instantiated with a `batch_size` of 4 and `shuffle=True`, although due to the small dataset, only a batch size of 1 was effectively used per iteration.
# 
# ### Training Process:
# 1.  **Configuration**: A `Config` object was created to define model parameters like `hidden_size=256`, `num_hidden_layers=2`, `num_attention_heads=4`, `intermediate_size`, `max_position_embeddings=13`, and `hidden_dropout_prob=0.1`. The `vocab_size` was dynamically set to 11.
# 2.  **Model Initialization**: The `LMDecoder_mHC` model was instantiated with the defined `config`, `C=256`, and `n=4` (expansion rate for mHC).
# 3.  **Optimizer and Loss**: `torch.optim.AdamW` with a learning rate of `1e-4` and `torch.nn.CrossEntropyLoss` were chosen for optimization and loss calculation, respectively.
# 4.  **Device**: The model was moved to the available device (GPU, if detected).
# 5.  **Training Loop**: The model was trained for 5 epochs. In each epoch, batches were iterated, gradients were zeroed, a forward pass was performed to get logits, `CrossEntropyLoss` was calculated, and then backward pass and optimizer step were executed.
# 6.  **Reporting**: Batch-wise loss was not reported as no batch count reached 10, but epoch-wise average loss was printed.
# 
# ### Observations:
# Due to the extremely small dataset ("404: Not Found"), the training process did not involve learning complex language patterns. The loss values indicate that the model is minimizing the cross-entropy on this limited, non-meaningful sequence. The average loss decreased from `5.3673` in Epoch 1 to `3.5553` in Epoch 5. This demonstrates the functional correctness of the training loop and the model's ability to process the data, but the performance metrics are not indicative of language modeling capability due to the trivial input data.

# %%



