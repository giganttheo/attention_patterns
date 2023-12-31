import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import math
from functools import reduce
import jax

class AttentionPattern():
  def __init__(self):
    self.receivers = {}
    self.senders = {}
    self.embeddings = None
    self.size = (0, 0)
    self.n_heads = 1
    self.graph_mask = {}
    self.dtype = jnp.float32
    self.batch_size = 0

  def _get_from_dict(self, dataDict, mapList):
    """Iterate nested dictionary"""
    return reduce(dict.get, mapList, dataDict)

  def _cleaning_duplicates(self, receivers_heads, senders_heads):
    def clean_adj_list_duplicates(r, s):
      edges = set()
      clean_r = []
      clean_s = []
      for i, j in zip(r, s):
        if (i, j) not in edges:
          edges.add((i, j))
          clean_r.append(i)
          clean_s.append(j)
      return clean_r, clean_s
    clean_receivers_heads = []
    clean_senders_heads = []
    for r, s in zip(receivers_heads, senders_heads):
      clean_r, clean_s = clean_adj_list_duplicates(r,s)
      clean_receivers_heads.append(jnp.array(clean_r))
      clean_senders_heads.append(jnp.array(clean_s))
    return clean_receivers_heads, clean_senders_heads

  def _padding_graphs(self, receivers_heads, senders_heads, attention_mask=None):
    max_graph_len = max([receivers.shape[0] for receivers in receivers_heads])
    r, s, m = [], [], []
    def pad_to(mat, padding):
      padded_mat = jnp.zeros((padding), dtype=jnp.uint16)
      padded_mat = padded_mat.at[:mat.shape[0]].set(mat)
      return padded_mat
    def get_mask(mat, padding):
      graph_mask = jnp.zeros((padding), dtype="i4")
      graph_mask = graph_mask.at[:mat.shape[0]].set(jnp.ones_like(mat, dtype="i4"))
      return graph_mask
    h = []
    m_h = []
    for receivers in receivers_heads:
      h.append(pad_to(receivers, max_graph_len))
      m_h.append(get_mask(receivers, max_graph_len))
    r = h
    h = []
    for senders in senders_heads:
      h.append(pad_to(senders, max_graph_len))
    m = m_h
    s = h
    return jnp.array(r, dtype=jnp.uint16), jnp.array(s, dtype=jnp.uint16), jnp.array(m, dtype="i4")

  def mask(self, mask):
    self.receivers = jax.tree_util.tree_map(lambda r, mask: r*mask, self.receivers, mask)
    self.senders = jax.tree_util.tree_map(lambda s, mask: s*mask, self.senders, mask)

  def get_causal_mask(self):
    f = lambda r,s: jnp.array(list(map(lambda i,j : i >= j, r, s)))
    return jax.tree_util.tree_map(lambda r,s: f (r,s), self.receivers, self.senders)

  def get_adj_mat(self):
    # receivers = self._get_from_dict(self.receivers, layer_path)
    # senders = self._get_from_dict(self.senders, layer_path)
    # graph_mask = self._get_from_dict(self.graph_mask, layer_path)   
    adj_mat = jnp.zeros((self.batch_size,) + self.size)
    for batch in range(self.batch_size):
      for head in range(self.n_heads):
        for i, (r, s) in enumerate(zip(self.receivers[batch, head], self.senders[batch, head])):
          if self.graph_mask[batch, head, i]:
            adj_mat = adj_mat.at[batch, r, s].set(adj_mat[batch, r, s] + (1 / self.n_heads))
    return adj_mat

  def get_rec_field(self, log=False):
    #receptive field is the normalized n_layers hops matrix
    #ie A^n_layers with A the adjacency matrix
    # receivers_values, _ = jax.tree_util.tree_flatten(self.receivers)
    # senders_values, _ = jax.tree_util.tree_flatten(self.senders)
    fn = lambda path, _: self.get_adj_mat([p.key for p in path])
    adj_mats = jax.tree_util.tree_map_with_path(fn, self.receivers)
    rec_field = jax.tree_util.tree_reduce(lambda value, element: value @ element,
                                          adj_mats,
                                          initializer=jnp.array([jnp.eye(self.size[0])]*self.batch_size)
                                          )
    if log:
      eps = jnp.finfo(self.dtype).eps
      rec_field = jnp.log(rec_field + eps)
    rec_field *= 100 / (jnp.max(rec_field))
    return rec_field

  def show_receptive_field(self, log=False):
    rec_field = self.get_rec_field(log=log)[0]

    fig, ax = plt.subplots()
    # Using matshow here just because it sets the ticks up nicely. imshow is faster.
    ax.matshow(rec_field, vmin=0, cmap=plt.cm.winter)#'seismic')

    for (i, j), z in np.ndenumerate(rec_field):
        ax.text(j, i, math.floor(z), ha='center', va='center')

  def show_attention_pattern(self):
    # if layer_path is None:
    #   layer_path = [path.key for (_, path) in jax.tree_util.tree_flatten_with_path(self.receivers)[0][0]]
    adj_mat = self.get_adj_mat()
    plt.imshow(adj_mat,vmin=0, vmax=1, cmap=plt.cm.winter)
    ax = plt.gca()

    ax.xaxis.tick_top()
    # Major ticks
    ax.set_xticks(np.arange(0, self.size[0], 1))
    ax.set_yticks(np.arange(0, self.size[1], 1))

    # Labels for major ticks
    ax.set_xticklabels(np.arange(0, self.size[0], 1))
    ax.set_yticklabels(np.arange(0, self.size[1], 1))

    # Minor ticks
    ax.set_xticks(np.arange(-.5, self.size[0], 1), minor=True)
    ax.set_yticks(np.arange(-.5, self.size[1], 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

    # Remove minor ticks
    ax.tick_params(which='minor', bottom=False, left=False)

  def get_attention_graph(self):
    return {"receivers": self.receivers, "senders": self.senders, "graph_mask": self.graph_mask}

class VanillaAttentionPattern(AttentionPattern):
  def __init__(self, seq_len_q, seq_len_kv, n_heads=4, batch_size = 2, dtype=jnp.float32):
    super().__init__()
    self.dtype = dtype
    self.batch_size = batch_size
    receivers = []
    senders = []
    seq_kv = range(seq_len_kv)
    seq_q = range(seq_len_q)
    for head in range(n_heads):
      layer_receivers = []
      layer_senders = []
      for i in seq_kv:
        for j in seq_q:
          layer_receivers.append(i)
          layer_senders.append(j)
      receivers.append(layer_receivers)
      senders.append(layer_senders)
    receivers, senders = self._cleaning_duplicates(receivers, senders)
    receivers, senders, graph_mask = self._padding_graphs(receivers, senders)
    receivers = jnp.array([receivers]*batch_size)
    senders = jnp.array([senders]*batch_size)
    graph_mask = jnp.array([graph_mask]*batch_size)
    self.receivers = receivers
    self.senders = senders
    self.graph_mask = graph_mask
    self.n_heads = n_heads
    self.size = (seq_len_kv, seq_len_q)

n_heads = 12

def graph_from_path(tree, enc_self_attn, dec_self_attn, encdec_attn, path=[]):
  if not isinstance(tree, dict):
    return None
  if 'SelfAttention' in path:
    #self attention
    if 'encoder' in path:
      return enc_self_attn
    else: #decoder attn
      return dec_self_attn
  elif 'EncDecAttention' in path:
    #encoder / decoder cross attention
    return encdec_attn
  return {k: graph_from_path(t, enc_self_attn=enc_self_attn, dec_self_attn=dec_self_attn, encdec_attn=encdec_attn, path=path+[k]) for (k, t) in tree.items()}

def create_dense_attn_patterns(model, max_source_length, max_target_length, n_heads, batch_size, dtype=jnp.float32, attn_type=VanillaAttentionPattern):

    enc_self_attn = attn_type(seq_len_q=max_source_length, seq_len_kv=max_source_length, n_heads=n_heads, batch_size=batch_size, dtype=dtype).get_attention_graph()
    dec_self_attn = attn_type(seq_len_q=1, seq_len_kv=max_target_length, n_heads=n_heads, batch_size=batch_size, dtype=dtype).get_attention_graph()
    #this is cross attn
    #kv is the receivers and in cross attention the encoder
    #q is the senders and in cross attention the decoder
    encdec_attn = attn_type(seq_len_q=1, seq_len_kv=max_source_length, n_heads=n_heads, batch_size=batch_size, dtype=dtype).get_attention_graph()

    graph = graph_from_path(model.params, enc_self_attn, dec_self_attn, encdec_attn)
    return graph

def create_dense_attn_patterns_bis(model, max_source_length, max_target_length, n_heads, batch_size, dtype=jnp.float32, attn_type=VanillaAttentionPattern):

    enc_self_attn = attn_type(seq_len_q=max_source_length, seq_len_kv=max_source_length, n_heads=n_heads, batch_size=batch_size, dtype=dtype).get_attention_graph()
    dec_self_attn = attn_type(seq_len_q=max_target_length, seq_len_kv=max_target_length, n_heads=n_heads, batch_size=batch_size, dtype=dtype).get_attention_graph()
    #this is cross attn
    #kv is the receivers and in cross attention the encoder
    #q is the senders and in cross attention the decoder
    encdec_attn = attn_type(seq_len_q=max_target_length, seq_len_kv=max_source_length, n_heads=n_heads, batch_size=batch_size, dtype=dtype).get_attention_graph()

    graph = graph_from_path(model.params, enc_self_attn, dec_self_attn, encdec_attn)
    return graph