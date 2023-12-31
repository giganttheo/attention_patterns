import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import math
from functools import reduce
import jax

class AttentionPattern():
  #this implementation is batch-friendly
  def __init__(self):
    self.receivers = {}
    self.senders = {}
    self.embeddings = None
    self.size = (0, 0)
    self.n_heads = 1
    self.graph_mask = {}
    self.dtype = jnp.float16
    self.batch_size = 0

  def _get_from_dict(self, dataDict, mapList):
    """Iterate nested dictionary"""
    return reduce(dict.get, mapList, dataDict)

  def _cleaning_duplicates(self, receivers_heads, senders_heads, causal=False):
    def clean_adj_list_duplicates(r, s):
      edges = set()
      clean_r = []
      clean_s = []
      for i, j in zip(r, s):
        if (i, j) not in edges:
          if not causal or (i >= j): #with the causal mask, we ignore edges where the receiver is before the sender
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

  def _padding_graphs(self, receivers_heads, senders_heads):
    max_graph_len = max([receivers.shape[0] for receivers in receivers_heads])
    r, s, m = [], [], []
    def pad_to(mat, padding):
      padded_mat = jnp.zeros((padding), dtype=jnp.int8)
      padded_mat = padded_mat.at[:mat.shape[0]].set(mat)
      return padded_mat
    def get_mask(mat, padding):
      graph_mask = jnp.zeros((padding), dtype=jnp.int8)
      graph_mask = graph_mask.at[:mat.shape[0]].set(jnp.ones_like(mat))
      return graph_mask
    h = []
    m_h = []
    for receivers in receivers_heads:
      h.append(pad_to(receivers, max_graph_len))
      m_h.append(get_mask(receivers, max_graph_len))
    r = h
    m = m_h
    h = []
    for senders in senders_heads:
      h.append(pad_to(senders, max_graph_len))
    s = h
    return jnp.array(r), jnp.array(s), jnp.array(m)

  def mask(self, mask):
    self.receivers = jax.tree_util.tree_map(lambda r, mask: r*mask, self.receivers, mask)
    self.senders = jax.tree_util.tree_map(lambda s, mask: s*mask, self.senders, mask)

  def get_causal_mask(self):
    f = lambda r,s: jnp.array(list(map(lambda i,j : i >= j, r, s)))
    return jax.tree_util.tree_map(lambda r,s: f (r,s), self.receivers, self.senders)

  def get_adj_mat(self, layer_path):
    receivers = self._get_from_dict(self.receivers, layer_path)
    senders = self._get_from_dict(self.senders, layer_path)
    graph_mask = self._get_from_dict(self.graph_mask, layer_path)
    adj_mat = jnp.zeros((self.batch_size,) + self.size)
    for batch in range(self.batch_size):
      for head in range(self.n_heads):
        for i, (r, s) in enumerate(zip(receivers[batch, head], senders[batch, head])):
          if graph_mask[batch, head, i]:
            adj_mat = adj_mat.at[batch, r, s].set(adj_mat[batch, r, s] + (1 / self.n_heads))
    return adj_mat

  def get_rec_field(self, log=False):
    #receptive field is the normalized n_layers hops matrix
    #ie A^n_layers with A the adjacency matrix
    receivers_values, _ = jax.tree_util.tree_flatten(self.receivers)
    senders_values, _ = jax.tree_util.tree_flatten(self.senders)
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

  def show_attention_pattern(self, layer_path=None):
    if layer_path is None:
      layer_path = [path.key for path in jax.tree_util.tree_leaves_with_path(self.receivers)[0][0]]
    adj_mat = self.get_adj_mat(layer_path)[0]
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
    return {"receivers": self.receivers, "senders": self.senders, "graph_mask": self.graph_mask, "embeddings": self.embeddings}
  

class VanillaAttentionPattern(AttentionPattern):
  def __init__(self, seq_len, n_heads=4, n_layers=3, batch_size = 2):
    super().__init__()
    self.batch_size = batch_size
    self.receivers = {}
    self.senders = {}
    self.graph_mask = {}
    for layer in range(n_layers):
      receivers = []
      senders = []
      for head in range(n_heads):
        layer_receivers = []
        layer_senders = []
        for i in range(seq_len):
          for j in range(seq_len):
            layer_receivers.append(i)
            layer_senders.append(j)
        receivers.append(layer_receivers)
        senders.append(layer_senders)
      receivers, senders = self._cleaning_duplicates(receivers, senders)
      receivers, senders, graph_mask = self._padding_graphs(receivers, senders)
      receivers = jnp.array([receivers]*batch_size)
      senders = jnp.array([senders]*batch_size)
      graph_mask = jnp.array([graph_mask]*batch_size)
      self.receivers[f"layer_{layer}"] = receivers
      self.senders[f"layer_{layer}"] = senders
      self.graph_mask[f"layer_{layer}"] = graph_mask
    self.n_heads = n_heads
    self.size = (seq_len, seq_len)