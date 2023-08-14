from base_attn_pattern import AttentionPattern
from functools import partial

class DilatedWindowSelfAttentionPattern(AttentionPattern):
  def __init__(self, seq_len, window_size, dilation=None, n_heads=1, n_layers=1, causal=False):
    super().__init__()
    if not dilation and not dilation is None:
      #no dilation
      dilation = [1]*n_heads
    if dilation is None:
      dilation = range(1, 1 + n_heads)
    receivers = []
    senders = []
    for head in range(n_heads):
      layer_receivers = []
      layer_senders = []
      for i in range(0, seq_len):
        for j in [i + offset * dilation[head] for offset in range(- (window_size // 2), (window_size % 2) + window_size // 2) if seq_len >= i + offset * dilation[head] >= 0]:
          layer_senders.append(i)
          layer_receivers.append(j)
      receivers.append(layer_receivers)
      senders.append(layer_senders)
    receivers, senders = self._cleaning_duplicates(receivers, senders, causal=causal)
    receivers, senders, graph_mask = self._padding_graphs(receivers, senders)
    self.receivers = {f"layer_{layer}": receivers for layer in range(n_layers)}
    self.senders = {f"layer_{layer}": senders for layer in range(n_layers)}
    self.graph_mask = {f"layer_{layer}": graph_mask for layer in range(n_layers)}
    self.n_heads = n_heads
    self.size = (seq_len, seq_len)

WindowSelfAttentionPattern = partial(DilatedWindowSelfAttentionPattern, dilation=False)