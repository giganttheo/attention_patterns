from base_attn_pattern import AttentionPattern
from functools import partial

class StaggeredBlockSelfAttentionPattern(AttentionPattern):
  def __init__(self, seq_len, block_size, n_heads = 1, n_layers = 4, staggered=True):
    super().__init__()
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
          for j in range(i, seq_len):
            offset = staggered * (layer % 2) * (block_size // 2)
            if (i + offset) // block_size == (j + offset) // block_size :
              layer_senders.append(i)
              layer_receivers.append(j)
              if i != j:
                layer_senders.append(j)
                layer_receivers.append(i)
        receivers.append(layer_receivers)
        senders.append(layer_senders)
      receivers, senders = self._cleaning_duplicates(receivers, senders)
      receivers, senders, graph_mask = self._padding_graphs(receivers, senders)
      self.receivers[f"layer_{layer}"] = receivers
      self.senders[f"layer_{layer}"] = senders
      self.graph_mask[f"layer_{layer}"] = graph_mask
    self.n_heads = n_heads
    self.size = (seq_len, seq_len)

BlockSelfAttentionPattern = partial(StaggeredBlockSelfAttentionPattern, staggered=False)
