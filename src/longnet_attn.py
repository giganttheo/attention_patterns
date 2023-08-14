from base_attn_pattern import AttentionPattern

class LongnetAttentionPattern(AttentionPattern):
  #
  def __init__(self, seq_len, block_size, dilations=[1, 2, 4], n_heads=4, n_layers=3, staggered=True):
    super().__init__()

    #dilated block attn
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
            for dilation in dilations:
              offset = staggered * (head % dilation)
              if (i) // block_size == (j) // block_size :
                if i * dilation + offset < seq_len and j * dilation + offset < seq_len:
                  layer_senders.append(dilation * i + offset)
                  layer_receivers.append(dilation * j + offset)
                  if i != j:
                    layer_senders.append(dilation * j + offset)
                    layer_receivers.append(dilation * i + offset)
        receivers.append(layer_receivers)
        senders.append(layer_senders)
      receivers, senders = self._cleaning_duplicates(receivers, senders)
      receivers, senders, graph_mask = self._padding_graphs(receivers, senders)
      self.receivers[f"layer_{layer}"] = receivers
      self.senders[f"layer_{layer}"] = senders
      self.graph_mask[f"layer_{layer}"] = graph_mask
    self.n_heads = n_heads
    self.size = (seq_len, seq_len)