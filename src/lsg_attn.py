from base_attn_pattern import AttentionPattern

class LSGAttentionPattern(AttentionPattern):
  def __init__(self, seq_len, block_size, sparsify_factor=2, global_tokens=[0], block_strided=False, n_heads=4, n_layers=3):
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
          for j in range(seq_len):
            if block_strided:
              sparse_context=(((i) // block_size == (j) // block_size + 2 ) and\
                             ((i // sparsify_factor)%sparsify_factor==head%sparsify_factor)) or\
                             (((i) // block_size == (j) // block_size - 2 ) and\
                             ((i // sparsify_factor)%sparsify_factor==head%sparsify_factor))
            else:
              sparse_context= (((i) // block_size == (j) // block_size + 2 ) and\
                              (i%sparsify_factor==head%sparsify_factor)) or\
                              (((i) // block_size == (j) // block_size - 2 ) and\
                              (i%sparsify_factor==head%sparsify_factor))
            local_context = (i) // block_size == (j) // block_size or\
                            (i) // block_size == (j) // block_size - 1 or\
                            (i) // block_size == (j) // block_size + 1
            global_context = (i in global_tokens) or (j in global_tokens)
            # local_context=False
            if local_context or\
            sparse_context or\
            global_context:
              layer_senders.append(i)
              layer_receivers.append(j)
        receivers.append(layer_receivers)
        senders.append(layer_senders)
      receivers, senders = self._cleaning_duplicates(receivers, senders)
      receivers, senders, graph_mask = self._padding_graphs(receivers, senders)
      self.receivers[f"layer_{layer}"] = receivers
      self.senders[f"layer_{layer}"] = senders
      self.graph_mask[f"layer_{layer}"] = graph_mask
    self.n_heads = n_heads
    self.size = (seq_len, seq_len)