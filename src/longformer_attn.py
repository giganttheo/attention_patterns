from base_attn_pattern import AttentionPattern

class LongformerAttentionPattern(AttentionPattern):
  #Longformer local global attention
  def __init__(self, seq_len, block_size, attention_windows=[3, 5, 7], n_heads=4, n_layers=3, sentence_tokens=[0]):
    super().__init__()
    
    # per layer attention window
    window_size = [attention_window * 2 + 1 for attention_window in attention_windows] #effective window size

    #global attn
    global_tokens = sentence_tokens

    #local window attn
    for layer in range(n_layers):
      receivers = []
      senders = []
      for head in range(n_heads):
        layer_receivers = []
        layer_senders = []
        for i in range(seq_len):
          for j in range(seq_len):
            if i in global_tokens or j in global_tokens or abs( (i // block_size ) - (j // block_size )) <= window_size[layer] // 2:
              layer_receivers.append(i)
              layer_senders.append(j)
        receivers.append(layer_receivers)
        senders.append(layer_senders)
      receivers, senders = self._cleaning_duplicates(receivers, senders)
      receivers, senders, graph_mask = self._padding_graphs(receivers, senders)
      self.receivers[f"layer_{layer}"] = receivers
      self.senders[f"layer_{layer}"] = senders
      self.graph_mask[f"layer_{layer}"] = graph_mask
    self.n_heads = n_heads
    self.size = (seq_len, seq_len)