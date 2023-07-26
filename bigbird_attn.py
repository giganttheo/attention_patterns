from base_attn_pattern import AttentionPattern
import jax
import jax.numpy as jnp

class BigBirdBlockSparseAttentionPattern(AttentionPattern):
  #BigBird block-sparse attention
  def __init__(self, seq_len, block_size, num_rand_tokens, n_layers=3, n_heads=4, mode="itc", rng_key=jax.random.PRNGKey(0)):
    # ITC: internal transformer construction
    #     global tokens: 2 x block_size
    #     window tokens: 3 x block_size
    #     random tokens: num_rand_tokens x block_size

    # ETC: extended transformer construction
    #     global tokens: extra_globals_tokens + 2 x block_size
    #     window tokens: 3 x block_size
    #     random tokens: num_rand_tokens x block_size

    if mode=="itc":
      n_global_tokens = 2 * block_size
      window_size = 3
      n_random = num_rand_tokens

    else:
      extra_global_tokens=0 #
      n_global_tokens = extra_global_tokens + 2 * block_size
      window_size = 3 * block_size
      n_random = num_rand_tokens

    super().__init__()

    #global attn
    global_tokens = range(n_global_tokens)

    self.receivers = {}
    self.senders = {}
    self.graph_mask = {}
    for layer in range(n_layers):
      receivers = []
      senders = []
      for head in range(n_heads):
        perm_key, rng_key = jax.random.split(rng_key)

        #local window attn

        #random attn
        random_attn = jnp.array([[i,j] for i in range(seq_len//block_size) for j in range(seq_len//block_size) if i > n_global_tokens // block_size and j > n_global_tokens // block_size and i != j and abs(i - j) > window_size // 2])
        random_attn = jax.random.permutation(key=perm_key, x=random_attn, independent=False)[:n_random]

        layer_receivers = [int(edge[0]) * block_size + i for i in range(block_size) for j in range(block_size) for edge in random_attn]
        layer_senders = [int(edge[1]) * block_size + j for i in range(block_size) for j in range(block_size) for edge in random_attn]

        for i in range(seq_len):
          for j in range(seq_len):
            if i in global_tokens or j in global_tokens or abs( (i // block_size ) - (j // block_size )) <= window_size // 2:
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