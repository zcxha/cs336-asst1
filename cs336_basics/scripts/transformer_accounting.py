batch_size = 1024
vocab_size = 50257
context_length = 1024
sequence_length = context_length
num_layers = 48
d_model = 1600
num_heads = 25
d_ff = 4288

# accounting flops

DIVIDER = 1000 * 1000 * 1000 # GFLOPs

per_block_projection = 2 * batch_size * sequence_length * d_model * d_model * 4

per_block_attention = 2 * batch_size * sequence_length * sequence_length * d_model * 2

per_block_SwiGLU = 2 * batch_size * sequence_length * d_model * d_ff * 3

Transformer_sum = per_block_projection + per_block_attention + per_block_SwiGLU

Transformer_all = Transformer_sum * num_layers

lm_head = 2 * batch_size * sequence_length * d_model * vocab_size

total = Transformer_all + lm_head

transformer_blocks_prop = Transformer_all / total

proj_prop = (per_block_projection / Transformer_sum) * transformer_blocks_prop
attn_prop = (per_block_attention / Transformer_sum) * transformer_blocks_prop
swig_prop = (per_block_SwiGLU / Transformer_sum) * transformer_blocks_prop


lm_head_prop = lm_head / total

print(f"{num_layers} of Transformer block {Transformer_sum / DIVIDER} GFLOPs takes total: {Transformer_sum / DIVIDER * num_layers} GFLOPs")
print(f"lm_head takes {lm_head / DIVIDER} GFLOPs")
print(f"""
Transformer: {transformer_blocks_prop * 100}%
    - projection: {proj_prop * 100}%
    - attention: {attn_prop * 100}%
    - SwiGLU: {swig_prop * 100}%
LMHead: {lm_head_prop * 100}%
""")

print(f"total {total / DIVIDER} GFLOPs {total / DIVIDER / 1000} TFLOPs")