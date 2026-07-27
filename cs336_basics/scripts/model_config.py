batch_size = 1
vocab_size = 10000
context_length = 1024
sequence_length = context_length
num_layers = 12
d_model = 768
num_heads = 12
d_ff = 2048 # d_ff = d_model * 8 / 3 , and multiple of 64

theta = 10000

# tokenizer
eos_token = "<|endoftext|>"