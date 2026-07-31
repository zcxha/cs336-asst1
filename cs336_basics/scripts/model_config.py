batch_size = 32
vocab_size = 10000
context_length = 256
sequence_length = context_length
num_layers = 4
d_model = 512
num_heads = 16
d_ff = 2048 # d_ff = d_model * 8 / 3 , and multiple of 64

theta = 10000

# tokenizer
special_tokens = ["<|endoftext|>"]
eos_token = "<|endoftext|>"