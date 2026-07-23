batch_size = 1024
vocab_size = 50257
context_length = 1024
sequence_length = context_length
num_layers = 48
d_model = 1600
num_heads = 25
d_ff = 4288
step = 400 * 1000

parameters = d_model * (vocab_size + num_layers * (2 + 3 * d_ff + 4 * d_model) + 1)
activate_transformer_rms2 = activate_transformer_rms1 = batch_size * sequence_length * d_model * 2
activate_transformer_mha = batch_size * sequence_length * (d_model * 5 + sequence_length)
activate_ffn = batch_size * sequence_length * d_model + batch_size * sequence_length * d_ff + batch_size * sequence_length * d_ff
norm_linear = batch_size * sequence_length * d_model * 3
cross_entropy = batch_size * sequence_length * vocab_size * 2
gradients = parameters
optimizer_state = 2 * parameters
DIVIDER = 1024 * 1024 * 1024
# fp32 to store
total_memory = 4  * (parameters + gradients + optimizer_state + (activate_ffn + activate_transformer_mha + activate_transformer_rms1 + activate_transformer_rms2) * num_layers + norm_linear + cross_entropy)

# calculate total FLOPS (forward + backward + optimizer)
DIVIDER_TFLOP = 1000 * 1000 * 1000 * 1000
forward_flops = 3601 # TFLOPs
backward_flops = 2 * forward_flops
optimizer_one_step_flops = d_model * (vocab_size + num_layers * (2 + 3 * d_ff + 4 * d_model) + 1) * 13 / DIVIDER_TFLOP

total_flops = (forward_flops + backward_flops + optimizer_one_step_flops) * step

print(f"""batch_size {batch_size} step {step}
Memory:
parameters: {4*parameters / DIVIDER} GB
activate: {4*((activate_transformer_rms2 + activate_transformer_rms1 + activate_transformer_mha + activate_ffn) * num_layers) / DIVIDER} GB
misc: {4*(norm_linear + cross_entropy) / DIVIDER} GB
gradients: {4*gradients / DIVIDER} GB
optimizer_state: {4*optimizer_state / DIVIDER} GB
 
the memory of activations+parameters+gradients+optimizer state {total_memory / DIVIDER}GB

FLOPs:
forward: {forward_flops} TFLOPs
backward: {backward_flops} TFLOPs
optimizer_one_step: {optimizer_one_step_flops} TFLOPs
total_flops of training with step {step} : {total_flops} TFLOPs
      """)