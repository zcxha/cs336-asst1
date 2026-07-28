from model_config import *
step = 40000

# GPU's compute capability (TFLOP/s)
compute_capability = 11
# algorithm's MFU (percentage of utilization of GPU)
MFU = 0.5

# checkpointing
save_interval = 100
val_interval = 100
val_batchnum = 50

# adamw optimizer
weight_decay = 0.1
betas = (0.9, 0.95)
eps = 1e-8

# learning rate scheduling
max_lr = 1e-3
min_lr = 0.1 * max_lr
warmup_iters = 0.02 * step
# gradient clipping
max_l2norm = 1