import argparse
import numpy as np
import cs336_basics.data_utils as data_utils
from cs336_basics.prenorm_transformer_block import TransformerLM
from cs336_basics.nn_utils import AdamW, cross_entropy, get_lr_cosine_schedule, gradient_clipping


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training Script")

    parser.add_argument("dataset", help="tokenized dataset which contains token ids. It's numpy uint16")
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--theta", type=float, default=10000)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--betas", nargs=2, type=float, default=(0.9, 0.95))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--steps", type=int, default=100, help="max training iterations")
    parser.add_argument("--device", type=str, default="cuda")

        
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_iters", type=int, default=10)
    parser.add_argument("--max_l2norm", type=float, default=1000)

    args = parser.parse_args()

    model = TransformerLM(args.vocab_size, args.context_length, args.num_layers, args.d_model, args.num_heads, args.d_ff, args.theta, device=args.device)
    adamw = AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay, betas=args.betas, eps=args.eps, device=args.device)

    dataset = np.memmap(args.dataset, dtype=np.uint16, mode="r")

    for it in range(args.steps):
        # 0 sample inputs & targets
        sample = data_utils.get_batch(dataset, args.batch_size, args.context_length, device=args.device)
        # 1 model forward
        logits = model(sample[0])
        # 2 calculate loss
        loss = cross_entropy(logits, sample[1])
        # 3 backward
        loss.backward()
        # 4 gradient clipping
        gradient_clipping(model.parameters(), args.max_l2norm)
        # 5 lr_schedule
        lr = get_lr_cosine_schedule(it, args.max_lr, args.min_lr, args.warmup_iters, args.steps)
        for group in adamw.param_groups:
            group["alpha"] = lr
        # 6 optimizer update
        adamw.step()
        # 7 clean grad
        adamw.zero_grad()
        print(loss)