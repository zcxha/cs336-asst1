import argparse
import numpy as np
import wandb
import torch
import os
import typing
import cs336_basics.data_utils as data_utils
import cs336_basics.scripts.training_config as config
from cs336_basics.prenorm_transformer_block import TransformerLM
from cs336_basics.nn_utils import AdamW, cross_entropy, get_lr_cosine_schedule, gradient_clipping
from pathlib import Path
from tqdm import tqdm

def save_model(model: torch.nn.Module, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    obj = {"model": model.state_dict()}
    torch.save(obj, out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training Script")

    parser.add_argument("dataset", help="tokenized dataset which contains token ids. It's numpy uint16")
    parser.add_argument("dataset_val", help="validation dataset")
    parser.add_argument("--load_checkpoint", type=Path, default="modelbackup.ckpt")
    parser.add_argument("--save_checkpoint", type=Path, default="modelbackup.ckpt")
    parser.add_argument("--best_model_savepath", type=Path, default="best_model.pth")
    parser.add_argument("--best_loss_savepath", type=Path, default="best_loss.txt")
    parser.add_argument("--save_interval", type=int, default=100, help="every [save_interval] steps of training, save model")
    parser.add_argument("--val_interval", type=int, default=100, help="every [val_interval] steps of training, run a validation")
    parser.add_argument("--val_batchnum", type=int, default=50, help="when validation, what number of batches needed to be validated")
    parser.add_argument("--vocab_size", type=int, default=config.vocab_size)
    parser.add_argument("--context_length", type=int, default=config.context_length)
    parser.add_argument("--num_layers", type=int, default=config.num_layers)
    parser.add_argument("--d_model", type=int, default=config.d_model)
    parser.add_argument("--num_heads", type=int, default=config.num_heads)
    parser.add_argument("--d_ff", type=int, default=config.d_ff)
    parser.add_argument("--theta", type=float, default=config.theta)

    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--betas", nargs=2, type=float, default=(0.9, 0.95))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--steps", type=int, default=config.step, help="max training iterations")
    parser.add_argument("--device", type=str, default="cuda")

        
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_iters", type=int, default=10)
    parser.add_argument("--max_l2norm", type=float, default=1000)

    args = parser.parse_args()

    # 将 argparse 的 Namespace 对象转换为普通字典
    config_dict = vars(args)
    # 初始化 wandb
    wandb.init(
    project="cs336-asst1",
    name=f"training_lr{args.max_lr}_bs{args.batch_size}", # 动态生成 run 名称
    config=config_dict
    )
    
    model = TransformerLM(args.vocab_size, args.context_length, args.num_layers, args.d_model, args.num_heads, args.d_ff, args.theta, device=args.device)
    adamw = AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay, betas=args.betas, eps=args.eps, device=args.device)

    dataset = np.memmap(args.dataset, dtype=np.uint16, mode="r")
    dataset_val = np.memmap(args.dataset_val, dtype=np.uint16, mode="r")

    last_iter = 0
    if args.load_checkpoint.is_file():
        last_iter = data_utils.load_checkpoint(args.load_checkpoint, model, adamw)
    best_loss = float("inf")
    if args.best_loss_savepath.is_file():
        with open(args.best_loss_savepath, "r") as f:
            best_loss = float(f.read())
    for it in tqdm(range(last_iter + 1, args.steps + 1), desc="training epochs"):
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
        # 8 logging
        wandb.log({"train_loss": loss.item()})
        # 9 checkpointing
        if it % args.save_interval == 0:
            data_utils.save_checkpoint(model, adamw, it, args.save_checkpoint)
        # 10 validate loss
        if it % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_loss: float = 0.0
                for i in range(args.val_batchnum):
                    sample = data_utils.get_batch(dataset_val, args.batch_size, args.context_length, device=args.device)
                    logits = model(sample[0])
                    val_loss += cross_entropy(logits, sample[1]).item()
                val_loss /= args.val_batchnum
                wandb.log({"val_loss": val_loss})
                if val_loss < best_loss:
                    best_loss = val_loss
                    save_model(model, args.best_model_savepath)
                    with open(args.best_loss_savepath, "w") as f:
                        f.write(str(best_loss))
            model.train()