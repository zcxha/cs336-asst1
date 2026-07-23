# data loader
import torch
import numpy.typing as npt
import numpy as np
import os
import typing
np.random.seed(42)
def get_batch(
        dataset: npt.NDArray,
        batch_size: int,
        context_length: int,
        device: str
        ) -> tuple[torch.Tensor, torch.Tensor]:
    indices = np.random.choice(
        a=len(dataset) - context_length,
        size=batch_size,
        replace=False
    )
    offsets = np.arange(context_length)
    index = indices[:, None] + offsets[None, :]
    sample = dataset[index]
    index = index + 1
    target = dataset[index]
    return (torch.from_numpy(sample).to(device, dtype=torch.long), torch.from_numpy(target).to(device, dtype=torch.long))

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    r"""
    should dump all the state from the model, optimizer and iteration into the file-like object out. You can use the state_dict method of both the model and the optimizer to get their relevant states and use torch.save(obj, out) to dump obj into out (PyTorch supports either a path or a file-like object here). A typical choice is to have obj be a dictionary, but you can use whatever format you want as long as you can load your checkpoint later.
    
    Args
        model: torch.nn.Module
        optimizer: torch.optim.Optimizer
        iteration: int
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    """
    obj = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    torch.save(obj, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer) -> int:
    r"""
    should load a checkpoint from src (path or file-like object), and then recover the model and optimizer states from that checkpoint. Your function should return the iteration number that was saved to the checkpoint. You can use torch.load(src) to recover what you saved in your save_checkpoint implementation, and the load_state_dict method in both the model and optimizer to return them to their previous states.

    Args
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
        model: torch.nn.Module
        optimizer: torch.optim.Optimizer
    """
    obj = torch.load(src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]