import logging
import os
from typing import Any, Dict, Tuple

import send2trash
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import clevr, custom_loader, hdf5_loader
from hps import Hparams
from utils import linear_warmup, seed_worker


def setup_dataloaders(args: Hparams) -> Dict[str, DataLoader]:

    if args.data_dir.__contains__('hdf5'):
        datasets = hdf5_loader(args)
    elif args.hps == "clevr":
        datasets = clevr(args)
    elif args.hps in ["clevr_hans3", "clevr_hans7", "bitmoji", "objects_room", "ffhq"]:
        datasets = custom_loader(args)
    else:
        NotImplementedError


    kwargs = {
        "batch_size": args.bs,
        # "num_workers": os.cpu_count() // 2,
        "pin_memory": True,
        "worker_init_fn": seed_worker,
    }
    dataloaders = {
        "train": DataLoader(datasets["train"], shuffle=True, drop_last=True, **kwargs),
        "valid": DataLoader(datasets["val"], shuffle=False, **kwargs),
        "test": DataLoader(datasets["test"], shuffle=False, **kwargs),
    }
    return dataloaders


def setup_optimizer(
    args: Hparams, model: nn.Module
) -> Tuple[torch.optim.Optimizer, Any]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd, betas=args.betas
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=linear_warmup(args.lr_warmup_steps)
    )

    return optimizer, scheduler


def setup_directories(args: Hparams) -> str:
    save_dir = os.path.join(args.ckpt_dir, args.exp_name, f'Run-{args.run_idx}')
    if os.path.isdir(save_dir):
        if (
            input(f"\nSave directory '{save_dir}' already exists, overwrite? [y/N]: ")
            == "y"
        ):
            if input(f"Send '{save_dir}', to Trash? [y/N]: ") == "y":
                send2trash.send2trash(save_dir)
                print("Done.\n")
            else:
                exit()
        else:
            if (
                input(f"\nResume training with save directory '{save_dir}'? [y/N]: ")
                == "y"
            ):
                pass
            else:
                exit()
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def setup_tensorboard(args: Hparams, model: nn.Module) -> SummaryWriter:
    """Setup metric summary writer."""
    writer = SummaryWriter(args.save_dir)

    hparams = {}
    for k, v in vars(args).items():
        if isinstance(v, list) or isinstance(v, torch.device):
            hparams[k] = str(v)
        elif isinstance(v, torch.Tensor):
            hparams[k] = v.item()
        else:
            hparams[k] = v

    writer.add_hparams(hparams, {"hparams": 0}, run_name=os.path.abspath(args.save_dir))

    if "vae" in type(model).__name__.lower():
        z_str = []
        if hasattr(model.decoder, "blocks"):
            for i, block in enumerate(model.decoder.blocks):
                if block.stochastic:
                    z_str.append(f"z{i}_{block.res}x{block.res}")
        else:
            z_str = ["z0_" + str(args.z_dim)]

        writer.add_custom_scalars(
            {
                "nelbo": {"nelbo": ["Multiline", ["nelbo/train", "nelbo/valid"]]},
                "nll": {"kl": ["Multiline", ["nll/train", "nll/valid"]]},
                "kl": {"kl": ["Multiline", ["kl/train", "kl/valid"]]}
                # "KL": {
                #     "KL_train": ["Multiline", ['KL_train/'+z[:2] for z in z_str]],
                #     "KL_valid": ["Multiline", ['KL_valid/'+z[:2] for z in z_str]]
                # }
            }
        )
    return writer


def setup_logging(args: Hparams) -> logging.Logger:
    # reset root logger
    [logging.root.removeHandler(h) for h in logging.root.handlers[:]]
    # info logger for saving command line outputs during training
    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(args.save_dir, "trainlog.txt")),
            logging.StreamHandler(),
        ],
        # filemode='a',  # append to file, 'w' for overwrite
        format="%(asctime)s, %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(args.exp_name)  # name the logger
    return logger