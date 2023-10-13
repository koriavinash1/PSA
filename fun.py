import copy
import os
import random
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Dataset

from torch import distributions as dist
from torchvision import transforms
from tqdm import tqdm


def init_params(p):
    if isinstance(p, (nn.Linear, nn.Conv2d, nn.Parameter)):
        nn.init.xavier_uniform_(p.weight)
        p.bias.data.fill_(0)


def seed_all(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class CLEVRN(Dataset):
    def __init__(
        self,
        clevr: Dataset,
        num_obj: int = 6,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        cache: bool = False,
    ):
        super().__init__()
        self.cache = cache
        assert num_obj >= 3 and num_obj <= 10
        assert clevr._split != "test"  # test set labels are None
        self.filter_idx = [i for i, y in enumerate(clevr._labels) if y <= num_obj]
        self._image_files = [clevr._image_files[i] for i in self.filter_idx]
        self._labels = [clevr._labels[i] for i in self.filter_idx]
        self.transform = transform
        self.target_transform = target_transform

        if self.cache:
            from concurrent.futures import ThreadPoolExecutor

            self._images = []
            with ThreadPoolExecutor() as executor:
                self._images = list(
                    tqdm(
                        executor.map(self._load_image, self._image_files),
                        total=len(self._image_files),
                        desc=f"Caching CLEVR {clevr._split}",
                        mininterval=(0.1 if os.environ.get("IS_NOHUP") is None else 90),
                    )
                )

    def _load_image(self, file):
        return Image.open(file).convert("RGB")

    def __len__(self):
        return len(self._image_files)

    def __getitem__(self, idx: int):
        if self.cache:
            image = self._images[idx]
        else:
            image = self._load_image(self._image_files[idx])
        label = self._labels[idx]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        print (image.dtype)
        return image, label


def run_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    optimizer: Optional[Optimizer] = None
):
    training = False if optimizer is None else True
    model.train(training)
    loader = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        mininterval=(0.1 if os.environ.get("IS_NOHUP") is None else 60),
    )

    stats = {k: 0 for k in ["elbo", "nll", "kl", "n"]}
    for _, (x, y) in loader:
        x = (x.cuda() - 127.5) / 127.5  # (-1,1)
        model.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            out = model(x)
            
        if training:
            out['elbo'].backward()
            optimizer.step()

        bs = x.shape[0]
        stats["n"] += bs  # samples seen counter
        stats["elbo"] += out["elbo"].detach() * bs
        stats["nll"] += out["nll"].detach() * bs
        stats["kl"] += out["kl"].detach() * bs
        loader.set_description(
           f' => nelbo: {stats["elbo"] / stats["n"]:.3f}'
                + f' - nll: {stats["nll"] / stats["n"]:.3f}'
                + f' - kl: {stats["kl"] / stats["n"]:.3f}'
            ,
            refresh=False,
        )
    return stats['elbo'] / stats["n"]


if __name__ == "__main__":
    import argparse
    from hps import setup_hparams, add_arguments
    from hvae import HVAE

    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = setup_hparams(parser)
    

    seed_all(args.seed, args.deterministic)
    os.makedirs(f"./checkpoints/{args.exp_name}", exist_ok=True)

    n = args.input_res * 0.004296875  # = 0.55 for 128
    h, w = int(320 * n), int(480 * n)
    aug = {
        "train": transforms.Compose(
            [
                transforms.Resize((h, w), antialias=None),
                # transforms.CenterCrop(args.input_res),
                transforms.RandomCrop(args.input_res),
                transforms.PILToTensor(),  # (0,255)
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((h, w), antialias=None),
                transforms.CenterCrop(args.input_res),
                transforms.PILToTensor(),  # (0,255)
            ]
        ),
    }

    datasets = {
        split: CLEVRN(
            torchvision.datasets.CLEVRClassification(
                root=args.data_dir,
                split=split,
            ),
            num_obj=args.max_num_obj,
            transform=aug[split],
        )
        for split in ["train", "val"]
    }
    # datasets['test'] = datasets.CLEVRClassification(
    #     root='./', split='test', transform=None
    # )
    datasets["test"] = copy.deepcopy(datasets["val"])

    kwargs = {
        "batch_size": args.batch_size,
        "num_workers": os.cpu_count(),  # 4 cores to spare
        "pin_memory": True,
    }

    dataloaders = {
        split: DataLoader(
            datasets[split],
            shuffle=(split == "train"),
            drop_last=(split == "train"),
            **kwargs,
        )
        for split in ["train", "val", "test"]
    }

    model = HVAE(args).cuda()
    model.apply(init_params)
    print(f"{model}\n#params: {sum(p.numel() for p in model.parameters()):,}")
    
    for k in sorted(vars(args)):
        print(f"--{k}={vars(args)[k]}")

    optimizer = AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )


    best_loss = 1e6
    print("\nRunning sanity check...")
    _ = run_epoch(model, dataloaders["val"])

    for epoch in range(1, args.epochs):
        print("\nEpoch {}:".format(epoch))
        train_loss = run_epoch(model, dataloaders["train"], optimizer)

        if epoch % 4 == 0:
            valid_loss = run_epoch(model, dataloaders["val"])

            if valid_loss < best_loss:
                best_loss = valid_loss
                step = int(epoch * len(dataloaders["train"]))
                save_dict = {
                    "hparams": vars(args),
                    "epoch": epoch,
                    "step": step,
                    "best_loss": best_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(save_dict, f"./checkpoints/{args.exp_name}/{step}_ckpt.pt")