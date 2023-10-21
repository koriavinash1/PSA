"""Utility script to download datasets."""
# code adapted from https://github.com/addtt/object-centric-library/blob/main/download_data.py

import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import urllib.request
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from torch import Tensor
from tqdm import tqdm



REMOTE_ROOT = "https://data-download.compute.dtu.dk/multi_object_datasets/"


def download_file(url: str, destination: str, chunk_size: int = 1024):
    """Downloads files from URL."""
    with open(destination, "wb") as f:
        request = urllib.request.Request(url, headers={"User-Agent": "OCL"})
        with urllib.request.urlopen(request) as response:
            destination_name = Path(destination).name
            with tqdm(total=response.length, desc=destination_name) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break
                    pbar.update(chunk_size)
                    f.write(chunk)

def _dataset_files(name: str, *, include_style: bool) -> List[str]:
    extended_name = {
        "clevr": "clevr_10",
        "multidsprites": "multidsprites_colored_on_grayscale",
        "objects_room": "objects_room_train",
    }.get(name, name)
    datasets_without_style = ["clevrtex"]
    out = [f"{extended_name}-{suffix}" for suffix in ["full.hdf5", "metadata.npy"]]
    if include_style and name not in datasets_without_style:
        out.append(f"{name}-style.hdf5")
    return out


def _get_remote_address(name: str) -> str:
    assert REMOTE_ROOT.endswith("/")
    return REMOTE_ROOT + name


def _get_destination(name: str) -> str:
    return str(DATA_ROOT / name)


if __name__ == "__main__":

    available_datasets = [
                        'clevr',
                        'clevrtex',
                        'multidsprites',
                        'objects_room',
                        'shapestacks',
                        'tetrominoes'
    ]

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        required=True,
        help="Names of the datasets to be downloaded (space-separated list). If the "
        "string 'all' is given, all available datasets will be downloaded.",
    )
    parser.add_argument(
        "--include-style-transfer",
        action="store_true",
        help="Whether style transfer versions of the datasets should be downloaded too",
    )
    parser.add_argument(
        "--data_root",
        default='../Datasets',
        help="download dataset to",
    )
    args = parser.parse_args()


    DATA_ROOT = Path(args.data_root)

    # Validate datasets and skip invalid ones.
    datasets = args.datasets
    if datasets == ["all"]:
        datasets = available_datasets
    else:
        missing = [d for d in datasets if d not in available_datasets]
        if missing:
            print(f"The following datasets are not valid: {missing}")
            datasets = [d for d in datasets if d in available_datasets]
    print(f"The following datasets will be downloaded: {datasets}")

    # Create data folder.
    if DATA_ROOT.exists():
        if not DATA_ROOT.is_dir():
            raise FileExistsError(f"Data path '{DATA_ROOT}' exists and is not a folder")
    else:
        while True:
            choice = input(f"Data folder '{DATA_ROOT}' will be created. Continue? (y/n) ")
            if choice.lower() == "n":
                print("Aborting")
                sys.exit(0)
            elif choice.lower() == "y":
                break
        DATA_ROOT.mkdir(parents=True)

    for dataset in datasets:
        print(f"\nDownloading files for '{dataset}'...")
        filenames = _dataset_files(dataset, include_style=args.include_style_transfer)
        for filename in filenames:
            destination = _get_destination(filename)
            if Path(destination).exists():
                print(f"Destination file {destination} exists: skipping.")
                continue
            download_file(_get_remote_address(filename), destination)