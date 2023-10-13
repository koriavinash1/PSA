import copy
import os
import random
from typing import Callable, Optional, Tuple, Dict

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
from hps import Hparams



def get_paths_with_properties_CLEVR(root_path, mode, max_objects=7):
    size_mapping = {'large': 0, 'small': 1}
    shape_mapping ={'sphere': 0, 'cube': 1, 'cylinder':2}
    color_mapping = {'red':0, 'green': 1, 'blue': 2, 'yellow': 3, 'gray': 4, 'cyan': 5, 'brown': 6, 'purple': 7}
    material_mapping = {'rubber': 0, 'metal': 1}
    data = json.load(open(os.path.join(root_path, mode, f'CLEVR_HANS_scenes_{mode}.json'), 'r'))['scenes']
    paths = []; properties = []
    for data_info in data:
        paths.append(os.path.join(root_path, mode, 'images', data_info['image_filename']))
        
        objects = []
        for object_info in data_info['objects']:
            object_property = np.eye(len(size_mapping))[size_mapping[object_info['size']]]
            object_property = np.concatenate([object_property, 
                            np.eye(len(shape_mapping))[shape_mapping[object_info['shape']]]])
            object_property = np.concatenate([object_property, 
                            np.eye(len(color_mapping))[color_mapping[object_info['color']]]])
            object_property = np.concatenate([object_property, 
                            np.eye(len(material_mapping))[material_mapping[object_info['material']]]])
            object_property = np.concatenate([object_property, [1]])
            object_property = np.concatenate([object_property, (np.array(object_info['3d_coords']) + 3.0)/6.0])
            objects.append(object_property)

        for _ in range(max_objects - len(objects)):
            objects.append(np.zeros_like(object_property))

        properties.append(np.array(objects, dtype='float32')[:max_objects, ...])
    
    properties = np.array(properties)
    return paths, properties




# 3D shapes dataset
# Shape stack
# bitmoji
# flying mnist
EXTS = ['jpg', 'jpeg', 'png']
class DataGenerator(Dataset):
    def __init__(self, root, 
                        mode: str ='train',
                        max_objects: int = 10,
                        properties: bool = False,
                        class_info: bool =False,
                        masks: bool = False,
                        reasoning_type: str = 'default',
                        resolution: Tuple[int, int] = (128, 128),
                        transform: Optional[Callable] = None):

        super(DataGenerator, self).__init__()
        
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.masks = masks
        self.root_dir = root     
        self.resolution = resolution
        self.properties = properties
        self.class_info = class_info
        self.transform = transform
        self.reasoning_type = reasoning_type

        if self.properties and root.lower().__contains__('hans'):
            self.files, self.properties_info = get_paths_with_properties_CLEVR(root, mode, max_objects)
        else:
            path = os.path.join(self.root_dir, self.mode)
            self.files = [str(p) for ext in EXTS for p in Path(f'{path}').glob(f'**/*.{ext}')]
        

        if masks:
            self.mask_dir = os.path.join(os.path.dirname(root), 'masks', mode)
            self.mask_names = os.listdir(self.mask_dir)



    def __getitem__(self, index):
        path = self.files[index]
        sample = {}

        if self.masks:
            return_mask = np.zeros(self.resolution)
            name = path.split('/')[-1].split('.')[0]
            if not (name in  self.mask_names):
                name_ = np.random.choice(self.mask_names)
                path = path.replace(name, name_)
            else:
                name_ = name

            mask_dir = os.path.join(self.mask_dir, name_)
            mask_dirs = np.array(os.listdir(mask_dir))
            idxs = np.array([int(n.split('.')[0]) for n in mask_dirs])
            mask_dirs = mask_dirs[np.argsort(idxs)]

            nobjects = 0
            for i, ns in enumerate(mask_dirs):
                mask = Image.open(os.path.join(mask_dir, ns)).convert("L")
                mask = np.array(mask.resize(self.resolution, Image.NEAREST))
                nobjects += 1 if np.sum(mask) > 10 else 0 
                return_mask += int(ns.split('.')[0])*(mask > 0)
            sample['mask'] = return_mask
            sample['nmasks'] = nobjects
            # print (np.unique(sample['mask']))

        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        sample['x'] = image


        # ====================================
        if self.properties:
            if path.lower().__contains__('hans'):
                property_info = self.properties_info[index]
            elif path.lower().__contains__('mnist'):
                property_info = []
                for d in path.split('/')[-1].split('_'):
                    if d == 'MNIST': break
                    property_info.append(np.eye(11)[int(d)])

                # background property
                property_info.append(np.eye(11)[-1])
                property_info = np.array(property_info)

            property_info = torch.from_numpy(property_info)
            sample['properties'] = property_info





        # ====================================
        if self.class_info:
            if path.lower().__contains__('hans'):
                target = int(path.split('/')[-1].split('_')[3])
            elif path.lower().__contains__('mnist'):
                digits = []
                for d in path.split('/')[-1].split('_'):
                    if d == 'MNIST': break
                    digits.append(int(d))

                digits = np.sort(digits)[::-1]
                if self.reasoning_type == 'diff':
                    target = digits[0]
                    for digit in digits[1:]:
                        target -= digit

                    target += (len(digits) - 2)*9
                elif self.reasoning_type == 'mixed':
                    target = 0
                    for digit in digits:
                        if digit > 5: target -= digit
                        else: target += digit
                    
                    target += len(digits)*9
                else:
                    target = np.sum(digits)

            sample['y'] = target   

        return sample
            
    
    def __len__(self):
        return len(self.files)



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

        return {'x': image, 'y':label}




def clevr(args: Hparams) -> Dict[str, CLEVRN]:
    # Load data
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
                download=True
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
    return datasets




def clevr_hans(args: Hparams) -> Dict[str, CLEVRN]:
    # Load data
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
        split: DataGenerator(root=args.data_dir, 
                                mode=split, 
                                resolution=(h,w),
                                max_objects=args.max_num_obj,
                                transform=aug[split],
                                properties=args.nconditions > 0,
                                class_info=True)
        for split in ["train", "val"]
    }
    # datasets['test'] = datasets.CLEVRClassification(
    #     root='./', split='test', transform=None
    # )
    datasets["test"] = copy.deepcopy(datasets["val"])
    return datasets