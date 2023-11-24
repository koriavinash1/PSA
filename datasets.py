import copy
import os
import random
from typing import Callable, Optional, Tuple, Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Dataset

import torch.nn.functional as F
from torch import distributions as dist
from torchvision import transforms
from tqdm import tqdm
from hps import Hparams
import matplotlib

import h5py
import json, random
from pathlib import Path
from tqdm import tqdm



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
            # object_property = np.eye(len(size_mapping))[size_mapping[object_info['size']]]
            # object_property = np.concatenate([object_property, 
            #                 np.eye(len(shape_mapping))[shape_mapping[object_info['shape']]]])
            # object_property = np.concatenate([object_property, 
            #                 np.eye(len(color_mapping))[color_mapping[object_info['color']]]])
            # object_property = np.concatenate([object_property, 
            #                 np.eye(len(material_mapping))[material_mapping[object_info['material']]]])
            # object_property = np.concatenate([object_property, [1]])
            # object_property = np.concatenate([object_property, (np.array(object_info['3d_coords']) + 3.0)/6.0])
            # objects.append(object_property)


            object_property = (np.array(object_info['3d_coords']) + 3.0)/6.0
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

        # initialize the cache
        self.cache = {}


    def __getitem__(self, index):
        
        
        if index in self.cache.keys():
            sample = self.cache[index] 
            return sample

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

        self.cache[index] = sample
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
        cache: bool = True,
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


def _normalize_numerical_feature(
    data: np.array, metadata: Dict, feature_name: str
) -> Tensor:
    mean = metadata[feature_name]["mean"].astype("float32")
    std = np.sqrt(metadata[feature_name]["var"]).astype("float32")
    return torch.as_tensor((data - mean) / (std + 1e-6), dtype=torch.float32)


def _onehot_categorical_feature(data: np.array, num_classes: int) -> Tensor:
    tensor = torch.as_tensor(data, dtype=torch.int64).squeeze(-1)
    return F.one_hot(tensor, num_classes=num_classes).to(torch.float32)



class HDF5Loader(Dataset):
    def __init__(self, root, 
                        start_idx: int = 0,
                        dataset_size: int = 90000,
                        max_objects: int = 10,
                        properties: bool = False,
                        properties_list: Optional[List] = [],
                        resolution: Tuple[int, int] = (128, 128),
                        transform: Optional[Callable] = None):
        super(HDF5Loader, self).__init__()
        
        self.root_dir = root  
        self.max_objects = max_objects   
        self.resolution = resolution
        self.properties = properties
        self.properties_list = properties_list
        self.transform = transform

        self.start_idx = start_idx
        self.dataset_size = dataset_size

        self.data, self.metadata = self._load_data_hdf5(self.root_dir)
        print ('all data loaded')
        self._filter_to_max_objects()

        # print (len(self.data['image']))

    def _filter_to_max_objects(self):
        # print ('filtering images based on num_objects')
        num_objects = np.array(self.data['num_actual_objects'])
        self.condition = np.where(num_objects < self.max_objects)[0]
        self.start_idx = int(self.start_idx*len(self.condition)/len(num_objects))
        self.dataset_size = int(self.dataset_size*len(self.condition)/len(num_objects))
        self.condition = self.condition[self.start_idx: self.start_idx + self.dataset_size]

        # for feature_name in self.data.keys():
        #     self.data[feature_name] = self.data[feature_name][condition]
         
        # for feature_name in self.data.keys():
        #     self.data[feature_name] = self.data[feature_name][self.start_idx: self.start_idx + self.dataset_size]


    def _preprocess_feature(self, feature: np.ndarray, feature_name: str) -> Any:
        """Preprocesses a dataset feature at the beginning of `__getitem__()`.

        Args:
            feature: Feature data.
            feature_name: Feature name.

        Returns:
            The preprocessed feature data.
        """
        if feature_name == "image":
            if self.transform:
                return self.transform(Image.fromarray(feature).convert('RGB'))

        if feature_name == "mask":
            feature = np.array(Image.fromarray(feature[:, :, 0]).convert("L").resize(
                                        self.resolution, 
                                        Image.NEAREST))[:, :, None]

            feature[feature > self.max_objects] = 0
            one_hot_masks = F.one_hot(
                torch.as_tensor(feature, dtype=torch.int64),
                num_classes=self.max_objects + 1,
            )

            # (num_objects, 1, height, width)
            return one_hot_masks.permute(3, 2, 0, 1).to(torch.float32)
        
        if feature_name == "visibility":
            feature = torch.as_tensor(feature, dtype=torch.float32)
            if feature.dim() == 1:  # e.g. in ObjectsRoom
                feature.unsqueeze_(1)
            return feature

        if feature_name == "num_actual_objects":
            return torch.as_tensor(feature, dtype=torch.float32)


        if feature_name in self.metadata.keys():
            # Type is numerical, categorical, or dataset_property.
            feature_type = self.metadata[feature_name]["type"]
            if feature_type == "numerical":
                return _normalize_numerical_feature(
                    feature, self.metadata, feature_name
                )
            if feature_type == "categorical":
                return _onehot_categorical_feature(
                    feature, self.metadata[feature_name]["num_categories"]
                )
        return feature


    def __getitem__(self, index) -> Dict:
        index = self.condition[index]
        out = {}
        for feature_name in self.data.keys():
            out[feature_name] = self._preprocess_feature(
                self.data[feature_name][index], feature_name
            )

        if self.properties:
            properties = []
            for feature_name in out.keys():
                properties.append(out[feature_name])
            
            properties = torch.cat(properties, dim = 1)
            out['properties'] = properties
        
        # renaming keys to match existing codebase
        out['x'] = out['image']
        out['y'] = out['mask']

        return out
        


    def _load_data(self) -> Tuple[Dict, Dict]:
        """Loads data and metadata.

        By default, the data is a dict with h5py.Dataset values, but when overriding
        this method we allow arrays too."""
        return self._load_data_hdf5(data_path=self.full_dataset_path)



    def _load_data_hdf5(
        self, 
        data_path: str, 
        metadata_suffix: str = "metadata.npy"
    ) -> Tuple[Dict[str, h5py.Dataset], Dict]:
        """Loads data and metadata assuming the data is hdf5, and converts it to dict."""
        data_path = Path(data_path)
        metadata_fname = f"{data_path.stem.split('-')[0]}-{metadata_suffix}"
        metadata_path = data_path.parent / metadata_fname
        metadata = np.load(str(metadata_path), allow_pickle=True).item()

        if not isinstance(metadata, dict):
            raise RuntimeError(f"Metadata type {type(metadata)}, expected instance of dict")
        
        dataset = h5py.File(data_path, "r")
        # From `h5py.File` to a dict of `h5py.Datasets`.
        dataset = {k: dataset[k] for k in dataset}
        
        return dataset, metadata


    def __len__(self):
        return len(self.condition)


class Clevr(HDF5Loader):
    def _load_data(self) -> Tuple[Dict, Dict]:
        data, metadata = super()._load_data()

        # 'pixel_coords' shape: (B, num objects, 3)
        data["x_2d"] = data["pixel_coords"][:, :, 0]
        data["y_2d"] = data["pixel_coords"][:, :, 1]
        data["z_2d"] = data["pixel_coords"][:, :, 2]
        del data["pixel_coords"]
        del metadata["pixel_coords"]
        return data, metadata


class ClevrTex(Clevr):
    pass


class Multidsprites(HDF5Loader):
    def _load_data(self) -> Tuple[Dict, Dict]:
        data, metadata = super()._load_data()
        hsv = matplotlib.colors.rgb_to_hsv(data["color"])
        data["hue"] = hsv[:, :, 0]
        data["saturation"] = hsv[:, :, 1]
        data["value"] = hsv[:, :, 2]
        return data, metadata


class Shapestacks(HDF5Loader):
    def _load_data(self) -> Tuple[Dict, Dict]:
        data, metadata = super()._load_data()
        data = rename_dict_keys(data, mapping={"rgba": "color"})
        metadata = rename_dict_keys(metadata, mapping={"rgba": "color"})
        data["x"] = data["com"][:, :, 0]
        data["y"] = data["com"][:, :, 1]
        data["z"] = data["com"][:, :, 2]
        return data, metadata


class Tetrominoes(HDF5Loader):
    pass

class ObjectsRoom(HDF5Loader):
    pass



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



def hdf5_loader(args: Hparams):
    # Load data
    if args.hps == 'clevr':
        datagenerator = Clevr
        index = {'train': {'start': 0, 'size': 90000},
                    'val': {'start': 90000, 'size': 5000},
                    'test': {'start': 95000, 'size': 5000}}

    elif args.hps == 'clevr_tex':
        datagenerator = ClevrTex
        index = {'train': {'start': 0, 'size': 40000},
                    'val': {'start': 40000, 'size': 5000},
                    'test': {'start': 45000, 'size': 5000}}

    elif args.hps == 'multidsprites':
        datagenerator = Multidsprites
        index = {'train': {'start': 0, 'size': 90000},
                    'val': {'start': 90000, 'size': 5000},
                    'test': {'start': 95000, 'size': 5000}}

    elif args.hps == 'objects_room':
        datagenerator = ObjectsRoom
        index = {'train': {'start': 0, 'size': 90000},
                    'val': {'start': 90000, 'size': 5000},
                    'test': {'start': 95000, 'size': 5000}}

    elif args.hps == 'shapestacks':
        datagenerator = Shapestacks
        index = {'train': {'start': 0, 'size': 90000},
                    'val': {'start': 90000, 'size': 5000},
                    'test': {'start': 95000, 'size': 5000}}

    elif args.hps == 'tetrominoes':
        datagenerator = Tetrominoes
        index = {'train': {'start': 0, 'size': 90000},
                    'val': {'start': 90000, 'size': 5000},
                    'test': {'start': 95000, 'size': 5000}}

    else:
        raise ValueError('Unknown dataset found')


    aug = {
        "train": transforms.Compose(
            [
                transforms.Resize((args.input_res, args.input_res), antialias=None),
                # transforms.CenterCrop(args.input_res),
                transforms.PILToTensor(),  # (0,255)
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((args.input_res, args.input_res), antialias=None),
                # transforms.CenterCrop(args.input_res),
                transforms.PILToTensor(),  # (0,255)
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize((args.input_res, args.input_res), antialias=None),
                # transforms.CenterCrop(args.input_res),
                transforms.PILToTensor(),  # (0,255)
            ]
        )
    }

    
    datasets = {
        split: datagenerator(root = args.data_dir, 
                            start_idx = index[split]['start'],
                            dataset_size = index[split]['size'],
                            max_objects = args.max_num_obj,
                            properties = args.nconditions > 0,
                            properties_list = args.properties_list,
                            resolution = (args.input_res, args.input_res),
                            transform = aug[split])
        for split in ["train", "val", "test"]
    }

    return datasets


def custom_loader(args: Hparams) -> Dict[str, DataGenerator]:
    # Load data
    n = args.input_res * 0.004296875  # = 0.55 for 128
    h, w = int(320 * n), int(480 * n)
    aug = {
        "train": transforms.Compose(
            [
                transforms.Resize((args.input_res, args.input_res), antialias=None),
                # transforms.CenterCrop(args.input_res),
                transforms.PILToTensor(),  # (0,255)
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((args.input_res, args.input_res), antialias=None),
                # transforms.CenterCrop(args.input_res),
                transforms.PILToTensor(),  # (0,255)
            ]
        ),
    }

    datasets = {
        split: DataGenerator(root=args.data_dir, 
                                mode=split, 
                                resolution=(args.input_res, args.input_res),
                                max_objects=args.max_num_obj,
                                transform=aug[split],
                                properties=args.nconditions > 0,
                                class_info=False)
        for split in ["train", "val"]
    }
    # datasets['test'] = datasets.CLEVRClassification(
    #     root='./', split='test', transform=None
    # )
    datasets["test"] = copy.deepcopy(datasets["val"])
    return datasets