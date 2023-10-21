import torch
import torch.nn as nn
import os
import numpy as np
import torchvision
from shutil import rmtree
from tqdm import tqdm
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from typing import Callable, Dict, List, Optional, Tuple

from torch.autograd import grad as torch_grad
from torch import Tensor, nn
from pytorch_fid import fid_score
from sklearn.metrics import accuracy_score
from mcc import slot_mean_corr_coef


def ari(
        true_mask: Tensor, 
        pred_mask: Tensor, 
        num_ignored_objects: int
) -> torch.FloatTensor:
    """Computes the ARI score.

    Args:
        true_mask: tensor of shape [batch_size x *] where values go from 0 to the number of objects.
        pred_mask:  tensor of shape [batch_size x *] where values go from 0 to the number of objects.
        num_ignored_objects: number of objects (in ground-truth mask) to be ignored when computing ARI.

    Returns:
        a vector of ARI scores, of shape [batch_size, ].
    """
    true_mask = true_mask.flatten(1)
    pred_mask = pred_mask.flatten(1)
    not_bg = true_mask >= num_ignored_objects
    result = []
    batch_size = len(true_mask)
    for i in range(batch_size):
        ari_value = adjusted_rand_score(
            true_mask[i][not_bg[i]], pred_mask[i][not_bg[i]]
        )
        result.append(ari_value)
    result = torch.FloatTensor(result)  # shape (batch_size, )
    return result.mean()



def mse(
        true_image: Tensor, 
        reconstruction: Tensor, 
        object_mask: Optional[Bool] = None,
        only_fg: Optional[Bool] = False
) -> Tensor:

    if only_fg:
        not_bg = object_mask > 0
        true_image = true_image[not_bg]
        reconstruction = reconstruction[not_bg]
    
    return ((true_image - reconstruction)**2).mean([1, 2, 3]).mean()



def compositional_contrast(
        latents: Tensor,
        outputs: Tensor
) -> Tensor:
    b, K, d = latents.shape

    cc = 0
    for k in range(K):
        for j in range(k, K):
            Jk = torch_grad(outputs=outputs, 
                            inputs=latents[:,k,:],
                            grad_outputs=torch.ones(output.size(), 
                                                device=latents.device),
                            create_graph=True, 
                            retain_graph=True, 
                            only_inputs=True)[0].reshape(b, -1)

            Jj = torch_grad(outputs=outputs, 
                            inputs=latents[:,k,:],
                            grad_outputs=torch.ones(output.size(), 
                                                device=latents.device),
                            create_graph=True, 
                            retain_graph=True, 
                            only_inputs=True)[0].reshape(b, -1)
            
            cc += (Jk.norm(2, dim=1)*Jj.norm(2, dim=1)).mean()

    return cc


def slot_mcc(
            run1_slots: Tensor,
            run2_slots: Tensor
    ):
    return slot_mean_corr_coef(run1_slots, run2_slots)