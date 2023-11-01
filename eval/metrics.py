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
from torch.func import jacfwd
from torchmetrics import R2Score


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
        object_mask: Optional[Tensor] = None,
        only_fg: Optional[bool] = False
) -> Tensor:

    if only_fg:
        not_bg = object_mask > 0
        true_image = true_image[not_bg]
        reconstruction = reconstruction[not_bg]
    
    return ((true_image - reconstruction)**2).mean([1, 2, 3]).mean()


def slot_mcc(
            run1_slots: Tensor,
            run2_slots: Tensor
    ):
    return slot_mean_corr_coef(run1_slots, run2_slots)




@torch.no_grad()
def calculate_fid(real_path: str,
                    fake_path: str):
    return fid_score.calculate_fid_given_paths(paths = [str(real_path), str(fake_path)], 
                                                dims = 2048, 
                                                device=0,
                                                batch_size= 256, 
                                                num_workers = 8)




def compositional_contrast(
        latents: Tensor,
        mixing_fn: Tensor
) -> Tensor:
    b, K, d = latents.shape
    
    jac = jacfwd(mixing_fn)(latents)
    import pdb; pdb.set_trace()


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


def r2_score(
    true_latents: Tensor, 
    ordered_predicted_latents: Tensor, 
) -> float:
    """
    Calculates R2 score. Slots are flattened before calculating R2 score.

    Args:
        true_latents: tensor of shape (batch_size, n_slots, n_latents)
        predicted_latents: tensor of shape (batch_size, n_slots, n_latents)
        indices: tensor of shape (batch_size, n_slots, 2) with indices of matched slots

    Returns:
        avg_r2_score: average R2 score over all latents
    """

    # BK x d

    predicted_latents = ordered_predicted_latents.detach().cpu()
    true_latents = true_latents.detach().cpu()

    r2 = R2Score(true_latents.shape[1], multioutput="raw_values")
    r2_score_raw = r2(predicted_latents, true_latents)
    r2_score_raw[torch.isinf(r2_score_raw)] = torch.nan
    avg_r2_score = torch.nanmean(r2_score_raw).item()
    return avg_r2_score
