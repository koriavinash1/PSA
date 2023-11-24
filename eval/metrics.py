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

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import kernel_ridge

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



def r2_score(Z: torch.Tensor, hZ: torch.Tensor) -> np.ndarray:
    """
    Computes matrix of R2 scores between all inferred and ground-truth latent slots

    Args:
        Z: Tensor containing all ground-truth latents
        hZ: Tensor containing all inferred latents, ordered wrt Z

    Returns:
        numpy array of R2 scores of shape: [num_slots]
    """
    num_slots = Z.shape[1]

    Z = Z.permute(1, 0, 2).numpy()
    hZ = hZ.permute(1, 0, 2).numpy()

    # Initialize matrix of R2 scores
    corr = np.zeros(num_slots)

    # 'hZ', 'Z' have shape: [num_slots, num_samples, slot_dim]

    # Use kernel ridge regression to predict ground-truth from inferred slots
    reg_func = lambda: kernel_ridge.KernelRidge(kernel="rbf", alpha=1.0, gamma=None)
    for i in range(num_slots):
        ZS = Z[i]
        hZS = hZ[i]

        # Standardize latents
        scaler_Z = StandardScaler()
        scaler_hZ = StandardScaler()

        z_train, z_eval = np.split(scaler_Z.fit_transform(ZS), [int(0.8 * len(ZS))])
        hz_train, hz_eval = np.split(
            scaler_hZ.fit_transform(hZS), [int(0.8 * len(hZS))]
        )

        # Fit KRR model
        reg_model = reg_func()
        reg_model.fit(hz_train, z_train)
        hz_pred_val = reg_model.predict(hz_eval)

        # Populate correlation matrix
        corr[i] = sklearn.metrics.r2_score(z_eval, hz_pred_val)

    return corr.mean()