import os
import argparse
from src.dataset import *
from src.model import SlotAttentionAutoEncoder
from src.metrics import compositional_fid

from tqdm import tqdm
import time, math
from datetime import datetime, timedelta

import numpy as np
import torch
import json
import pandas as pd 
from tqdm import tqdm
from sklearn.metrics.cluster import adjusted_rand_score

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true', '1')


parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_batches', default=1, type=int)
parser.add_argument('--num_slots', default=7, type=int)
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')

opt = parser.parse_args()


normalize = lambda x: (x - np.min(x))/(np.max(x) - np.min(x) + 1e-5)

def get_ari(config):
    # load parameters....
    exp_arguments = json.load(open(config, 'r'))
    print(exp_arguments)
    print ('='*25)

    resolution = (exp_arguments['img_size'], exp_arguments['img_size'])
    # ===========================================
    # model init
    model = SlotAttentionAutoEncoder(resolution, 
                                        opt.num_slots, 
                                        exp_arguments['num_iterations'], 
                                        exp_arguments['hid_dim'],
                                        exp_arguments['max_slots'],
                                        exp_arguments['nunique_objects'],
                                        exp_arguments['quantize'],
                                        exp_arguments['cosine'],
                                        exp_arguments['cb_decay'],
                                        exp_arguments['encoder_res'],
                                        exp_arguments['decoder_res'],
                                        exp_arguments['kernel_size'],
                                        exp_arguments['cb_qk'],
                                        exp_arguments['eigen_quantizer'],
                                        exp_arguments['restart_cbstats'],
                                        exp_arguments['implicit'],
                                        exp_arguments['gumble'],
                                        exp_arguments['temperature'],
                                        exp_arguments['kld_scale']).to(device)


    ckpt=torch.load(os.path.join(exp_arguments['model_dir'], 'discovery_best.pth' ), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.device = device


    def _get_segmentation_mask(masks, cbidxs, fg):
        segmentation_mask = np.zeros(resolution)
        masks = masks.cpu().detach().numpy()
        sampled_idxs = cbidxs.cpu().detach().numpy()
        segmentation_mask = np.argmax(masks*fg[None, ..., None], axis=0)[:,:,0]

        # threshold = 0.7
        # unique_idxs = np.unique(sampled_idxs)
        # for ii, uidx in enumerate(unique_idxs):
        #     idxs = np.where(sampled_idxs == uidx)[0]

        #     for js in idxs:
        #         slot_mask = masks[js][:,:,0]
        #         slot_mask = normalize(slot_mask)
                
        #         if np.sum(slot_mask) > 0.3*np.prod(resolution):
        #             js=0

        #         if len(unique_idxs) > 1:
        #             segmentation_mask[slot_mask > threshold] = js # ii # CLEVR does not have object specific mask idx
        #         else:
        #             segmentation_mask[slot_mask > threshold] = js 

        # for js in range(opt.num_slots):
        #     slot_mask = masks[js][:,:,0]*fg
        #     slot_mask = normalize(slot_mask) 
            
        #     slot_mask = slot_mask > 0.5 # threshold

        #     # if np.sum(slot_mask) > 0.25*np.prod(resolution):
        #     #     js=0

        #     segmentation_mask += (js +1)*slot_mask


        return segmentation_mask


    test_set = DataGenerator(root='/vol/biomedic3/agk21/datasets/multi-objects/RawData-subset/clevr_with_masks/images', 
                                mode='test',
                                masks=True,
                                resolution=resolution)
    test_dataloader = torch.utils.data.DataLoader(test_set, 
                                        batch_size=opt.batch_size,
                                        shuffle=True, 
                                        num_workers=opt.num_workers, 
                                        drop_last=True)

    test_epoch_size = min(10000, len(test_dataloader))

    ari = []
    for _ in range(3):
        ari_fold = []
        for _ in tqdm(range(300)):
            imgs = next(iter(test_dataloader))
            gt_masks = imgs['mask'].cpu().numpy()
            
            imgs = imgs['image']
            imgs = imgs.to(device)
            
            recon_combined, recons, masks, slots, cbidxs, qloss, perplexity = model(imgs)
            for i in range(imgs.shape[0]):
                gt_mask = gt_masks[i]
                gt_mask_fg = np.array(gt_mask != 0)
                segmentation_mask = _get_segmentation_mask(masks[i], cbidxs[i], gt_mask_fg)
                gt_seq = gt_mask[np.where(gt_mask_fg > 0)]
                pred_seq = segmentation_mask[np.where(gt_mask_fg > 0)]
                ari_fold.append(adjusted_rand_score(pred_seq, gt_seq))

                # plt.clf()
                # plt.subplot(1,4,1)
                # plt.imshow(imgs[i].cpu().numpy().transpose(1,2,0))

                # plt.subplot(1,4,2)
                # plt.imshow(gt_mask)

                # plt.subplot(1,4,3)
                # plt.imshow(segmentation_mask)

                # plt.subplot(1,4,4)
                # plt.imshow(gt_mask_fg)
                # plt.colorbar()
                # plt.savefig(f'{i}.png')

            # import pdb;pdb.set_trace()
        print (np.mean(ari_fold))
        ari.append(np.mean(ari_fold))
    
    return np.mean(ari), np.std(ari)
    


if __name__ == '__main__':
    configs = [
                '/vol/biomedic3/agk21/testEigenSlots2/LOGS-IMPLICIT2/ObjectDiscovery/clevrdefaultBaseline/exp-parameters.json',
                '/vol/biomedic3/agk21/testEigenSlots2/LOGS-IMPLICIT2/ObjectDiscovery/clevrdefaultCosine/exp-parameters.json',
                '/vol/biomedic3/agk21/testEigenSlots2/LOGS-IMPLICIT2/ObjectDiscovery/clevrdefaultEuclidian/exp-parameters.json',
                '/vol/biomedic3/agk21/testEigenSlots2/LOGS-IMPLICIT2/ObjectDiscovery/clevrdefaultGumble/exp-parameters.json',
                '/vol/biomedic3/agk21/testEigenSlots2/LOGS-NOIMPLICIT/ObjectDiscovery/clevrdefaultBaseline/exp-parameters.json',
                '/vol/biomedic3/agk21/testEigenSlots2/LOGS-NOIMPLICIT/ObjectDiscovery/clevrdefaultCosine/exp-parameters.json',
                '/vol/biomedic3/agk21/testEigenSlots2/LOGS-NOIMPLICIT/ObjectDiscovery/clevrdefaultEuclidian/exp-parameters.json',
                '/vol/biomedic3/agk21/testEigenSlots2/LOGS-NOIMPLICIT/ObjectDiscovery/clevrdefaultGumble/exp-parameters.json',
   
                ]
    
    means = []
    stds = []

    for i, config in enumerate(configs):
        mean, std = get_ari(config)
        means.append(mean)
        stds.append(std)

        df = pd.DataFrame({'config': configs[:i+1], 'mean': means, 'std': stds})
        df.to_csv(f'/vol/biomedic3/agk21/testEigenSlots2/CSVS/aris.csv')