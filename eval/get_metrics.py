import os
import argparse

import torchvision
from tqdm import tqdm
import time, math
from datetime import datetime, timedelta

import numpy as np
import torch
import json
import pandas as pd 
from tqdm import tqdm
from metrics import (ari, 
                    compositional_contrast, 
                    slot_mcc, 
                    slot_mean_corr_coef,
                    mse, 
                    r2_score,
                    calculate_fid)


import sys 
import copy

sys.path.append('..')
from hps import Hparams
from train_setup import setup_dataloaders
from model import SlotAutoEncoder
from utils import preprocess_batch
import matplotlib.pyplot as plt
from tqdm import tqdm


@torch.no_grad()
def main(opt: argparse.ArgumentParser):

    ckpt_dir = opt.checkpoint_dir
    run_dirs = [dir_str for dir_str in os.listdir(ckpt_dir) if dir_str.__contains__('Run')]
    
    config = torch.load(os.path.join(ckpt_dir, run_dirs[0], 'checkpoint.pt'))['hparams']
    hparams = Hparams()
    hparams.update(config)

    test_dataloader = setup_dataloaders(hparams)['test']

    metric_logs = [] 
    run_logs = []
    for ir, run_dir in enumerate(run_dirs):
        ckpt_path = os.path.join(ckpt_dir, run_dir, 'checkpoint.pt')
        ckpt = torch.load(ckpt_path)
        model = SlotAutoEncoder(hparams)

        fid_dir = os.path.join(ckpt_dir, run_dir, 'fid_directory')
        os.makedirs(fid_dir, exist_ok=True)



        if opt.use_ema_model:
            model.load_state_dict(ckpt['ema_model_State_dict'])
        else:
            model.load_state_dict(ckpt['model_state_dict'])

        model.eval()
        cpu_model = copy.deepcopy(model)
        model.to(opt.device)

        logs = {
                'initial_slots': [],
                'final_slots': [],
                'compositional_initial_slots': [],
                'compositional_final_slots': []
                }
        eval_metrics = {'count': 0,
                        'ari': 0,
                        'mse_full': 0,
                        'mse_fg': 0,
                        'compositional_constraint': 0}

        


        for i, batch in tqdm(enumerate(test_dataloader), desc=f'Computing metrics for Run-{ir}'):
            if i > 100: break

            batch = preprocess_batch(hparams, batch)
            bs = batch["x"].shape[0]

            if 'properties' not in batch.keys():
                batch['properties'] = None

            zs = model.encoder(x=batch["x"])

            (x_rec, _), recons, masks, attn, (init_slots, final_slots) = \
                                            model.forward_latents(latents = zs, 
                                                properties = batch['properties'])



            # metrics
            eval_metrics['count'] += 1

            if (not hparams.no_additive_decoder) and ('mask' in batch.keys()):
                # compute ari
                slot_mask = masks.argmax(dim = 1).squeeze(1) # B x 1 x H x W
                gt_mask = batch['mask'].argmax(dim = 1).squeeze(1)
                eval_metrics['ari'] += ari(gt_mask, slot_mask, 1)
            

            eval_metrics['mse_full'] += mse(batch['x'], x_rec, only_fg = False)
            if 'mask' in batch.keys(): eval_metrics['mse_fg']   += mse(batch['x'], x_rec, gt_mask, only_fg = True)
            # eval_metrics['compositional_constraint'] += compositional_contrast(final_slots.cpu(), cpu_model.decode_slots) # memory peaked at 120GB on cpu


            # append latents for mcc calculation
            logs['initial_slots'].append(init_slots.detach().cpu())
            logs['final_slots'].append(final_slots.detach().cpu())


            # save real images
            real_img_path = os.path.join(fid_dir, 'real_img')
            os.makedirs(real_img_path, exist_ok=True)

            recon_img_path = os.path.join(fid_dir, 'recon_img')
            os.makedirs(recon_img_path, exist_ok=True)

            real_slots_path = os.path.join(fid_dir, 'real_slots')
            os.makedirs(real_slots_path, exist_ok=True)

            if not hparams.no_additive_decoder:
                recon_slots_path = os.path.join(fid_dir, 'recon_slots')
                os.makedirs(recon_slots_path, exist_ok=True)


            for k in range(bs):
                filename = str(k + i * bs)
                image = batch['x'].cpu()[k]
                torchvision.utils.save_image(image, os.path.join(real_img_path, f'{filename}.png'))

                recon_img = x_rec.detach().cpu()[k]
                torchvision.utils.save_image(recon_img, os.path.join(recon_img_path, f'{filename}.png'))


                if 'mask' in batch.keys():
                    for islot, slot in enumerate(batch['mask'].cpu()[k]):
                        slot = image*slot
                        torchvision.utils.save_image(slot, os.path.join(real_slots_path, f'{filename}-{islot}.png'))

                if not hparams.no_additive_decoder:
                    recons = recons* masks + (1 - masks)

                    for islot, slot in enumerate(recons[k].cpu()):
                        torchvision.utils.save_image(slot, os.path.join(recon_slots_path, f'{filename}-{islot}.png'))
 

            if hparams.model in ['VSA', 'VASA', 'SSA', 'SSAU']:
                composition_img_path = os.path.join(fid_dir, 'composition_img')
                os.makedirs(composition_img_path, exist_ok=True)

                if not hparams.no_additive_decoder:
                    sampled_slot_path = os.path.join(fid_dir, 'sampled_slots')
                    os.makedirs(sampled_slot_path, exist_ok=True)


                (x_comp, _), recons, masks, attn, (sampled_init_slots, sampled_final_slots) = model.sample(bs, 
                                                                        device = opt.device, 
                                                                        properties = batch['properties'], 
                                                                        return_loc=True)

                logs['compositional_initial_slots'].append(sampled_init_slots.detach().cpu())
                logs['compositional_final_slots'].append(sampled_final_slots.detach().cpu())


                for k in range(bs):
                    filename = str(k + i * bs)
                    comp_img = x_comp.detach().cpu()[k]
                    torchvision.utils.save_image(comp_img, os.path.join(composition_img_path, f'{filename}.png'))

                    if not hparams.no_additive_decoder:
                        recons = recons* masks + (1 - masks)

                        for islot, slot in enumerate(recons[k].cpu()):
                            torchvision.utils.save_image(slot, os.path.join(sampled_slot_path, f'{filename}-{islot}.png'))


        # ========================================================================
        for key in eval_metrics.keys():
            if isinstance(eval_metrics[key], torch.Tensor):
                eval_metrics[key] = eval_metrics[key].cpu().numpy() * 1.0/eval_metrics['count']
    

        eval_metrics['recon_fid'] = calculate_fid(real_img_path, recon_img_path)

        if ('mask' in batch.keys()) and (not hparams.no_additive_decoder):
            eval_metrics['slot_fid'] = calculate_fid(real_slots_path, recon_slots_path)


        if hparams.model in ['VSA', 'VASA', 'SSA', 'SSAU']:
            eval_metrics['model_compositional_fid'] = calculate_fid(recon_img_path, composition_img_path)
            eval_metrics['true_compositional_fid'] = calculate_fid(real_img_path, composition_img_path)

            if not hparams.no_additive_decoder:
                eval_metrics['model_compositional_slot_fid'] = calculate_fid(recon_slots_path, sampled_slot_path)
                if 'mask' in batch.keys():
                    eval_metrics['true_compositional_slot_fid'] = calculate_fid(real_slots_path, sampled_slot_path)



        # =====================================================================
        for key in logs:
            if len(logs[key]): 
                logs[key] = torch.cat(logs[key], 0)

                mcc_score, r2_ = 0.0, 0.0
                if len(run_logs):
                    mcc_score, ordered_z = slot_mean_corr_coef(run_logs[-1][key], logs[key], return_ordered = True)
                    r2_ = r2_score(run_logs[-1][key].flatten(0, 1), ordered_z)

                eval_metrics['SMCC_' + key] = mcc_score
                eval_metrics['R2_' + key]   = r2_ 


        run_logs.append(logs)

        metric_logs.append(eval_metrics)


    all_metrics = metric_logs[-1].keys()
    return_metrics = {}

    for metric in all_metrics:
        if metric == 'count': continue

        tmp = []
        for ii in range(len(metric_logs)):
            if metric_logs[ii][metric] == 0.0:
                tmp.append(metric_logs[ii][metric])
        mean = np.mean(tmp); var = np.var(tmp)

        return_metrics[metric + '_mean'] = float(mean)
        return_metrics[metric + '_var'] = float(var)

    return return_metrics, metric_logs






if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', 
                        default=16, 
                        type=int)
    parser.add_argument('--use_ema_model',
                        action="store_true",
                        default=False)
    parser.add_argument('--checkpoint_dir', 
                        type=str)

    opt = parser.parse_args()
    opt.device = device

    import json

    return_metrics, metric_logs = main(opt)

    metric_logs = {f'Run-{i}': {k: float(v) for k, v in info.items()} for i, info in enumerate(metric_logs)}

    with open(os.path.join(opt.checkpoint_dir, 'final_logs.json'), 'w') as json_file:
        json.dump(return_metrics, json_file, indent=4)
    
    with open(os.path.join(opt.checkpoint_dir, 'run_logs.json'), 'w') as json_file:
        json.dump(metric_logs, json_file, indent=4)
    