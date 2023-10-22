import os
import argparse


from tqdm import tqdm
import time, math
from datetime import datetime, timedelta

import numpy as np
import torch
import json
import pandas as pd 
from tqdm import tqdm
from sklearn.metrics.cluster import adjusted_rand_score
from metrics import ari, compositional_contrast, slot_mcc, mse, calculate_fid, calculate_sfid

import sys 
sys.path.append('..')
from model import SlotAutoEncoder
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--checkpoint_dir', default=16, type=int)
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')

opt = parser.parse_args()



def main(run):
    pass 
    


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