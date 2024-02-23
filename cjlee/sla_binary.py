# import library
import numpy as np
import torch
import sys
import argparse
from ad_model import AD_SUP2_MODEL3
import os

class Args:
    def __init__(self):
        self.use_feature_mapping=None
        self.dim_feature_mapping=None
        self.dim_input=None
        self.nhead=None
        self.dim_feedforward=None
        self.reduce=None
        self.nlayer=None
        self.clf_n_fc_layers=None
        self.clf_dim_fc_hidden=None
        self.clf_dim_output=None

def call_model(load_path, device):
    args = Args() 
    args.use_feature_mapping = 1
    args.dim_feature_mapping = 24
    args.dim_input = 6
    args.nhead = 4
    args.dim_feedforward = 48
    args.reduce = 'max'
    args.nlayer = 2
    args.clf_n_fc_layers = 3
    args.clf_dim_fc_hidden = 600
    args.clf_dim_output = 2
    model = AD_SUP2_MODEL3(args)
    model.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
    model = model.to(device)
    
    return model
    
class myNormalizer:
    def __init__(self, stat_path):
        fp = open(stat_path, 'r')
        lines = fp.readlines()
        self.x_avg= np.asarray([float(s) for s in lines[0].split(',')])
        self.x_std= np.asarray([float(s) for s in lines[1].split(',')])
        
        for i in range(len(self.x_std)):
            if self.x_std[i] == 0:
                self.x_std[i] = 0.001
        fp.close()
        
    def __call__(self, x_data):
        x_data = (x_data - self.x_avg) / self.x_std

        return x_data

#load_path = '/ni-anomaly-detection-public/cjlee/AT-7.pth'
#stat_path = '/ni-anomaly-detection-public/cjlee/tpi_train.csv.stat'

load_path = './cjlee/AT-7.pth'
stat_path = './cjlee/tpi_train.csv.stat'

current_directory = os.getcwd()
print("Current Directory:", current_directory)

V, D = 5, 6

# load model
device = torch.device('cpu')
print('l', load_path)
model = call_model(load_path=load_path, device=device)

# load normalizer
normalizer = myNormalizer(stat_path)

# obtain arguments
input = []
for i, arg in enumerate(sys.argv):
    if i==0: continue
    input.append(float(arg))
input = np.expand_dims(np.array(input), axis=0)

input = normalizer(input)

input = input.reshape(V, D)
input = torch.tensor(input).type(torch.float32).to(device)
input = input.unsqueeze(0)

# run model with arguments
output = model(input)
res = int(torch.argmax(output.detach()))

print(res)
