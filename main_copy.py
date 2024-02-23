from pprint import pprint
#import nfvo_client
from config import cfg

#from nfvo_client.rest import ApiException as NfvoApiException

from ad_module import *
from time import sleep
import csv
import subprocess
import datetime as dt
import torch
import numpy as np

import sys
sys.path.insert(1, './cjlee')

from ad_model import AD_SUP2_MODEL3

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
def new_model(device):
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
    model=model.to(device)
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
        self.x_avg=self.x_avg.reshape(5,6)
        self.x_std=self.x_std.reshape(5,6)
        self.x_avg=self.x_avg[[0,2]].flatten()
        self.x_std=self.x_std[[0,2]].flatten()

    def __call__(self, x_data):
        print(x_data.shape)
        print(self.x_avg.shape)
        x_data = (x_data - self.x_avg) / self.x_std

        return x_data

def str2np_array(vnf_resource_string):
    '''
    in: vnf resource string
    out: np array (1, num_features)
    '''    
    arr = vnf_resource_string.strip().split(' ')

    arr = np.array(arr).astype(np.float32)
    arr = arr.reshape(1,-1)

    return arr

if __name__ == '__main__':
    # 1. setup model and device
#    load_path = '/ni-anomaly-detection-public/cjlee/AT-7.pth'
#    stat_path = '/ni-anomaly-detection-public/cjlee/tpi_train.csv.stat'

    load_path = './cjlee/AT-7.pth'
    stat_path = './cjlee/tpi_train.csv.stat'
    
    V, D = 5, 6

    device = torch.device('cpu')
    model = call_model(load_path=load_path, device=device)
    normalizer = myNormalizer(stat_path)

    # 2. Get VNF Information
    #sfc_vnfs = ["fw", "dpi", "fm", "ids", "lb"]
    #prefix = "jb_"

    sfc_vnfs = ["firewall", "flowmonitor", "dpi", "ids", "lb"]
    prefix = "and-"

    vnfi_info = get_vnf_info(prefix, sfc_vnfs)
    vnfi_list = vnfi_info["vnfi_list"]

    while(True):
        vnf_resources = get_vnf_resources(vnfi_list)
        vnf_resources_string = get_vnf_resources_toString(vnf_resources)

        x_data = str2np_array(vnf_resources_string)        
        x_data = normalizer(x_data)

        x_data = x_data.reshape(V, D)
        x_data = torch.tensor(x_data).type(torch.float32).to(device)
        x_data = x_data.unsqueeze(0)

        output = model(x_data)
        sla_binary_result = int(torch.argmax(output.detach()))
        # save into csv file
        # sla_binary_body = "python3 ./cjlee/sla_binary.py " + vnf_resources_string
        # sla_binary_result = subprocess.check_output(sla_binary_body.split()).decode('utf-8').strip()
        print(sla_binary_result)
        line = vnf_resources_string + str(sla_binary_result) 

        print(line)
        with open("./data.csv", "at") as fp:
            fp.write(line + "\n")
            
        sleep(4)
