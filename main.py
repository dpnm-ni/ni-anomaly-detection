from pprint import pprint
#import nfvo_client
from config import cfg

#from nfvo_client.rest import ApiException as NfvoApiException

import csv
import subprocess
import datetime as dt
from ad_module import *
from time import sleep
import os
import sys
#sys.path.insert(1, './cjlee')

if __name__ == '__main__':
    # 1. Get VNF Information
    #sfc_vnfs = ["fw", "fm", "dpi", "ids", "lb"]
    sfc_vnfs = ["firewall", "flowmonitor", "dpi", "ids", "lb"]
    #prefix = "jb_"
    prefix = "and-"

    vnfi_info = get_vnf_info(prefix, sfc_vnfs)
    vnfi_list = vnfi_info["vnfi_list"]

    while(True):
        vnf_resources = get_vnf_resources(vnfi_list)
        vnf_resources_string = get_vnf_resources_toString(vnf_resources)
        # save into csv file
        sla_binary_body = "python3 cjlee/sla_binary.py " + vnf_resources_string
        current_directory = os.getcwd()
        print("Current Directory:", current_directory)
        sla_binary_result = subprocess.check_output(sla_binary_body.split()).decode('utf-8').strip()

        line = vnf_resources_string + sla_binary_result

        with open("./data_try2.csv", "at") as fp:
            fp.write(line + "\n")
            
        sleep(1)
