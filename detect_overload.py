from pprint import pprint
#import nfvo_client
from config import cfg

#from nfvo_client.rest import ApiException as NfvoApiException

import csv
import subprocess
import datetime as dt

from ad_module import *
from time import sleep

if __name__ == '__main__':
    # 1. Get VNF Information
    #sfc_vnfs = ["firewall", "flowmonitor", "dpi", "ids", "lb"]
    #prefix = "jb_"
    sfc_vnfs = ["firewall", "flowmonitor", "dpi", "ids", "lb"]
    prefix = "and-"
    
    vnfi_info = get_vnf_info(prefix, sfc_vnfs)
    vnfi_list = vnfi_info["vnfi_list"]
        
    while(True):
        vnf_resources = get_vnf_resources(vnfi_list)
        vnf_resources_string = get_vnf_resources_toString(vnf_resources)
        
        body = "java -cp .:models/resource_overload/h2o-genmodel_overload.jar resource_overload "+vnf_resources_string
        result = subprocess.check_output(body.split()).decode('utf-8').split("\n")[0]
        current_time = dt.datetime.now()
        print(current_time)
        if result == "none":
            print("Normal")
        else:
            print("Abnormal - "+result)
        sleep(1)
