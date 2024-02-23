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
	#sfc_vnfs = ["firewall", "ids", "proxy", "dpi", "flowmonitor"]
	sfc_vnfs = ["firewall", "flowmonitor", "dpi", "ids", "lb"]
        #prefix = "suman-"
        prefix = "and-"

	vnfi_info = get_vnf_info(prefix, sfc_vnfs)
	vnfi_list = vnfi_info["vnfi_list"]
	print(vnfi_list)

	while(True):
		vnf_resources = get_vnf_resources(vnfi_list)
		import pdb; pdb.set_trace()
		vnf_resources_string = get_vnf_resources_toString(vnf_resources)

		sla_binary_body = "java -cp .:models/sla_binary/h2o-genmodel_sla.jar sla_binary "+vnf_resources_string
		sla_binary_result = subprocess.check_output(sla_binary_body.split()).decode('utf-8').split("\n")[0]

		print(dt.datetime.now())

		if sla_binary_result == "0":
			print("Normal")
		else:
			print("Abnormal, Root-Cause Localization Analysis start..")
            
			sla_rcl_body = "java -cp .:models/sla_rcl/h2o-genmodel_rcl.jar sla_rcl "+vnf_resources_string
			sla_rcl_result = subprocess.check_output(sla_rcl_body.split()).decode('utf-8').split("\n")[0]
 		
			if sla_rcl_result == "none":
				print("Analysis Result: traffic overload")
			else:
				print("Analysis Result: "+sla_rcl_result)
			
		sleep(1)
