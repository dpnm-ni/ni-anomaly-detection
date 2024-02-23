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
    sfc_vnfs = ["firewall", "flowmonitor", "dpi", "ids", "proxy"]
    prefix = "and-"
    vnfi_info = get_vnf_info(prefix, sfc_vnfs)
    vnfi_list = vnfi_info["vnfi_list"]
    print(vnfi_list)
    sfcr_name = "and_sfcr"
    ni_nfvo_sfcr_api = get_nfvo_sfcr_api()
    sfcrs = ni_nfvo_sfcr_api.get_sfcrs()
    src_ip_prefix = get_ip_from_id('53566d10-5725-42de-bcbd-bd386761e05e')
    source_client = '53566d10-5725-42de-bcbd-bd386761e05e'
    nf_chain = ["aef4913d-e392-4b15-b364-8a1a3d0c1027", \
                "541e5d40-379b-49c6-8487-b2e22cde625c", \
                "251e6293-693a-491e-903a-b1aebe385944", \
                "2c162efb-ff0d-4abc-8cec-f5f099bfe840", \
                "53566d10-5725-42de-bcbd-bd386761e05e"]
    print(sfcrs)
    sfcr_id = ""
    #nf_chain = get_sfcr_by_id(sfcr_id).nf_chain
    sfcr_spec = ni_nfvo_client.SfcrSpec(name=sfcr_name,
                                     src_ip_prefix=src_ip_prefix,
                                     nf_chain=sfc_vnfs,
                                     source_client=source_client)
    print(sfcr_spec)
