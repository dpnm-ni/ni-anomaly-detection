import connexion
import six
#import json

from swagger_server.models.vnf_instance import VNFInstance  # noqa: E501
from swagger_server import util

import ad_module as ad
#from swagger_server.models.ad_info import adInfo
#from swagger_server.models.network_port import NetworkPort

def get_vnf_info(prefix):
    
    sfc_vnfs = ["firewall", "dpi"]
    vnf_info = ad.get_vnf_info(prefix, sfc_vnfs)
    vnfi_list = vnf_info["vnfi_list"]
    result = ad.convert_vnf_info(vnfi_list)

    return result


def get_sla_detection_result(prefix):

    sfc_vnfs = ["firewall", "dpi"]
    result = ad.get_sla_detection_result(prefix, sfc_vnfs)

    return result

def get_ad_f1score():

    result = ad.get_ad_f1score()

    return result

def post_train_model():
    ad.train_model()

def get_resource_overload_detection_result(prefix):

    sfc_vnfs = ["firewall", "dpi"]
    result = ad.get_resource_overload_detection_result(prefix, sfc_vnfs)

    return result

def get_vnf_resource_usage(prefix):

    #sfc_vnfs = ["firewall", "flowmonitor", "dpi", "ids", "lb"]
    sfc_vnfs = ["firewall", "dpi"]
    result = ad.get_vnf_resource_usage(prefix, sfc_vnfs)

    return result

def create_sfc(prefix):

    sfc_vnfs = ["firewall", "flowmonitor", "dpi", "ids", "lb"]
    result = ad.get_vnf_resource_usage(prefix, sfc_vnfs)

    return result

