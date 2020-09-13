from __future__ import print_function
import ni_nfvo_client
import ni_mon_client
from datetime import datetime, timedelta
from ni_nfvo_client.rest import ApiException as NfvoApiException
from ni_mon_client.rest import ApiException as NimonApiException
from pprint import pprint
from config import cfg

import random
import numpy as np
import datetime as dt

import csv
import subprocess
import json


# get_monitoring_api(): get ni_monitoring_client api to interact with a monitoring module
# Input: null
# Output: monitoring moudle's client api
def get_monitoring_api():

    ni_mon_client_cfg = ni_mon_client.Configuration()
    ni_mon_client_cfg.host = cfg["ni_mon"]["host"]
    ni_mon_api = ni_mon_client.DefaultApi(ni_mon_client.ApiClient(ni_mon_client_cfg))

    return ni_mon_api



# get_nfvo_sfc_api(): get ni_nfvo_sfc api to interact with a nfvo module
# Input: null
# Output: nfvo moudle's sfc api
def get_nfvo_sfc_api():

    nfvo_client_cfg = ni_nfvo_client.Configuration()
    nfvo_client_cfg.host = cfg["ni_nfvo"]["host"]
    ni_nfvo_sfc_api = ni_nfvo_client.SfcApi(ni_nfvo_client.ApiClient(nfvo_client_cfg))

    return ni_nfvo_sfc_api


# get_nfvo_sfcr_api(): get ni_nfvo_sfcr api to interact with a nfvo module
# Input: null
# Output: nfvo moudle's sfcr api
def get_nfvo_sfcr_api():

    nfvo_client_cfg = ni_nfvo_client.Configuration()
    nfvo_client_cfg.host = cfg["ni_nfvo"]["host"]
    ni_nfvo_sfcr_api = ni_nfvo_client.SfcrApi(ni_nfvo_client.ApiClient(nfvo_client_cfg))

    return ni_nfvo_sfcr_api



# get_ip_from_vm(vm_id):
# Input: vm instance id
# Output: port IP of the data plane
def get_ip_from_id(vm_id):

    ni_mon_api = get_monitoring_api()
    query = ni_mon_api.get_vnf_instance(vm_id)

    ## Hard coding for network id of the data plane
    ports = query.ports
    network_id = "52b3b564-0be1-49f2-9b67-1cee170acbdb"

    for port in ports:
        if port.network_id == network_id:
            return port.ip_addresses[-1]


# get_vnf_info(sfc_prefix, sfc_vnfs): get each VNF instance ID and information from monitoring module
# Input: Prefix of VNF instance name, SFC order tuple [example] ("client", "firewall", "dpi", "ids", "proxy")
# Output: Dict. object = {'vnfi_info': vnfi information, 'num_vnf_type': number of each vnf type}
def get_vnf_info(sfc_prefix, sfc_vnfs):

    # Get information of VNF instances which are used for SFC
    ni_mon_api = get_monitoring_api()
    query = ni_mon_api.get_vnf_instances()

    selected_vnfi = [ vnfi for vnfi in query for vnf_type in sfc_vnfs if vnfi.name.startswith(sfc_prefix + vnf_type) ]
    node_ids = [ vnfi.node_id for vnfi in selected_vnfi ]
    node_ids = list(set(node_ids))

    vnfi_list = []
    num_vnf_type = []
    temp = []

    # Sort VNF informations for creating states
    for vnf_type in sfc_vnfs:
        i =  sfc_vnfs.index(vnf_type)

        temp.append([])

        temp[i] = [ vnfi for vnfi in selected_vnfi if vnfi.name.startswith(sfc_prefix + vnf_type) ]
        temp[i].sort(key=lambda vnfi: vnfi.name)

        for vnfi in temp[i]:
            vnfi.node_id = node_ids.index(vnfi.node_id)

        vnfi_list = vnfi_list + temp[i]
        num_vnf_type.append(len(temp[i]))

    return {'vnfi_list': vnfi_list, 'num_vnf_type': num_vnf_type}


# get_vnf_resources(vnfi_list): get resources info. of VNF instance from the monitoring module
# Input: VNF instance list
# Output: Resource array -> [(CPU utilization, Memory utilization, Physical location), (...), ...]
def get_vnf_resources(vnfi_list):

    # In this codes, we regard CPU utilization, Memory utilization, Physicil node location
    resource_type = ("cpu_usage___value___gauge", "memory_free___value___gauge", "vda___disk_octets___read___derive", "vda___disk_octets___write___derive")
    ni_mon_api = get_monitoring_api()

    # Create an initial resource table initialized by 0
    resources = np.zeros((len(vnfi_list), len(resource_type)+2))
    
    # Query to get resource data
    for vnfi in vnfi_list:
        i = vnfi_list.index(vnfi)

        vnf_id = vnfi.id

        #set time
        end_time = dt.datetime.now()
        start_time = end_time - dt.timedelta(seconds = 10)

        for type in resource_type:
            j = resource_type.index(type)
            
            measurement_type = type

            response = ni_mon_api.get_measurement(vnf_id, measurement_type, start_time, end_time)
            resources[i, j] = response[-1].measurement_value

            # Calculate CPU utilization as persent
            if j == 0:
                resources[i, j] = resources[i, j]

            # Calculate Memory utilization as percent
            elif j == 1:
                flavor_id = vnfi_list[i].flavor_id
                memory_query = ni_mon_api.get_vnf_flavor(flavor_id)
                memory_total = 1000000 * memory_query.ram_mb
                resources[i, j] = (resources[i, j]/memory_total)*100

            # calculate disk io read
            elif j == 2:
                resources[i, j] = resources[i, j]

            # calculcate disk io write
            elif j == 3:
                resources[i, j] = resources[i, j]

        # get vnf network_interface(subnet 10.10.20.xx) info
        nic_addresses = vnfi.ports
        nic_id = ""

        for k in range(len(nic_addresses)):
            #print(nic_addresses[k].ip_addresses)
            #print(nic_addresses[k].ip_addresses[-1][6:8])
            if nic_addresses[k].ip_addresses[-1][6:8] == "20":
                nic_id = nic_addresses[k].port_name
            else:
                continue
       
        nic_info_type = (nic_id+"___if_octets___rx___derive", nic_id+"___if_octets___tx___derive")
        j = 4
        for info_type in nic_info_type:
            measurement_type = info_type
            response = ni_mon_api.get_measurement(vnf_id, measurement_type, start_time, end_time)
            resources[i, j] = (response[-1].measurement_value)
            j = j+1

    resources = np.round(resources,3)

    return resources



# get_vnf_type(current_state, num_vnf_type): get vnf type showing vnf order of SFC
# Input: current state number, number of each vnf instance
# Output: vnf type (the order which is index number of vnf in sfc)
def get_vnf_type(current_state, num_vnf_type):

    index = len(num_vnf_type)
    pointer = num_vnf_type[0]

    for i in range (0, index):
        if current_state < pointer:
            return i
        else:
            pointer = pointer + num_vnf_type[i+1]


def get_vnf_resources_toString(vnf_resources):

    vnf_resources_string = ""
    for i in range(0, len(vnf_resources)):
        for j in range(0,len(vnf_resources[i])):
            if i == len(vnf_resources)-1 and j == len(vnf_resources[i]-1) :
                vnf_resources = vnf_resources_string + str(vnf_resources[i][j])
            else:
                vnf_resources_string = vnf_resources_string + str(vnf_resources[i][j])+" "

    return vnf_resources_string


# get SLA detection results in real-time
# input: vnf_instance name prefix, VNF types
# Output: result (string) 

def get_sla_detection_result(prefix, sfc_vnfs):

    result=""
    result_dict={}

    vnfi_info = get_vnf_info(prefix, sfc_vnfs)
    vnfi_list = vnfi_info["vnfi_list"]
    vnf_resources = get_vnf_resources(vnfi_list)
    vnf_resources_string = get_vnf_resources_toString(vnf_resources)

    sla_binary_body = "java -cp .:models/sla_binary/h2o-genmodel_sla.jar sla_binary "+vnf_resources_string
    sla_binary_result = subprocess.check_output(sla_binary_body.split()).decode('utf-8').split("\n")[0]

    if sla_binary_result == "0":
        result="Normal"
    else:
        result="Abnormal - "
        sla_rcl_body = "java -cp .:models/sla_rcl/h2o-genmodel_rcl.jar sla_rcl "+vnf_resources_string
        sla_rcl_result = subprocess.check_output(sla_rcl_body.split()).decode('utf-8').split("\n")[0]

        if sla_rcl_result == "none":
            result = result + "traffic overload"
        else:
            result = result + sla_rcl_result
    
    result_dict['detection_result'] = result
    result_dict['time'] = dt.datetime.now()

    return result_dict

def get_resource_overload_detection_result(prefix, sfc_vnfs):

    response=""
    response_dict={}

    vnfi_info = get_vnf_info(prefix, sfc_vnfs)
    vnfi_list = vnfi_info["vnfi_list"]
    vnf_resources = get_vnf_resources(vnfi_list)
    vnf_resources_string = get_vnf_resources_toString(vnf_resources)

    request_body = "java -cp .:models/resource_overload/h2o-genmodel_overload.jar resource_overload "+vnf_resources_string
    result = subprocess.check_output(request_body.split()).decode('utf-8').split("\n")[0]
    
    if result == "none":
        response = "Normal"
        response_dict["detection_result"] = response
    else:
        response = "Abnormal - "+result
        response_dict["detection_result"] = response

    response_dict['time'] = dt.datetime.now()

    return response_dict


def arrange_resource_usage(resource_list):
    
    response_dict = {}
    
    response_dict["cpu"] = resource_list[0]
    response_dict["memory"] = resource_list[1]
    response_dict["disk_read"] = resource_list[2]
    response_dict["disk_write"] = resource_list[3]
    response_dict["rx_bytes"] = resource_list[4]
    response_dict["tx_bytes"] = resource_list[5]

    return response_dict

def get_vnf_resource_usage(prefix, sfc_vnfs):

    response_dict={}

    sfc_vnfs = ["firewall", "flowmonitor", "dpi", "ids", "lb"]
    vnfi_info = get_vnf_info(prefix, sfc_vnfs)
    vnfi_list = vnfi_info["vnfi_list"]
    result = get_vnf_resources(vnfi_list)
        
    response_dict["firewall"] = arrange_resource_usage(result[0].tolist())
    response_dict["flow_monitor"] = arrange_resource_usage(result[1].tolist())
    response_dict["dpi"] = arrange_resource_usage(result[2].tolist())
    response_dict["ids"] = arrange_resource_usage(result[3].tolist())
    response_dict["lb"] = arrange_resource_usage(result[4].tolist())

    return response_dict


def convert_vnf_info(vnfi_list):
    
    response = []

    for i in range(0,len(vnfi_list)):

        instance_dict = {}
        temp = vnfi_list[i].__dict__

        instance_dict['flavor_id'] = temp['_flavor_id']
        instance_dict['id'] = temp['_id']
        instance_dict['node_id'] = temp['_node_id']
        instance_dict['name'] = temp['_name']
        instance_dict['ports'] = convert_network_port_object(temp['_ports'])
        instance_dict['status'] = temp['_status']

        response.append(instance_dict)

    return response


def convert_network_port_object(ports):

    response = []

    for i in range(0,len(ports)):
        port_dict = {}
        temp = ports[i].__dict__
        
        port_dict['ip_addresses'] = temp['_ip_addresses']
        port_dict['network_id'] = temp['_network_id']
        port_dict['port_id'] = temp['_port_id']
        port_dict['port_name'] = temp['_port_name']

        response.append(port_dict)

    return response



