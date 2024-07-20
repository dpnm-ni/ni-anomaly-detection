from __future__ import print_function
import ni_nfvo_client
import ni_mon_client
from datetime import datetime, timedelta, timezone
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from ni_nfvo_client.rest import ApiException as NfvoApiException
from ni_mon_client.rest import ApiException as NimonApiException
from pprint import pprint
from config import cfg
import torch
import sys
sys.path.insert(1, './cjlee')
import csv
import random
import numpy as np
import datetime as dt
from main_copy import *
from sklearn.model_selection import train_test_split

import csv
import subprocess
import json

import pytz

# get_monitoring_api(): get ni_monitoring_client api to interact with a monitoring module
# Input: null
# Output: monitoring moudle's client api
def get_monitoring_api():

    ni_mon_client_cfg = ni_mon_client.Configuration()
    ni_mon_client_cfg.host = cfg["ni_mon"]["host"]
    ni_mon_api = ni_mon_client.DefaultApi(ni_mon_client.ApiClient(ni_mon_client_cfg))

    return ni_mon_api


# get_nfvo_vnf_spec(): get ni_nfvo_vnf spec to interact with a nfvo module
# Input: null
# Output: nfvo moudle's vnf spec
def get_nfvo_vnf_spec():
#    print("5")

    nfvo_client_cfg = ni_nfvo_client.Configuration()
    nfvo_client_cfg.host = cfg["ni_nfvo"]["host"]
    ni_nfvo_vnf_spec = ni_nfvo_client.VnfSpec(ni_nfvo_client.ApiClient(nfvo_client_cfg))
    ni_nfvo_vnf_spec.flavor_id = cfg["flavor"]["default"]
    ni_nfvo_vnf_spec.user_data = sample_user_data % cfg["instance"]["password"]

    return ni_nfvo_vnf_spec


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
        #end_time = dt.datetime.now(pytz.utc) #now(pytz.utc)
        end_time = dt.datetime.now()
        start_time = end_time - dt.timedelta(seconds = 10)

        if str(end_time)[-1]!='Z':
            end_time = str(end_time.isoformat())+ 'Z'
        if str(start_time)[-1]!='Z':
            start_time = str(start_time.isoformat()) + 'Z'
        #print(end_time)
        for type in resource_type:
            j = resource_type.index(type)
            measurement_type = type
            response = ni_mon_api.get_measurement(vnf_id, measurement_type, start_time, end_time)
            print(vnf_id, measurement_type, start_time, end_time)
            print(response)
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
    load_path = './cjlee/AT-7.pth'
    stat_path = './cjlee/tpi_train.csv.stat'

    V, D = 2, 6

    device = torch.device('cpu')
    model = call_model(load_path=load_path, device=device)
    normalizer = myNormalizer(stat_path)

    vnfi_info = get_vnf_info(prefix, sfc_vnfs)
    vnfi_list = vnfi_info["vnfi_list"]
    vnf_resources = get_vnf_resources(vnfi_list)
    vnf_resources_string = get_vnf_resources_toString(vnf_resources)
    x_data = str2np_array(vnf_resources_string)
    x_data = normalizer(x_data)

    x_data = x_data.reshape(V, D)
    x_data = torch.tensor(x_data).type(torch.float32).to(device)
    x_data = x_data.unsqueeze(0)
    
    output = model(x_data)
    sla_binary_result = str(int(torch.argmax(output.detach())))

    if sla_binary_result == "0":
        result="Normal"
    else:
        result="Abnormal - "

    result_dict['detection_result'] = result
    result_dict['time'] = dt.datetime.now()

    return result_dict

def train_model():
    model = call_model(load_path=load_path, device=device)

def change_date(input_timestamp):
    # change timestamp to rfc3339 format
    # change inputted number to seconds (origninally it was nanoseconds)
    timestamp_in_seconds = int(input_timestamp) / 10**9
    # change seconds to datetime object
    datetime_obj = datetime.utcfromtimestamp(timestamp_in_seconds)
    rfc3339_formatted = datetime_obj.strftime('%Y-%m-%dT%H:%M:%SZ')
    return rfc3339_formatted

def change_memory_to_percent(value, total):
    return (value/total/1000000)*100

def read_csv(file_name):
    data=pd.read_csv(file_name)
    data=data.drop(data[data['time'] == 'time'].index)
    del data['name']
    data['time']=data['time'].apply(change_date)
    data['value']=data['value'].astype(float)
    if file_name.split('_')[3]=='memory':
        if file_name.split('_')[2].endswith('fw'):
            data['value']=data['value'].apply(change_memory_to_percent, total=512)
        else:
            data['value']=data['value'].apply(change_memory_to_percent, total=2048)
    data.rename(columns={'value':file_name.split('_')[3]}, inplace=True)
    return data

def fw_normalize(row):
    fp = open('./cjlee/tpi_train.csv.stat', 'r')
    lines = fp.readlines()
    x_avg= np.asarray([float(s) for s in lines[0].split(',')])
    x_std= np.asarray([float(s) for s in lines[1].split(',')])
    
    for i in range(len(x_std)):
        if x_std[i] == 0:
            x_std[i] = 0.001
    fp.close()
    x_avg=x_avg.reshape(5,6)
    x_std=x_std.reshape(5,6)
    x_avg=x_avg[[0]].flatten()
    x_std=x_std[[0]].flatten()
    return(row - x_avg) / x_std


def dpi_normalize(row):
    fp = open('./cjlee/tpi_train.csv.stat', 'r')
    lines = fp.readlines()
    x_avg= np.asarray([float(s) for s in lines[0].split(',')])
    x_std= np.asarray([float(s) for s in lines[1].split(',')])
    
    for i in range(len(x_std)):
        if x_std[i] == 0:
            x_std[i] = 0.001
    fp.close()
    x_avg=x_avg.reshape(5,6)
    x_std=x_std.reshape(5,6)
    x_avg=x_avg[[2]].flatten()
    x_std=x_std[[2]].flatten()
    return(row - x_avg) / x_std

def read_data(file_path):
    features_list=['cpu', 'memory', 'octetread', 'octetwrite', 'octetrx', 'octettx']

    # read normal data
    vnf1_file_list=['fw_cpu_normal.csv', 'fw_memory_normal.csv', 'fw_octetread_normal.csv', 'fw_octetwrite_normal.csv', 
        'fw_octetrx_normal.csv', 'fw_octettx_normal.csv']
    vnf2_file_list=['dpi_cpu_normal.csv', 'dpi_memory_normal.csv', 'dpi_octetread_normal.csv', 'dpi_octetwrite_normal.csv', 
        'dpi_octetrx_normal.csv', 'dpi_octettx_normal.csv']
    vnf1_data = read_csv(file_path+'/normal/'+vnf1_file_list[0])
    for file_name in vnf1_file_list[1:]:
        data=read_csv(file_path+'/normal/'+file_name)
        vnf1_data=pd.merge(vnf1_data,data,on='time',how='inner')
    vnf1_data['abnormal']=0
    vnf2_data = read_csv(file_path+'/normal/'+vnf2_file_list[0])
    for file_name in vnf2_file_list[1:]:
        data=read_csv(file_path+'/normal/'+file_name)
        vnf2_data=pd.merge(vnf2_data,data,on='time',how='inner')
    vnf2_data['abnormal']=0
    # read abnormal data
    vnf1_file_list=['fw_cpu_abnormal.csv', 'fw_memory_abnormal.csv', 'fw_octetread_abnormal.csv', 'fw_octetwrite_abnormal.csv', 
        'fw_octetrx_abnormal.csv', 'fw_octettx_abnormal.csv']
    vnf2_file_list=['dpi_cpu_abnormal.csv', 'dpi_memory_abnormal.csv', 'dpi_octetread_abnormal.csv', 'dpi_octetwrite_abnormal.csv', 
        'dpi_octetrx_abnormal.csv', 'dpi_octettx_abnormal.csv']
    vnf1_abnormal_data = read_csv(file_path+'/abnormal/'+vnf1_file_list[0])
    for file_name in vnf1_file_list[1:]:
        data=read_csv(file_path+'/abnormal/'+file_name)
        vnf1_abnormal_data=pd.merge(vnf1_abnormal_data,data,on='time',how='inner')
    vnf1_abnormal_data['abnormal']=1
    vnf1_data=pd.concat([vnf1_data,vnf1_abnormal_data], axis=0)
    vnf2_abnormal_data = read_csv(file_path+'/abnormal/'+vnf2_file_list[0])
    for file_name in vnf2_file_list[1:]:
        data=read_csv(file_path+'/abnormal/'+file_name)
        vnf2_abnormal_data=pd.merge(vnf2_data,data,on='time',how='inner')
    vnf2_abnormal_data['abnormal']=1
    vnf2_data=pd.concat([vnf2_data,vnf2_abnormal_data], axis=0)

    # Read end
    print(f'vnf1_data.shape: {vnf1_data.shape}')
    # vnf1_data.shape: (188946, 8)
    # Normalize
    vnf1_data[features_list]=vnf1_data[features_list].apply(fw_normalize, axis=1)
    vnf2_data[features_list]=vnf2_data[features_list].apply(dpi_normalize, axis=1)
    data=pd.concat([vnf1_data,vnf2_data], axis=0)
    print(data['abnormal'].value_counts())
    # Seperate data
    x_data=data[features_list].values
    y_data=data['abnormal'].values
    return x_data, y_data

def read_from_all_csv(file_path):
    data = pd.read_csv(file_path)
    x_data= data.iloc[:,:-1].values
    y_data= data.iloc[:,-1].values
    return x_data, y_data 

def get_ad_f1score():
    # read data
    #file_path='/home/dpnm/tmp/ni_ad_data'
    #x_data, y_data = read_data(file_path)
    #x_data=x_data[:x_data.shape[0]]
    #y_data=y_data[:y_data.shape[0]]
    file_path='ad_all_data.csv'
    x_data, y_data = read_from_all_csv(file_path)
    print (x_data.shape, y_data.shape)
    result_dict={}
    load_path = './cjlee/AT-7.pth'
    stat_path = './cjlee/tpi_train.csv.stat'
    if torch.cuda.is_available() and False:
        print('run on gpu')
        device = torch.device('cuda')
        model = new_model(device=device)
        #model = call_model(load_path=load_path, device=device)
        model.cuda()
    else:
        print('run on cpu')
        device = torch.device('cpu')
        model = new_model(device=device)
        #model = call_model(load_path=load_path, device=device)
    #x_data = x_data.unsqueeze(1)
    x_data=np.expand_dims(x_data,axis=1)
    train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, test_size=0.3)
    data_train=np.c_[train_x.squeeze(1),train_y]
    data_test=np.c_[test_x.squeeze(1),test_y]

    #x_data = torch.tensor(x_data).type(torch.float32).to(device)    
    test_x=torch.tensor(test_x).type(torch.float32).to(device)
    train_x=torch.tensor(train_x).type(torch.float32).to(device)
    print(test_x.shape)
    output = model(test_x)
    #int(torch.argmax(output.detach())))
    #print(output.shape)
    output=torch.argmax(output.detach(),dim=1).to(torch.device('cpu')).numpy()
    print(output[:30])
    print(test_y[:30])
    f1=f1_score(test_y, output)
    print(f1)
    result_dict['untrained_f1_score'] = str(f1)
    result_dict['untrained_accuracy'] = str(accuracy_score(test_y, output))
    result_dict['untrained_precision'] = str(precision_score(test_y, output))
    result_dict['untrained_recall'] = str(recall_score(test_y, output))

    # train model
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(30):
        optimizer.zero_grad()
        output = model(train_x)
        loss = loss_fn(output, torch.tensor(train_y).long().to(device))
        loss.backward()
        optimizer.step()
        if epoch%10==0:
            print(f'epoch: {epoch}, loss: {loss.item()}')
            model.eval()
            output = model(test_x)
            output=torch.argmax(output.detach(),dim=1).to(torch.device('cpu')).numpy()
            f1=f1_score(test_y, output)
            print(f'test f1 : {f1}')
            model.train()
    model.eval()
    output = model(test_x)
    output=torch.argmax(output.detach(),dim=1).to(torch.device('cpu')).numpy()
    f1=f1_score(test_y, output)
    print(f1)
    result_dict['trained_f1_score'] = str(f1)  
    result_dict['trained_accuracy'] = str(accuracy_score(test_y, output))
    result_dict['trained_precision'] = str(precision_score(test_y, output))
    result_dict['trained_recall'] = str(recall_score(test_y, output))
    data_test=np.c_[data_test,output]
    #np.savetxt('/home/dpnm/data/ad_data_train.csv',data_train,delimiter=',')
    #np.savetxt('/home/dpnm/data/ad_data_test.csv',data_test,delimiter=',')
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

    #sfc_vnfs = ["firewall", "flowmonitor", "dpi", "ids", "lb"]
    sfc_vnfs = ["firewall", "dpi"]
    vnfi_info = get_vnf_info(prefix, sfc_vnfs)
    vnfi_list = vnfi_info["vnfi_list"]
    result = get_vnf_resources(vnfi_list)
        
    response_dict["firewall"] = arrange_resource_usage(result[0].tolist())
    #response_dict["flow_monitor"] = arrange_resource_usage(result[1].tolist())
    response_dict["dpi"] = arrange_resource_usage(result[1].tolist())
    #response_dict["ids"] = arrange_resource_usage(result[3].tolist())
    #response_dict["lb"] = arrange_resource_usage(result[4].tolist())

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


# create_monitor_classifier(scaler): create flow classifier of traffic generator in the testbed
# Input: scaler
# Output: response
def create_monitor_classifier(scaler):
#    print("23")
    ni_nfvo_sfcr_api = get_nfvo_sfcr_api()
    name = scaler.get_scaling_name() + cfg["instance"]["prefix_splitter"] + "monitor"
    source_client = scaler.get_monitor_src_id()
    src_ip_prefix = get_ip_from_id(source_client) + "/32"
    dst_ip_prefix = get_ip_from_id(scaler.get_monitor_dst_id()) + "/32"
    sfcr_id = get_sfc_by_name(scaler.get_sfc_name()).sfcr_ids[-1] # HDN : It returns the last SFCR of the SFC, but usually, a SFC corresponds to one SFCR, so it will return the original SFCR's id.
    nf_chain = get_sfcr_by_id(sfcr_id).nf_chain

    sfcr_spec = ni_nfvo_client.SfcrSpec(name=name,
                                 src_ip_prefix=src_ip_prefix,
                                 dst_ip_prefix=dst_ip_prefix,
                                 nf_chain=nf_chain,
                                 source_client=source_client)

    api_response = ni_nfvo_sfcr_api.add_sfcr(sfcr_spec)

    return api_response # HDN : it returns the id of the generated SFCR


def create_sfc(scaler):
    def my_create_monitor(source_client_name):
        vnf_spec = get_nfvo_vnf_spec() # Get empty default VNF deployment spec (same config that we use on Openstack dashboard)
        vnf_spec.image_id = cfg["image"]["sla_monitor"]
        source_client = [ inst for inst in get_monitoring_api().get_vnf_instances() \
                                if inst.name == (my_sfc.sfc_prefix + source_client_name) ][0]
        target_node = source_client.node_id # HDN : target_node is not destination, but it is source node. I guess it is because the traversed traffic should be return at the end after flowing through the defined SFC path. 

        monitor_vnf_name = my_sfc.sfc_prefix + 'scaling' + \
                                    cfg["instance"]["prefix_splitter"] + "monitor-dst"
        monitor_vnf = [ inst for inst in get_monitoring_api().get_vnf_instances() \
                            if inst.name == monitor_vnf_name ]
        monitor_vnf_check = 1 if len(monitor_vnf) > 0 and check_active_instance(monitor_vnf[0].id) \
                                else 0

        if monitor_vnf_check == 1:
            print("----- Deployment of monitoring VNF is skipped -----")
            scaler.set_monitor_src_id(cfg["sla_monitoring"]["id"])
            scaler.set_monitor_dst_id(monitor_vnf[0].id)
            return True, monitor_vnf[0].id

        # Repeat to try creating SLA monitors if fails
        for k in range(0, 5):

            # If enough resourcecs in a target node, create monitor
            if check_available_resource(target_node):
                vnf_spec.vnf_name = my_sfc.sfc_prefix + 'scaling' + \
                                     cfg["instance"]["prefix_splitter"] + "monitor-dst"

                vnf_spec.node_name = target_node
                dst_id = deploy_vnf(vnf_spec)

                # Wait 1 minute untill creating SLA monitors
                for i in range (0, 30):
                    time.sleep(2)

                    # Success to create SLA monitors
                    if check_active_instance(dst_id):
                        scaler.set_monitor_src_id(cfg["sla_monitoring"]["id"])
                        scaler.set_monitor_dst_id(dst_id)
                        return True, dst_id

            # If not enough resources in a target node, select another one randomly
            else:
                target_node = random.choice(get_node_info()).id

        # Fail to create SLA monitors
        destory_vnf(dst_id)

        return False, None

    def make_random_SFCR(num_destinations):
        sfc_type_num = random.choice(list(my_sfc.sfc_spec.keys()))
        sfc_dest_num = random.randint(1, num_destinations)
        sfc_vnfs = ['src'] + my_sfc.sfc_spec[sfc_type_num]['vnf_chain'] + \
                   ['dst'+str(sfc_dest_num)]
        sfc_info = {
                    'number_of_vnfs': None,\
                    'sfc_name': my_sfc.sfc_prefix+'sfc',\
                    'sfc_prefix': my_sfc.sfc_prefix,\
                    'sfc_type_num': sfc_type_num,\
                    'sfc_vnfs': sfc_vnfs,\
                    'sfcr_name': my_sfc.sfc_prefix+'sfcr',\
                    'status': False,\
                    'sfcr_id': None,\
                    'sfc_id': None,\
                    'monitor_sfcr_id': None }
        return sfc_info


    my_sfc.sfc_prefix = 'and-'
    source_client_name = 'src' ##
    node_mask_name = ['ni-compute-181-154', 'ni-compute-181-155', 'ni-compute-181-156', 'ni-compute-181-162', 'ni-compute-181-203', 'Switch-edge-01', 'Switch-edge-02', 'Switch-edge-03', 'Switch-core-01', 'ni-compute-kisti']

    print("(0) Creating monitoring VNF")
    create_flag, monitor_dst_id = my_create_monitor(source_client_name)
    print(create_flag)
    if create_flag:
        print("Successful to create monitor ID : {}".format(monitor_dst_id))
    else:
        raise SyntaxError("Failed to create monitor VNF.. Check resources of testbed")

    #print("SFCR generation..", end=" ")
    #sfcr_id = set_flow_classifier(sfc_info, source_client_name)
    #sfc_info['sfcr_id'] = sfcr_id
    #print("Successful ID:{}, VNF-Chain:{}".format(sfcr_id, sfc_info['sfc_vnfs']))
    
    sfc_hypo = handong_sfc_hypo(sfc_info)
