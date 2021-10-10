# ni-anomaly-detection-module
ni-anomaly-detection-module detects resource overload and SLA(SLO) violations by monitoring VNFs' system resource metrics running from the OpenStack testbed. (You can access the [Technical Document](https://github.com/dpnm-ni/ni-anomaly-detection-public/blob/master/Technical_Document_Korean.pdf) for Korean description in this repository)

## Main Responsibilities
Supervised Learning-based VNF anomaly detection module.
- Provide APIs to detect VNFs' resource overload states in real-time ()
- Provide APIs to detect SLA(SLO) violations in real-time
- Provide APIs to show VNFs' basic information consist of SFC
- Provide APIs to show VNFs' system resource usage in real-time

## Requirements
```
Ubuntu 16.04, 18.04
Python 3.5.2+
Java 11+
```

Please install pip3 and requirements by using the command as below.
```
sudo apt-get update
sudo apt-get install python3-pip
pip3 install -r requirements.txt
```

## Configuration
This module runs as web server to handle requests that detects abnormal status of VNFs that configures specific service (web service). The SFC consists of 5 VNFs: firewall, flow monitor, DPI (Deep Packet Inspection), IDS (Intrusion Detection System), load balancer.

To use a web UI of this module or send an SFC request to the module, a port number can be configured (a default port number is 8005)

```
# server/__main__.py

def main():
    app = connexion.App(__name__, specification_dir='./swagger/')
    app.app.json_encoder = encoder.JSONEncoder
    app.add_api('swagger.yaml', arguments={'title': 'NI Project Anomaly Detection Module'})
    app.run(port=8005)
```

This module interacts with ni-mano to create SFC in OpenStack environment.
To communicate with ni-mano, this module should know URI of ni-mano.
In ni-mano, ni_mon and ni_nfvo are responsible for interacting with this module so their URI should be configured as follows.

```
# config/config.yaml

ni_mon:
  host: http://<ni_mon_ip>:<ni_mon_port>      # Configure here to interact with a monitoring module
ni_nfvo:
  host: http://<ni_nfvo_ip>:<ni_nfvo_port>    # Configure here to interact with an NFVO module
```


## Usage

After installation and configuration of this module, you can run this module by using the command as follows.

```
sudo python3 -m swagger_server
```

This module provides web UI based on Swagger:

```
http://<host IP running this module>:<port number>/ui/
```

To detect the VNFs' real-time status in OpenStack testbed, this module processes a HTTP GET message including in its body.
You can generate an request by using web UI or using other library creating HTTP messages.

Required data to create HTTP request is VNF instances' prefix.
(We assume that the prefix of the VNF instances' name that consists of the SFC is the same.)

- **prefix**: a prefix to identify instances which can be components of an SFC from OpenStack


The format of response for each function is as follows.


- **GET /get_vnf_info/{prefix}**

Array of VNF instance info

```
	{
    	"flavor_id": "flavor_id",
    	"id": "id",
    	"name": "name",
    	"node_id": "node_id",
    	"ports": [
      		{
	        	"ip_addresses": [
          			"ip_addresses",
          			"ip_addresses"
        		],
        		"network_id": "network_id",
        		"port_id": "port_id",
        		"port_name": "port_name"
      		},
      		{
        		"ip_addresses": [
          			"ip_addresses",
          			"ip_addresses"
        		],
        		"network_id": "network_id",
        		"port_id": "port_id",
        		"port_name": "port_name"
      		}
    	],
    	"status": "status"
  	}
```

- **GET /get_vnf_resource_usage/{prefix}**

Array of Resource metrics

```
    {
		"dpi": {
    		"cpu": "cpu_usage",
    		"memory": "mem_usage",
	    	"disk_read": "disk_read_bytes",
    		"disk_write": "disk_write_bytes",
    		"rx_bytes": "network_interfaces' rx_bytes",
    		"tx_bytes": "network_interfaces' tx_bytes",
  		}
    }
```

- **GET /get_resource_overload_detection_result/{prefix}**

Detection result, time for sending request

```
    {
      “detection_result”: "detection_result",
      "time": “time”,
    }
```

- **GET /get_sla_detection_result/{prefix}**

Detection result, time for sending request

```
    {
      “detection_result”: "detection_result",
      "time": “time”,
    }
```

## Release information
* Release 1 - Detecting the VNFs' resource overload and SLO violations in fixed SFC.
* Release 2 - Detecting the VNFs' resource overload and SLO violations in dynamic SFC. (TBA)

Note that
* The OpenStack testbed is implemented by [NI Project](https://github.com/dpnm-ni)
