# NetworkPort


## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**port_id** | **str** |  | [optional] 
**port_name** | **str** |  | [optional] 
**network_id** | **str** |  | [optional] 
**ip_addresses** | **List[str]** |  | [optional] 

## Example

```python
from openapi_client.models.network_port import NetworkPort

# TODO update the JSON string below
json = "{}"
# create an instance of NetworkPort from a JSON string
network_port_instance = NetworkPort.from_json(json)
# print the JSON string representation of the object
print NetworkPort.to_json()

# convert the object into a dict
network_port_dict = network_port_instance.to_dict()
# create an instance of NetworkPort from a dict
network_port_form_dict = network_port.from_dict(network_port_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


