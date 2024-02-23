# VNFInstance


## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** |  | [optional] 
**name** | **str** |  | [optional] 
**status** | **str** | state of VNF VM. (ACTIVE, SHUTOFF, ERROR, etc.) | [optional] 
**flavor_id** | **str** |  | [optional] 
**node_id** | **str** |  | [optional] 
**ports** | [**List[NetworkPort]**](NetworkPort.md) |  | [optional] 

## Example

```python
from openapi_client.models.vnf_instance import VNFInstance

# TODO update the JSON string below
json = "{}"
# create an instance of VNFInstance from a JSON string
vnf_instance_instance = VNFInstance.from_json(json)
# print the JSON string representation of the object
print VNFInstance.to_json()

# convert the object into a dict
vnf_instance_dict = vnf_instance_instance.to_dict()
# create an instance of VNFInstance from a dict
vnf_instance_form_dict = vnf_instance.from_dict(vnf_instance_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


