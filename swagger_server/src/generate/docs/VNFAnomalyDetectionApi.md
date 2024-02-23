# openapi_client.VNFAnomalyDetectionApi

All URIs are relative to *http://localhost*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_sla_detection_result**](VNFAnomalyDetectionApi.md#get_sla_detection_result) | **GET** /get_sla_detection_result/{prefix} | Get VNF Anomaly Detection Model&#39;s Prediction Results - SLA Violations
[**get_vnf_info**](VNFAnomalyDetectionApi.md#get_vnf_info) | **GET** /get_vnf_info/{prefix} | Get VNFs&#39; basic information that configure SFC
[**get_vnf_resource_usage**](VNFAnomalyDetectionApi.md#get_vnf_resource_usage) | **GET** /get_vnf_resource_usage/{prefix} | Get VNFs&#39; Resource Usages in Real-Time


# **get_sla_detection_result**
> str get_sla_detection_result(prefix)

Get VNF Anomaly Detection Model's Prediction Results - SLA Violations

### Example

```python
import time
import os
import openapi_client
from openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.VNFAnomalyDetectionApi(api_client)
    prefix = 'prefix_example' # str | VNF instance name prefix

    try:
        # Get VNF Anomaly Detection Model's Prediction Results - SLA Violations
        api_response = api_instance.get_sla_detection_result(prefix)
        print("The response of VNFAnomalyDetectionApi->get_sla_detection_result:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling VNFAnomalyDetectionApi->get_sla_detection_result: %s\n" % e)
```



### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **prefix** | **str**| VNF instance name prefix | 

### Return type

**str**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | OK |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_vnf_info**
> List[VNFInstance] get_vnf_info(prefix)

Get VNFs' basic information that configure SFC

### Example

```python
import time
import os
import openapi_client
from openapi_client.models.vnf_instance import VNFInstance
from openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.VNFAnomalyDetectionApi(api_client)
    prefix = 'prefix_example' # str | vnf instance name prefix

    try:
        # Get VNFs' basic information that configure SFC
        api_response = api_instance.get_vnf_info(prefix)
        print("The response of VNFAnomalyDetectionApi->get_vnf_info:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling VNFAnomalyDetectionApi->get_vnf_info: %s\n" % e)
```



### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **prefix** | **str**| vnf instance name prefix | 

### Return type

[**List[VNFInstance]**](VNFInstance.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | OK |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_vnf_resource_usage**
> str get_vnf_resource_usage(prefix)

Get VNFs' Resource Usages in Real-Time

### Example

```python
import time
import os
import openapi_client
from openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.VNFAnomalyDetectionApi(api_client)
    prefix = 'prefix_example' # str | VNF instance name prefix

    try:
        # Get VNFs' Resource Usages in Real-Time
        api_response = api_instance.get_vnf_resource_usage(prefix)
        print("The response of VNFAnomalyDetectionApi->get_vnf_resource_usage:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling VNFAnomalyDetectionApi->get_vnf_resource_usage: %s\n" % e)
```



### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **prefix** | **str**| VNF instance name prefix | 

### Return type

**str**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | OK |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

