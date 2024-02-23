import time
import openapi_client
from openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "http://localhost:8005"
)



# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.VNFAnomalyDetectionApi(api_client)
    prefix = 'hg-0-' # str | VNF instance name prefix

    try:
        # Get VNF Anomaly Detection Model's Prediction Results - SLA Violations
        api_response = api_instance.get_sla_detection_result(prefix)
        print("The response of VNFAnomalyDetectionApi->get_sla_detection_result:\n")
        pprint(api_response)
    except ApiException as e:
        print("Exception when calling VNFAnomalyDetectionApi->get_sla_detection_result: %s\n" % e)