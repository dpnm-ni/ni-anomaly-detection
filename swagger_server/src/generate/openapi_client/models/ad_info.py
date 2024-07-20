# coding: utf-8

"""
    NI Project Anomaly Detection Module

    NI VNF Anomaly Detection Module for the NI Project.

    The version of the OpenAPI document: 1.0.0
    Generated by OpenAPI Generator (https://openapi-generator.tech)

    Do not edit the class manually.
"""  # noqa: E501


from __future__ import annotations
import pprint
import re  # noqa: F401
import json


from typing import Optional
from pydantic import BaseModel, StrictStr

class AdInfo(BaseModel):
    """
    AdInfo
    """
    vnf_prefix: Optional[StrictStr] = None
    __properties = ["vnf_prefix"]

    class Config:
        """Pydantic configuration"""
        allow_population_by_field_name = True
        validate_assignment = True

    def to_str(self) -> str:
        """Returns the string representation of the model using alias"""
        return pprint.pformat(self.dict(by_alias=True))

    def to_json(self) -> str:
        """Returns the JSON representation of the model using alias"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> AdInfo:
        """Create an instance of AdInfo from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> AdInfo:
        """Create an instance of AdInfo from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return AdInfo.parse_obj(obj)

        _obj = AdInfo.parse_obj({
            "vnf_prefix": obj.get("vnf_prefix")
        })
        return _obj

