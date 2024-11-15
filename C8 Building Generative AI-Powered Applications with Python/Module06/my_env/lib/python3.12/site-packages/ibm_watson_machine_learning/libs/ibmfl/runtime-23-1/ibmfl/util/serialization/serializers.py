#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------


import logging

from ibmfl.util.serialization.serializer import Serializer

import msgpack

from sklearn.ensemble._hist_gradient_boosting.predictor import TreePredictor

from ddsketch.ddsketch import DDSketch
from ddsketch.pb.proto import DDSketchProto
from google.protobuf.json_format import MessageToJson,Parse

from skops.io import dumps, loads

import numpy as np

from enum import Enum

logger = logging.getLogger(__name__)

class Serializers(Enum):
    Sketch = 1
    Predictor = 2
    DType = 3



#
# DDSketch Serializer
#
class SketchSerializer(Serializer):

    @staticmethod
    def serialize(x):
        """
        Serialize 
        """
        proto = DDSketchProto.to_proto(x)
        json = MessageToJson(proto,preserving_proto_field_name=True)
        return msgpack.packb(json)

    @staticmethod
    def deserialize(serialization):
        """
        Deserialize 

        :param serialization: Serialization of the object
        :return: deserialized object
        """
        unpacked = msgpack.unpackb(serialization)
        proto = Parse(unpacked, DDSketchProto.to_proto(DDSketch()))
        sketch = DDSketchProto.from_proto(proto)
        return sketch
        
#
# TreePredictor Serializer
#
class PredictorSerializer(Serializer):

    @staticmethod
    def serialize(x):
        """
        Serialize 
        """
        
        return dumps(x)

    @staticmethod
    def deserialize(serialization):
        """
        Deserialize 

        :param serialization: Serialization of the object
        :return: deserialized object
        """
        t = loads(serialization, trusted = ['sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor'])

        return t
        

#
# Numpy DType Serializer
#
class DTypeSerializer(Serializer):

    @staticmethod
    def serialize(x):
        """
        Serialize 
        """
        
        # Take typestr
        return bytes(x.descr[0][1],'utf-8')

    @staticmethod
    def deserialize(serialization):
        """
        Deserialize 

        :param serialization: Serialization of the object
        :return: deserialized object
        """

        # typestr to dtype
        t = np.dtype(str(serialization,'utf-8'))

        return t
        
