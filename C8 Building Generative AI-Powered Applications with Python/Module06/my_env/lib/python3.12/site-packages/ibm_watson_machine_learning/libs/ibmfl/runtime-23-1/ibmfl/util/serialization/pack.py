#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------


from ibmfl.exceptions import SerializationException
import logging

import msgpack
import msgpack_numpy as m
import numpy as np


from ibmfl.util.serialization.serializers import SketchSerializer, PredictorSerializer, DTypeSerializer
from ibmfl.util.serialization.serializers import Serializers

from sklearn.ensemble._hist_gradient_boosting.predictor import TreePredictor

from ddsketch.ddsketch import DDSketch

logger = logging.getLogger(__name__)


def encode(obj):
    if isinstance(obj, DDSketch):
        return msgpack.ExtType(Serializers.Sketch.value,SketchSerializer.serialize(obj))
    elif isinstance(obj, TreePredictor):
        return msgpack.ExtType(Serializers.Predictor.value,PredictorSerializer.serialize(obj))
    elif isinstance(obj, np.dtype):
        return msgpack.ExtType(Serializers.DType.value, DTypeSerializer.serialize(obj))
    else :
        return m.encode(obj)
    

def decode(obj, chain=None):
    return m.decode(obj)

def ext_hook(code, data):
    if code == Serializers.Sketch.value:
        return SketchSerializer.deserialize(data)
    elif code == Serializers.Predictor.value:
        return PredictorSerializer.deserialize(data)
    elif code == Serializers.DType.value:
        return DTypeSerializer.deserialize(data)
    return msgpack.ExtType(code.data)

def pack(x):
    try:
        return msgpack.packb(x, default=encode)
    except TypeError :
        logger.error("Failed to serialize object",exc_info=True)
        raise SerializationException("Unable to serialize object")

def unpack(x_enc):
    try:
        return msgpack.unpackb(x_enc, object_hook=decode, ext_hook=ext_hook, strict_map_key=False)
    except TypeError :
        logger.error("Failed to serialize {}".format(x_enc))
        raise SerializationException("Unable to deserialize object")
        


