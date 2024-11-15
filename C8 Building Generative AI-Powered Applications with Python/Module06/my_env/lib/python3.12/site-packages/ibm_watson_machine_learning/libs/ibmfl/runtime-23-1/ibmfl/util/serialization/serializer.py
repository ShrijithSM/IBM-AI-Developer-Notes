#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from abc import ABC, abstractmethod

import logging


logger = logging.getLogger(__name__)

class Serializer(ABC):
    """
    Abstract class for Serializer
    """

    @abstractmethod
    def serialize(self):
        """
        Serialize 

        :return: serialized byte stream
        :rtype: `b[]`
        """
        pass

    @abstractmethod
    def deserialize(self, serialization):
        """
        Deserialize 

        :param serialization: Serialization of the object
        :return: deserialized object
        """
        pass
