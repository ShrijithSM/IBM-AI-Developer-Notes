#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from typing import Optional

from ibm_watson_machine_learning.wml_client_error import WMLClientError
from ibm_watson_machine_learning.messages.messages import Messages


class CPDVersion:
    """Storage for cpd version. Comparison operators are 
    overloaded to allow comparison with numeric values. 

    Class attribute:
    :param supported_version_list: List of supported CPD versions.
    :type supported_version_list: list

    Attribute:
    :param cpd_version: Store CPD version.
    :type cpd_version: Optional[str], optional

    .. code-block:: python

        from ibm_watson_machine_learning.utils import CPDVersion

        version = CPDVersion()

        if not version:
            print("CPD version is None")

        version.cpd_version = '4.5'

        if version > 4:
            print("version greater than 4.0")

    """
    supported_version_list = ['4.0', '4.5', '4.6', '4.7', '4.8', '5.0']

    def __init__(self, version: Optional[str] = None):
        self.cpd_version = version

    def __str__(self):
        version = self.__cpd_version
        return f"CPD version {version}" if version is not None else ""

    @property
    def cpd_version(self):
        """Attribute that stores cpd version. Before the value is set,
          validation is performed against a supported versions.
        """
        return self.__cpd_version

    @cpd_version.setter
    def cpd_version(self, value):
        if value is None:
            self.__cpd_version = value
        elif str(value) in CPDVersion.supported_version_list:
            self.__cpd_version = str(value)
        else:
            raise WMLClientError(Messages.get_message(', '.join(CPDVersion.supported_version_list),
                                                      message_id="invalid_version"))

    @cpd_version.getter
    def cpd_version(self):
        value = self.__cpd_version
        if value is None:
            return None
        else:
            dot_index = value.find(".")
            value = value[:dot_index + 1] + value[dot_index + 1:].replace(".", "")
            return float(value)

    def __bool__(self):
        return bool(self.cpd_version)

    def __eq__(self, value):
        return self.cpd_version == value

    def __ne__(self, value):
        return not self.__eq__(value)

    def __lt__(self, value):
        return bool(self) and self.cpd_version.__lt__(value)

    def __le__(self, value):
        return bool(self) and (self.__lt__(value) or self.__eq__(value))

    def __gt__(self, value):
        return bool(self) and not self.__le__(value)

    def __ge__(self, value):
        return bool(self) and not self.__lt__(value)
