#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from enum import Enum


class RShinyAuthenticationValues(Enum):
    """Allowable values of R_Shiny authentication."""
    ANYONE_WITH_URL = "anyone_with_url"
    ANY_VALID_USER = "any_valid_user"
    MEMBERS_OF_DEPLOYMENT_SPACE = "members_of_deployment_space"
