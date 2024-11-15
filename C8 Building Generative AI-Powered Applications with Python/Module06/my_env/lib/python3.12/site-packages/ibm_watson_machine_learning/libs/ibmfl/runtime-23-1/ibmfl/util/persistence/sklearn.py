#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import logging

import joblib
import time

from ibmfl.exceptions import FLException
logger = logging.getLogger(__name__)


class SKLearnPersistence():

    @staticmethod
    def load_model(full_path):
        try:
           model = joblib.load(full_path)
           return model
        except Exception as e:
           logger.exception(e)
           raise FLException("Unable to load model.")

    @staticmethod
    def model_filename(file):
        return "{}_{}.pickle".format(file, time.time())
    
    @staticmethod
    def save_model(model, full_path):
        with open(full_path, "wb") as f:
            joblib.dump(model, f)
        logger.info("Model saved in path: %s.", full_path)
