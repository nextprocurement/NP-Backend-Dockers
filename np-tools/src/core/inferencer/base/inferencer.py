"""
This module provides the class Inferencer, which consists of a wrapper for perfoming inference on a new unseen corpus. It contains specific implementations according to the trainer used for the generation of the topic model that is being used for inference. It is based in the inference process of the NP-Search-Tools project.

Author: Lorena Calvo-Bartolomé
Date: 18/03/2024
"""

import json
import logging
import os
import pathlib
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from src.core.inferencer.base.TopicModeling.BaseModel import BaseModel
from src.core.inferencer.base.utils import sum_up_to


class Inferencer(object):
    """
    Wrapper for a NP-Search-Tools Topic Models Inferencers
    Assumes input is given by a inferConfigFile with the following format:
    infer_config = {
        "description": Description of the inference,
        "infer_path": Path where the inference will be saved,
        "model_for_infer_path": Path to the model to be used for inference,
        "trainer": Model trainer that will be used for inference,
        "TrDtSet": Path to the dataset used for training,
        "text_to_infer": Text to infer,
        "Preproc": Preprocessing object,
        "TMparam": Parameters of the topic model,
        "creation_date": Creation date of the inference,
        }
    """

    def __init__(
        self,
        logger: logging.Logger
    ) -> None:
        """
        Initilization Method

        Parameters
        ----------
        logger: Logger object
            To log object activity
        """

        if logger:
            self._logger = logger
        else:
            import logging
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('Inferencer')

        return

    def transform_inference_output(self,
                                   thetas32: np.array,
                                   max_sum: int) -> List[dict]:
        """Saves the topic distribution for each document in text format (tXX|weightXX)

        Parameters
        ----------
        thetas32: np.ndarray
            Doc-topic distribution of the inferred documents

        Returns
        -------
        List[dict]: List of dictionaries with the topic distribution for each document
        """

        self._logger.info(
            '-- Inference: Saving the topic distribution in text format')

        def get_doc_str_rpr(vector: np.array, max_sum: int) -> str:
            """Calculates the string representation of a document's topic proportions in the format 't0|100 t1|200 ...', so that the sum of the topic proportions is at most max_sum.

            Parameters
            ----------
            vector: numpy.array
                Array with the topic proportions of a document.
            max_sum: int
                Maximum sum of the topic proportions.

            Returns 
            -------
            rpr: str
                String representation of the document's topic proportions.
            """
            vector = sum_up_to(vector, max_sum)
            rpr = ""
            for idx, val in enumerate(vector):
                if val != 0:
                    rpr += "t" + str(idx) + "|" + str(val) + " "
            rpr = rpr.rstrip()
            return rpr

        if thetas32.ndim == 2:
            doc_tpc_rpr = [get_doc_str_rpr(thetas32[row, :], max_sum)
                           for row in range(len(thetas32))]
        elif thetas32.ndim == 1:
            doc_tpc_rpr = [get_doc_str_rpr(thetas32, max_sum)]
        else:
            self._logger.error(
                f"-- -- Thetas32 has wrong number of dimensions when transforming inference output")
        ids = np.arange(len(thetas32))
        df = pd.DataFrame(list(zip(ids, doc_tpc_rpr)),
                          columns=['id', 'thetas'])

        infer_path = Path(self._inferConfig["infer_path"])
        doc_topics_file_csv = infer_path.joinpath("doc-topics.csv")
        df.to_csv(doc_topics_file_csv, index=False)

        return df.to_dict(orient='records')

    def predict(
        self,
        inferConfigFile: pathlib.Path,
        texts: List[str],
        max_sum: int = 1000,
        thetas_thr: float = 3e-3
    ):

        with pathlib.Path(inferConfigFile).open('r', encoding='utf8') as fin:
            self._inferConfig = json.load(fin)

        # Check if the model to perform inference on exists
        model_for_inf = Path(
            self._inferConfig["model_for_infer_path"])
        if not os.path.isdir(model_for_inf):
            self._logger.error(
                f'-- -- Provided path for the model to perform inference on path is not valid -- Stop')
            return

        # Load model
        model = BaseModel.load_model(
            path=model_for_inf.as_posix()
        )

        infer_thetas = model.predict(texts)
        thetas32_rpr = super().get_final_thetas(
            thetas32=infer_thetas,
            thetas_thr=thetas_thr,
            max_sum=max_sum)

        return thetas32_rpr

    def get_final_thetas(
        self,
        thetas32: np.ndarray,
        thetas_thr: float,
        max_sum: int
    ) -> List[dict]:
        """
        Given the inferred document-topic proportions, it returns the final thetas in the desired format

        Parameters
        ----------
        thetas32: np.ndarray
            Doc-topic distribution of the inferred documents
        thetas_thr: float
            Threshold for the inferred document-topic proportions (it should be the same used during training)
        max_sum: int
            Maximum sum of the topic proportions when attaining their string representation

        Returns
        -------
        List[dict]
            List of dictionaries with the inferred topics in string representation
        """

        # Thresholding and normalization
        thetas32[thetas32 < thetas_thr] = 0
        if thetas32.ndim == 2:
            thetas32 = normalize(thetas32, axis=1, norm='l1')
        elif thetas32.ndim == 1:
            thetas32 = normalize(thetas32.reshape(1, -1), axis=1, norm='l1')

        # Transform thetas into string representation
        thetas32_rpr = self.transform_inference_output(thetas32, max_sum)

        return thetas32_rpr
