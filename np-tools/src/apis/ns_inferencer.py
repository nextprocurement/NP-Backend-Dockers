"""
This script defines a Flask RESTful namespace for performing inference.

Author: Lorena Calvo-Bartolomé
Date: 19/03/2024
"""

import os
import pathlib
import time
from flask_restx import Namespace, Resource, reqparse

from src.core.inferencer import Inferencer

import logging
logging.basicConfig(level='DEBUG')
logger = logging.getLogger('Inferencer')

# ======================================================
# Define namespace for inference operations
# ======================================================
api = Namespace('Inference operations')

# ======================================================
# Create Inferencer object
# ======================================================
inferencer = Inferencer(logger=logger)

# ======================================================
# Define parsers to take inputs from user
# ======================================================
infer_doc_parser = reqparse.RequestParser()
infer_doc_parser.add_argument('text_to_infer',
                              help='Text to be inferred',
                              required=True)
infer_doc_parser.add_argument('model_for_infer',
                              help='Model to be used for the inference', required=True)


@api.route('/inferDoc/')
class InferDoc(Resource):
    """Given a text and a model (trained toppic model), this endpoint returns the inference of the text using the model.
    """
    @api.doc(
        parser=infer_doc_parser,
        responses={
            200: 'Success: Inference generated successfully',
            501: 'Model for inference not found: An error occurred while trying to find the model for inference',
            502: 'Inference generation error: An error occurred while generating the inference'
        })
    def get(self):

        start_time = time.time()

        args = infer_doc_parser.parse_args()

        text_to_infer = args['text_to_infer']
        
        # We look for the model in case the user did not write the name properly
        look_dir = pathlib.Path("/data/source")
        model_path = None
        for folder in os.listdir(look_dir):
            if folder.lower() == args["model_for_infer"].lower():
                model_path = folder
                logger.info(f"-- -- Model found at: {model_path}")

        if model_path is not None:
            model_for_infer = look_dir / model_path
            logger.info(
            f"-- -- Model for inference: {model_for_infer.as_posix()}")
        else:        
            model_for_infer = args["model_for_infer"]
            logger.error(
                f"-- -- Model for inference not found: {model_for_infer}")
            end_time = time.time() - start_time
            sc = 501
            responseHeader = {
                "status": sc,
                "time": end_time,
                "error": f"Model for inference not found: {model_for_infer}"
            }
            response = {
                "responseHeader": responseHeader,
                "response": None
            }
            return response, sc

        # Perform inference
        try:
            thetas = inferencer.predict(
                texts=[text_to_infer],
                model_for_infer_path=model_for_infer
            )

            end_time = time.time() - start_time

            # Generate response
            sc = 200
            responseHeader = {
                "status": sc,
                "time": end_time
            }

            response = {
                "responseHeader": responseHeader,
                "response": thetas
            }

            return response, sc

        except Exception as e:
            end_time = time.time() - start_time
            sc = 502
            responseHeader = {
                "status": sc,
                "time": end_time,
                "error": str(e)
            }

            response = {
                "responseHeader": responseHeader,
                "response": None
            }

            return response, sc
