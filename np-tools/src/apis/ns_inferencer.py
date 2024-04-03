"""
This script defines a Flask RESTful namespace for performing inference.

Author: Lorena Calvo-Bartolomé
Date: 19/03/2024
"""

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
    @api.doc(parser=infer_doc_parser)
    def get(self):
        args = infer_doc_parser.parse_args()
        text_to_infer = args['text_to_infer']
        model_for_infer = pathlib.Path(
            "/data/source") / (args["model_for_infer"])

        # Perform inference
        try:
            return inferencer.predict(
                texts=[text_to_infer],
                model_for_infer_path=model_for_infer
            )
        except Exception as e:
            return {'message': 'An error occurred while performing inference'}, 502
