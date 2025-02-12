"""
This script defines a Flask RESTful namespace for performing lemmatization operations.

Author: Lorena Calvo-Bartolomé
Date: 03/04/2024
"""

import logging
import pathlib
import time

from flask_restx import Namespace, Resource, reqparse
from src.core.lemmatizer import Lemmatizer

logging.basicConfig(level='DEBUG')
logger = logging.getLogger('Lemmtizer')

# ======================================================
# Define namespace for lemmatization operations
# ======================================================
api = Namespace('Lemmatization operations')

# ======================================================
# Define parsers to take inputs from user
# ======================================================
get_lemmas_parser = reqparse.RequestParser()
get_lemmas_parser.add_argument("text_to_lemmatize",
                                help="Text to be lemmatize",
                                required=True)
get_lemmas_parser.add_argument('lang',
                                help='Language of the text to be lemmatize (es/en)',
                                required=False,
                                default='es')


# ======================================================
# Create Lemmatizer objects
# ======================================================
lemmatizer_manager = Lemmatizer(logger=logger)


@api.route('/getLemmas/')
class getLemmas(Resource):
    @api.doc(
        parser=get_lemmas_parser,
        responses={
            200: 'Success: Embeddings generated successfully',
            501: 'Lemmas generation error: An error occurred while generating the lemmas',
        }
    )
    def get(self):

        start_time = time.time()

        args = get_lemmas_parser.parse_args()

        try:
            lemmas = lemmatizer_manager.lemmatize(
                args['text_to_lemmatize'],
                args['lang']
            )
            
            end_time = time.time() - start_time

            # Generate response
            sc = 200
            responseHeader = {
                "status": sc,
                "time": end_time,
            }
            response = {
                "responseHeader": responseHeader,
                # Update this if at one points we want to make it for N documents at once
                "response": [
                    {
                        "id": 0,
                        "lemmas": lemmas
                    }   
                ]
            }
            logger.info(
                f"-- -- Lemmas generated successfully:{lemmas}")
            
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

       