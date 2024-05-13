import time
from flask_restx import Namespace, Resource, reqparse
from src.core.cpv_classifier import CpvClassifier

import logging
logging.basicConfig(level='DEBUG')
logger = logging.getLogger('PredictCpv')

# ======================================================
# Define namespace for prediction operations
# ======================================================
api = Namespace('Cpv operations')

# ======================================================
# Create CpvClassifier object
# ======================================================
cpv_classifier = CpvClassifier(logger=logger)

# ======================================================
# Define parsers to take inputs from user
# ======================================================
cpv_parser = reqparse.RequestParser()
cpv_parser.add_argument("text_to_infer",
                        help="Text to whose CPV code is to be inferred",
                        required=True)


@api.route('/predictCpv')
class predictCpv(Resource):
    @api.doc(
        parser=cpv_parser,
        responses={
            200: 'Success: CPV code generated successfully',
            404: 'Not Found: Model or vectorizer required for prediction not found',
            500: 'Internal Server Error: An unexpected error occurred during the prediction process',
            502: "Error: Bad Gateway",
            503: 'Service Unavailable: Model or vectorizer not loaded properly'
        }

    )
    def post(self):

        start_time = time.time()

        args = cpv_parser.parse_args()

        try:
            result = cpv_classifier.predict_description(args['text_to_infer'])

            end_time = time.time() - start_time

            # Generate response
            sc = 200
            responseHeader = {
                "status": sc,
                "time": end_time,
            }
            response = {
                "responseHeader": responseHeader,
                "response": result
            }
            logger.info(
                f"-- -- CPV code generated successfully:{result}")

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
                "responseHeader": responseHeader
            }
            logger.error(
                f"-- -- CPV code generation error: {str(e)}")

            return response, sc
