import time
import logging
from flask import Flask, request, jsonify
from flask_restx import Namespace, Resource, Api, reqparse
from src.core.cpv_classifier_5 import CPVClassifierOpenAI

# -----------------------------
# ✅ Logging Configuration
# -----------------------------
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('PredictCpv')

# -----------------------------
# ✅ Flask Application Initialization
# -----------------------------
api = Namespace('Cpv operations')

# -----------------------------
# ✅ Request Parser for Input Validation
# -----------------------------
cpv_parser = reqparse.RequestParser()
cpv_parser.add_argument("api_key", help="OpenAI API key", required=True)
cpv_parser.add_argument("texts", help="Text(s) to predict CPV codes", action='split', required=True)

# -----------------------------
# ✅ API Endpoint for CPV Prediction
# -----------------------------
@api.route('/predictCpv')
class PredictCpv(Resource):
    @api.doc(
        parser=cpv_parser,
        responses={
            200: 'Success: CPV code generated successfully',
            400: 'Bad Request: Missing required parameters',
            502: 'CPV code generation error',
        }
    )
    def post(self):
        """
        API endpoint to predict CPV codes.

        - Expects JSON with:
            - "api_key": OpenAI API key
            - "texts": A list of texts to predict CPV codes
        - Returns CPV codes in JSON format
        """
        start_time = time.time()

        args = cpv_parser.parse_args()
        api_key = args['api_key']
        texts = args['texts']

        # Ensure texts is always a list
        if not isinstance(texts, list):
            texts = [texts]

        # Initialize CPVClassifierOpenAI with the provided API key
        cpv_classifier = CPVClassifierOpenAI(api_key, logger=logger)

        try:
            results = []
            for text in texts:
                prediction = cpv_classifier.predict_cpv_code(text)
                results.append({
                    "text": text,
                    "cpv_code": prediction["cpv_predicted"] if prediction else None
                })

            # Measure execution time
            execution_time = round(time.time() - start_time, 2)

            # Prepare successful response
            response = {
                "status": 200,
                "execution_time_seconds": execution_time,
                "predictions": results
            }

            logger.info(f"✅ CPV code(s) generated successfully: {results}")
            return response, 200

        except Exception as e:
            execution_time = round(time.time() - start_time, 2)
            logger.error(f"❌ CPV code generation error: {str(e)}")

            # Prepare error response
            response = {
                "status": 502,
                "execution_time_seconds": execution_time,
                "error": str(e)
            }
            return response, 502
