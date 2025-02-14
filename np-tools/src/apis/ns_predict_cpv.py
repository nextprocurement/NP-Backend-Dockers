import time
import logging
import os
from dotenv import load_dotenv
from flask import request
from flask_restx import Namespace, Resource, reqparse
from src.core.cpv_classifier_5 import CPVClassifierOpenAI

# -----------------------------
# ✅ Cargar Variables de Entorno
# -----------------------------
load_dotenv()  # Cargar las variables desde el archivo .env
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("La variable de entorno OPENAI_API_KEY no está configurada")

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
        start_time = time.time()

        args = cpv_parser.parse_args()
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

            execution_time = round(time.time() - start_time, 2)

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

            response = {
                "status": 502,
                "execution_time_seconds": execution_time,
                "error": str(e)
            }
            return response, 502
