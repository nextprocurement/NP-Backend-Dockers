import logging
import time
from flask_restx import Namespace, Resource, reqparse
from src.core.objective_parser import ObjectiveParser

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ObjectiveExtractor")

# ======================================================
# Define namespace for Objective Extraction
# ======================================================
api = Namespace("Objective Extraction")

# ======================================================
# Define parser to accept text input
# ======================================================
text_parser = reqparse.RequestParser()
text_parser.add_argument(
    "text", type=str, required=True, help="Text from which to extract objectives."
)

# ======================================================
# Create Objective Parser object
# ======================================================
objective_parser = ObjectiveParser(logger=logger)


@api.route("/extract/")
class extract(Resource):
    @api.doc(
        parser=text_parser,
        responses={
            200: "Success: Objectives extracted successfully",
            400: "Bad Request: Invalid input",
            500: "Server Error: Failed to process the text",
        },
    )
    def post(self):
        start_time = time.time()
        args = text_parser.parse_args()
        input_text = args["text"]

        if not input_text.strip():
            return {"error": "Invalid input. Please provide non-empty text."}, 400

        try:
            # Extract objectives from text
            extracted_objectives = objective_parser.parse(input_text)

            end_time = time.time() - start_time
            response = {
                "responseHeader": {"status": 200, "time": end_time},
                "response": extracted_objectives,
            }
            logger.info("Objectives extracted successfully")
            
            return response, 200

        except Exception as e:
            logger.error(f"Error extracting objectives: {str(e)}")
            return {"error": "Failed to process the text", "details": str(e)}, 500