import logging
import os
from dotenv import load_dotenv
from flask_restx import Namespace, Resource, reqparse
from werkzeug.datastructures import FileStorage
from src.core.metadata_extractor import MetadataExtractor


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "La variable de entorno OPENAI_API_KEY no está configurada")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('MetadataExtractor')

api = Namespace('Metadata Extraction')

metadata_parser = reqparse.RequestParser()
metadata_parser.add_argument(
    'file',
    type=FileStorage,
    location='files',
    required=True,
    help='Parquet file containing procurement documents'
)

@api.route('/extract')
class extract(Resource):
    @api.doc(
        parser=metadata_parser,
        responses={
            200: 'Success: Metadata extracted successfully',
            400: 'Bad Request: Missing required parameters or file issue',
            502: 'Metadata extraction error',
        }
    )
    def post(self):

        args = metadata_parser.parse_args()
        file = args['file']

        # Inicializar el extractor
        extractor = MetadataExtractor(api_key=api_key, logger=logger)

        try:
            # Ejecutar la extracción
            result, code = extractor.extract_metadata_from_file(file)
            return result, code

        except Exception as e:
            logger.error(f"❌ Metadata extraction error: {str(e)}")
            return {
                "status": 502,
                "error": str(e)
            }, 502