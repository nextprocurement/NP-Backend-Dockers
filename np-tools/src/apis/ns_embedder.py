"""
This script defines a Flask RESTful namespace for performing embedding operations.

Author: Lorena Calvo-Bartolomé
Date: 07/03/2024
"""

import pathlib
from flask_restx import Namespace, Resource, reqparse, fields, marshal

from src.core.embedder import Embedder
from src.core.lemmatizer import Lemmatizer

import logging
logging.basicConfig(level='DEBUG')
logger = logging.getLogger('Embedder')

# ======================================================
# Define namespace for embedding operations
# ======================================================
api = Namespace('Embedding operations')

# ======================================================
# Define models for the responses
# ======================================================
embedding_model = api.model('Embedding', {
    'embedding': fields.List(fields.List(fields.Float()), required=False, description='The embedding vector (e.g., word embeddings)'),
    'embedding_type': fields.String(required=False, description='The type of embedding used.'),
})

# ======================================================
# Define parsers to take inputs from user
# ======================================================
get_embedding_parser = reqparse.RequestParser()
get_embedding_parser.add_argument("text_to_embed",
                                  help="Text to be embedded",
                                  required=True)
get_embedding_parser.add_argument("embedding_model",
                                  help="Model to be used for embedding (either 'word2vec' or 'bert')",
                                  required=True)
get_embedding_parser.add_argument("model",
                                  help="Topic model on the basis of which the embeddings will be generated.",
                                  required=True)
get_embedding_parser.add_argument('lang',
                                  help='Language of the text to be lemmatize (es/en)',
                                  required=False,
                                  default='es')


# ======================================================
# Create Lemmatizer and Embedder objects
# ======================================================
embedder_manager = Embedder(logger=logger)
lemmatizer_manager = Lemmatizer(logger=logger)


@api.route('/getEmbedding/')
class getEmbedding(Resource):
    @api.doc(parser=get_embedding_parser)
    @api.doc(
        parser=get_embedding_parser,
        responses={
            200: 'Success: Embeddings generated successfully',
            501: 'Lemmatization error: An error occurred while lemmatizing the text',
            502: 'Embedding generation error: An error occurred while generating the embedding'
        }, model=embedding_model)
    def get(self):
        args = get_embedding_parser.parse_args()
        if args['embedding_model'] == 'word2vec':
            # Lemmatize the text
            try:
                text_to_embed = lemmatizer_manager.lemmatize(
                    args['text_to_embed'],
                    args['lang']
                )
            except Exception as e:
                return marshal({"status_response": str(e)}, embedding_model), 501

            try:
                model_path = pathlib.Path(
                    "data/source") / (args["model"])/("train_data") / ("model_w2v_corpus.model")
                logger.info(f"Model path: {model_path.as_posix()}")
                embeddings = embedder_manager.infer_embeddings(
                    embed_from=text_to_embed,
                    method=args["embedding_model"],
                    model_path=model_path.as_posix()
                )
                logger.info(f"Embeddings generated successfully:{embeddings}")
                logger.info(f"Type of embeddings generated: {type(embeddings)}")
                data = {
                    "embedding": embeddings,
                    "status_response": "Embeddings generated successfully"
                }
                return marshal(data, embedding_model), 200

            except Exception as e:
                return marshal({"status_response": str(e)}, embedding_model), 502
