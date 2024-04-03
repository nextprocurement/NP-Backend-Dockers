"""
This script defines a Flask RESTful namespace for performing embedding operations.

Author: Lorena Calvo-Bartolomé
Date: 07/03/2024
"""

import pathlib
import time
from flask_restx import Namespace, Resource, reqparse

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
# Define parsers to take inputs from user
# ======================================================
get_embedding_parser = reqparse.RequestParser()
get_embedding_parser.add_argument("text_to_embed",
                                  help="Text to be embedded",
                                  required=True)
get_embedding_parser.add_argument("embedding_model",
                                  help="Model to be used for embedding (either 'word2vec' or 'bert')",
                                  required=False,
                                  default='word2vec')
get_embedding_parser.add_argument("model",
                                  help="Topic model on the basis of which the embeddings will be generated.",
                                  required=False)
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
    @api.doc(
        parser=get_embedding_parser,
        responses={
            200: 'Success: Embeddings generated successfully',
            501: 'Lemmatization error: An error occurred while lemmatizing the text',
            502: 'Embedding generation error: An error occurred while generating the embedding'
        }
    )
    def get(self):
        args = get_embedding_parser.parse_args()

        start_time = time.time()

        if args['embedding_model'] == 'word2vec':
            # If the embedding model is word2vec, lemmatize the text
            try:
                text_to_embed = lemmatizer_manager.lemmatize(
                    args['text_to_embed'],
                    args['lang']
                )
            except Exception as e:
                end_time = time.time() - start_time
                sc = 501
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

            # Get the path of the model (topic model) on the basis of which the embeddings will be generated
            model_path = pathlib.Path(
                "/data/source") / (args["model"])/("train_data") / ("model_w2v_corpus.model")
            logger.info(f"Model path: {model_path.as_posix()}")

        elif args['embedding_model'] == 'bert':
            # If the embedding model is bert, no need to lemmatize the text
            text_to_embed = args['text_to_embed']
            model_path = None

        elif args['embedding_model'] not in ['word2vec', 'bert']:
            end_time = time.time() - start_time
            sc = 503
            responseHeader = {
                "status": sc,
                "time": end_time,
                "error": f"Invalid embedding model: {args['embedding_model']}"
            }
            response = {
                "responseHeader": responseHeader,
                "response": None
            }
            return response, sc

        try:

            # Generate embeddings
            embeddings = embedder_manager.infer_embeddings(
                embed_from=[text_to_embed],
                method=args["embedding_model"],
                model_path=model_path
            )

            # Generate string representation of embeddings
            def get_topic_embeddings(vector):
                repr = " ".join(
                    [f"e{idx}|{val}" for idx, val in enumerate(vector)]).rstrip()

                return repr
            embeddings_str = get_topic_embeddings(embeddings)

            end_time = time.time() - start_time

            # Generate response
            sc = 200
            responseHeader = {
                "status": sc,
                "time": end_time,
            }
            response = {
                "responseHeader": responseHeader,
                "response": embeddings_str
            }
            logger.info(
                f"String representation of embeddings generated successfully:{embeddings_str}")

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
