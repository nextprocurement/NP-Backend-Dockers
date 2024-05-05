"""
This script defines a Flask RESTful namespace for performing embedding operations.

Author: Lorena Calvo-Bartolomé
Date: 07/03/2024
"""

import logging
import pathlib
import time
import os

from flask_restx import Namespace, Resource, reqparse
from src.core.embedder import Embedder
from src.core.lemmatizer import Lemmatizer

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
            501: 'Model for embeddings not found: An error occurred while trying to find the model for embeddings',
            502: 'Lemmas generation error: An error occurred while generating the lemmas',
            503: 'Invalid embedding model: The embedding model provided is not valid',
            504: 'Embeddings generation error: An error occurred while generating the embeddings'
        }
    )
    def get(self):

        start_time = time.time()

        args = get_embedding_parser.parse_args()

        if args['embedding_model'] == 'word2vec':
            # Get the path of the model (topic model) on the basis of which the embeddings will be generated
            look_dir = pathlib.Path("/data/source")
            tm_path = None
            for folder in os.listdir(look_dir):
                if folder.lower() == args["model"].lower():
                    tm_path = folder
            
            if tm_path is not None:
                model_path = look_dir / tm_path /("train_data") / ("model_w2v_corpus.model")
                logger.info(
                    f"-- -- Model for embeddings: {model_path.as_posix()}"
                )
            else:
                model_path = args["model"]
                end_time = time.time() - start_time
                sc = 501
                responseHeader = {
                    "status": sc,
                    "time": end_time,
                    "error": f"Model for embeddings not found: {model_path}"
                }
                response = {
                    "responseHeader": responseHeader,
                    "response": None
                }
                return response, sc

            logger.info(f"-- --Model path: {model_path.as_posix()}")

        if args['embedding_model'] == 'word2vec':
            # If the embedding model is word2vec, lemmatize the text
            try:
                text_to_embed = lemmatizer_manager.lemmatize(
                    args['text_to_embed'],
                    args['lang']
                )
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
                f"-- -- String representation of embeddings generated successfully:{embeddings_str}")

            return response, sc

        except Exception as e:
            end_time = time.time() - start_time
            sc = 504
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
