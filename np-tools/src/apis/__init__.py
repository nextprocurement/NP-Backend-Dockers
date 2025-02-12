from flask_restx import Api

from .ns_embedder import api as ns1
from .ns_inferencer import api as ns2
from .ns_lemmas import api as ns3
from .ns_predict_cpv import api as ns4

api = Api(
    title="NP Tools API",
    version='1.0',
    description='RESTful API with a series of auxiliary endpoints. Right now it contains enpoints to: \n - Retrieve embeddings for a given document or word based on a Word2Vec (a precalculated Word2Vec model is assumed) or SBERT.\n - Retrieve document-topic representation of a given document based on a trained topic model. \n - Retrieve the lemmas of a given document.',
)

api.add_namespace(ns1, path='/embedder')
api.add_namespace(ns2, path='/inferencer')
api.add_namespace(ns3, path='/lemmatizer')
api.add_namespace(ns4, path='/predict_cpv')
