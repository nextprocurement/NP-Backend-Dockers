from flask_restx import Api

from .ns_embedder import api as ns1
from .ns_inferencer import api as ns2

api = Api(
    title="NP Tools API",
    version='1.0',
    description='',
)

api.add_namespace(ns1, path='/embedder')
#api.add_namespace(ns2, path='/inferencer')