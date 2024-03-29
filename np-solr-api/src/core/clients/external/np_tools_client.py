"""
This  module provides 2 classes to handle NP Tools API responses and requests.

The NPToolsResponse class handles NP Tools API response and errors, while the NPToolsClient class handles requests to the NP Tools API.

Author: Lorena Calvo-Bartolomé
Date: 21/05/2023
"""

import logging
import os
from urllib.parse import urlencode

import requests
from src.core.clients.external.api_generic.client import Client
from src.core.clients.external.api_generic.response import Response


class NPToolsResponse(Response):
    """
    A class to handle Inferencer API response and errors.
    """

    def __init__(
        self,
        resp: requests.Response,
        logger: logging.Logger
    ) -> None:

        super().__init__(resp, logger)
        return


class NPToolsClient(Client):
    """
    A class to handle NP Tools API requests.
    """

    def __init__(
        self,
        logger: logging.Logger
    ) -> None:
        """
        Parameters
        ----------
        logger : logging.Logger
            The logger object to log messages and errors.
        """

        super().__init__(logger, "NP Tools Client")

        # Get the NP Tools URL from the environment variables
        self.nptools_url = os.environ.get('NP_TOOLS_URL')

        return

    def _do_request(
        self,
        type: str,
        url: str,
        timeout: int = 10,
        **params
    ) -> NPToolsResponse:
        """Sends a request to the Inferencer API and returns an object of the NPToolsResponse class.

        Parameters
        ----------
        type : str
            The type of the request.
        url : str
            The URL of the Inferencer API.
        timeout : int, optional
            The timeout of the request in seconds, by default 10.
        **params: dict
            The parameters of the request.

        Returns
        -------
        NPToolsResponse: NPToolsResponse
            An object of the NPToolsResponse class.
        """

        # Send request
        resp = super()._do_request(type, url, timeout, **params)

        # Parse NP Tools response
        inf_resp = NPToolsResponse(resp, self.logger)

        return inf_resp

    def get_embedding(
        self,
        text_to_embed: str,
        embedding_model: str,
        lang: str,
        model_for_embedding: str = None
    ) -> NPToolsResponse:
        """Get the embedding of a word using the given model.

        Parameters
        ----------
        word_to_embed : str
            The word to embed.
        embedding_model : str
            The model to use for embedding (either 'word2vec' or 'bert')
        model_for_embedding : str
            The model to use for embedding, i.e., the topic model on the basis of which the embeddings where generated,
        lang : str
            The language of the text to be lemmatized (es/en)

        Returns
        -------
        NPToolsResponse: NPToolsResponse
            An object of the NPToolsResponse class.
        """

        headers_ = {'Accept': 'application/json'}

        if model_for_embedding is None:
            params_ = {
                'text_to_embed': text_to_embed,
                'embedding_model': embedding_model,
                'lang': lang
            }
        else:
            params_ = {
                'text_to_embed': text_to_embed,
                'embedding_model': embedding_model,
                'model': model_for_embedding,
                'lang': lang
            }
            
        encoded_params = urlencode(params_)

        url_ = '{}/embedder/getEmbedding/?{}'.format(
            self.nptools_url, encoded_params)
        
        self.logger.info(f"URL: {url_}")

        # Send request to NPtooler
        resp = self._do_request(
            type="get", url=url_, timeout=120, headers=headers_)
        
        self.logger.info(f"Response: {resp}")
        
        return resp
