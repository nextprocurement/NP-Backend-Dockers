"""
This script offers a basic lemmatizer that leverages the spaCy library to lemmatize a given text.

Author: Lorena Calvo-Bartolomé
Date: 07/03/2024
"""

import configparser
import logging
import spacy


class Lemmatizer(object):
    def __init__(
        self,
        logger: logging.Logger = None,
        config_file: str = "/config/config.cf",
    ) -> None:

        # Set logger
        if logger:
            self._logger = logger
        else:
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger(__name__)

        # Read configuration from config file
        cf = configparser.ConfigParser()
        cf.read(config_file)

        # By default uses these but they could be configured in the config file (they need to be also downloaded from spacy. The latter is done in the Dockerfile)
        self._nlp_es = spacy.load("es_dep_news_trf")
        self._nlp_en = spacy.load("en_core_web_trf")

    def lemmatize(
        self,
        text: str,
        language: str = "es",
    ) -> str:
        """
        Lemmatizes a given text.

        Parameters
        ----------
        text : str
            The text to lemmatize.
        language : str, optional
            The language of the text. Default is "es" (Spanish).

        Returns
        -------
        str
            The lemmatized text.
        """

        if language == "es":
            nlp = self._nlp_es
        elif language == "en":
            nlp = self._nlp_en
        else:
            self._logger.error(
                f"Language not supported. Please use 'es' or 'en'.")
            raise ValueError(
                "Language not supported. Please use 'es' or 'en'.")

        doc = nlp(text)
        lemmatized_words = [token.lemma_ for token in doc]

        return " ".join(lemmatized_words)
