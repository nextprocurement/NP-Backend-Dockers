"""
This module provides a specific class for handeling the Solr API responses and requests of the NP-Solr-Service.

Author: Lorena Calvo-Bartolomé
Date: 17/04/2023
Modifed: 24/01/2024 (Updated for NP-Solr-Service (NextProcurement Proyect))
"""

import configparser
import logging
import pathlib
import re
import time
import pandas as pd
from typing import List, Union
# from src.core.clients.external.np_inferencer_client import NPInferencerClient 
from src.core.clients.external.np_tools_client import NPToolsClient
# #TODO:Add and update if necessary
from src.core.clients.base.solr_client import SolrClient
from src.core.entities.corpus import Corpus
from src.core.entities.model import Model
from src.core.entities.queries import Queries


class NPSolrClient(SolrClient):

    def __init__(self,
                 logger: logging.Logger,
                 config_file: str = "/config/config.cf") -> None:
        super().__init__(logger)

        # Read configuration from config file
        cf = configparser.ConfigParser()
        cf.read(config_file)
        self.solr_config = "np_config"
        self.batch_size = int(cf.get('restapi', 'batch_size'))
        self.corpus_col = cf.get('restapi', 'corpus_col')
        self.no_meta_fields = cf.get('restapi', 'no_meta_fields').split(",")
        self.path_source = pathlib.Path(cf.get('restapi', 'path_source'))
        # TODO: Check if necessary
        self.thetas_max_sum = int(cf.get('restapi', 'thetas_max_sum'))
        # TODO: Check if necessary
        self.betas_max_sum = int(cf.get('restapi', 'betas_max_sum'))

        # Create Queries object for managing queries
        self.querier = Queries()

        # Create InferencerClient to send requests to the Inferencer API
        # TODO: Uncomment and update if necessary
        # self.inferencer = NPInferencerClient(logger)
        self.nptooler = NPToolsClient(logger)

        return

    # ======================================================
    # CORPUS-RELATED OPERATIONS
    # ======================================================

    def index_corpus(self,
                     corpus_raw: str) -> None:
        """
        This method takes the name of a corpus raw file as input. It creates a Solr collection with the stem name of the file, which is obtained by converting the file name to lowercase (for example, if the input is 'Cordis', the stem would be 'cordis'). However, this process occurs only if the directory structure (self.path_source / corpus_raw / parquet) exists.

        After creating the Solr collection, the method reads the corpus file, extracting the raw information of each document. Subsequently, it sends a POST request to the Solr server to index the documents in batches.

        Parameters
        ----------
        corpus_raw : str
            The string name of the corpus raw file to be indexed.

        """

        # 1. Get full path and stem of the logical corpus
        corpus_to_index = self.path_source / (corpus_raw + ".parquet")
        corpus_logical_name = corpus_to_index.stem.lower()
        
        self.logger.info(f"Corpus to index: {corpus_to_index}")
        self.logger.info(f"Corpus logical name: {corpus_logical_name}")

        # 2. Create collection
        corpus, err = self.create_collection(
            col_name=corpus_logical_name, config=self.solr_config)
        if err == 409:
            self.logger.info(
                f"-- -- Collection {corpus_logical_name} already exists.")
            return
        else:
            self.logger.info(
                f"-- -- Collection {corpus_logical_name} successfully created.")

        # 3. Add corpus collection to self.corpus_col. If Corpora has not been created already, create it
        corpus, err = self.create_collection(
            col_name=self.corpus_col, config=self.solr_config)
        self.logger.info(f"-- -- Collection {self.corpus_col} successfully created.")
        if err == 409:
            self.logger.info(
                f"-- -- Collection {self.corpus_col} already exists.")

            # 3.1. Do query to retrieve last id in self.corpus_col
            # http://localhost:8983/solr/#/{self.corpus_col}/query?q=*:*&q.op=OR&indent=true&sort=id desc&fl=id&rows=1&useParams=
            sc, results = self.execute_query(q='*:*',
                                             col_name=self.corpus_col,
                                             sort="id desc",
                                             rows="1",
                                             fl="id")
            if sc != 200:
                self.logger.error(
                    f"-- -- Error getting latest used ID. Aborting operation...")
                return
            # Increment corpus_id for next corpus to be indexed
            corpus_id = int(results.docs[0]["id"]) + 1
        else:
            self.logger.info(
                f"Collection {self.corpus_col} successfully created.")
            corpus_id = 1

        # 4. Create Corpus object and extract info from the corpus to index
        corpus = Corpus(corpus_to_index)
        json_docs = corpus.get_docs_raw_info()
        self.logger.info(f"-- -- Corpus info extracted")
        corpus_col_upt = corpus.get_corpora_update(id=corpus_id)
        self.logger.info(f"-- -- corpus_col_upt extracted")
        self.logger.info(f"{corpus_col_upt}")

        # 5. Index corpus and its fiels in CORPUS_COL
        self.logger.info(
            f"-- -- Indexing of {corpus_logical_name} info in {self.corpus_col} starts.")
        self.index_documents(corpus_col_upt, self.corpus_col, self.batch_size)
        self.logger.info(
            f"-- -- Indexing of {corpus_logical_name} info in {self.corpus_col} completed.")

        # 6. Index documents in corpus collection
        self.logger.info(
            f"-- -- Indexing of {corpus_logical_name} in {corpus_logical_name} starts.")
        self.index_documents(json_docs, corpus_logical_name, self.batch_size)
        self.logger.info(
            f"-- -- Indexing of {corpus_logical_name} in {corpus_logical_name} completed.")

        return

    def list_corpus_collections(self) -> Union[List, int]:
        """Returns a list of the names of the corpus collections that have been created in the Solr server.

        Returns
        -------
        corpus_lst: List
            List of the names of the corpus collections that have been created in the Solr server.
        """

        sc, results = self.execute_query(q='*:*',
                                         col_name=self.corpus_col,
                                         fl="corpus_name")
        if sc != 200:
            self.logger.error(
                f"-- -- Error getting corpus collections in {self.corpus_col}. Aborting operation...")
            return

        corpus_lst = [doc["corpus_name"] for doc in results.docs]

        return corpus_lst, sc

    def get_corpus_coll_fields(self, corpus_col: str) -> Union[List, int]:
        """Returns a list of the fields of the corpus collection given by 'corpus_col' that have been defined in the Solr server.

        Parameters
        ----------
        corpus_col : str
            Name of the corpus collection whose fields are to be retrieved.

        Returns
        -------
        models: list
            List of fields of the corpus collection
        sc: int
            Status code of the request
        """
        sc, results = self.execute_query(q='corpus_name:"'+corpus_col+'"',
                                         col_name=self.corpus_col,
                                         fl="fields")

        if sc != 200:
            self.logger.error(
                f"-- -- Error getting fields of {corpus_col}. Aborting operation...")
            return

        return results.docs[0]["fields"], sc

    def get_corpus_raw_path(self, corpus_col: str) -> Union[pathlib.Path, int]:
        """Returns the path of the logical corpus file associated with the corpus collection given by 'corpus_col'.

        Parameters
        ----------
        corpus_col : str
            Name of the corpus collection whose path is to be retrieved.

        Returns
        -------
        path: pathlib.Path
            Path of the logical corpus file associated with the corpus collection given by 'corpus_col'.
        sc: int
            Status code of the request
        """

        sc, results = self.execute_query(q='corpus_name:"'+corpus_col+'"',
                                         col_name=self.corpus_col,
                                         fl="corpus_path")
        if sc != 200:
            self.logger.error(
                f"-- -- Error getting corpus path of {corpus_col}. Aborting operation...")
            return

        self.logger.info(results.docs[0]["corpus_path"])
        return pathlib.Path(results.docs[0]["corpus_path"]), sc

    def get_id_corpus_in_corpora(self, corpus_col: str) -> Union[int, int]:
        """Returns the ID of the corpus collection given by 'corpus_col' in the self.corpus_col collection.

        Parameters
        ----------
        corpus_col : str
            Name of the corpus collection whose ID is to be retrieved.

        Returns
        -------
        id: int
            ID of the corpus collection given by 'corpus_col' in the self.corpus_col collection.
        """

        sc, results = self.execute_query(q='corpus_name:"'+corpus_col+'"',
                                         col_name=self.corpus_col,
                                         fl="id")
        if sc != 200:
            self.logger.error(
                f"-- -- Error getting corpus ID. Aborting operation...")
            return

        return results.docs[0]["id"], sc

    def get_corpus_MetadataDisplayed(self, corpus_col: str) -> Union[List, int]:
        """Returns a list of the fileds of the corpus collection indicating what metadata will be displayed in the NP front upon user request.

        Parameters
        ----------
        corpus_col : str
            Name of the corpus collection whose MetadataDisplayed are to be retrieved.
        sc: int
            Status code of the request
        """

        sc, results = self.execute_query(q='corpus_name:"'+corpus_col+'"',
                                         col_name=self.corpus_col,
                                         fl="MetadataDisplayed")

        if sc != 200:
            self.logger.error(
                f"-- -- Error getting MetadataDisplayed of {corpus_col}. Aborting operation...")
            return

        return results.docs[0]["MetadataDisplayed"], sc

    def get_corpus_SearcheableField(self, corpus_col: str) -> Union[List, int]:
        """Returns a list of the fields used for autocompletion in the document search in the similarities function and in the document search function.

        Parameters
        ----------
        corpus_col : str
            Name of the corpus collection whose SearcheableField are to be retrieved.
        sc: int
            Status code of the request
        """

        sc, results = self.execute_query(q='corpus_name:"'+corpus_col+'"',
                                         col_name=self.corpus_col,
                                         fl="SearcheableFields")

        if sc != 200:
            self.logger.error(
                f"-- -- Error getting SearcheableField of {corpus_col}. Aborting operation...")
            return

        return results.docs[0]["SearcheableFields"], sc

    def get_corpus_models(self, corpus_col: str) -> Union[List, int]:
        """Returns a list with the models associated with the corpus given by 'corpus_col'

        Parameters
        ----------
        corpus_col : str
            Name of the corpus collection whose models are to be retrieved.

        Returns
        -------
        models: list
            List of models associated with the corpus
        sc: int
            Status code of the request
        """

        sc, results = self.execute_query(q='corpus_name:"'+corpus_col+'"',
                                         col_name=self.corpus_col,
                                         fl="models")

        if sc != 200:
            self.logger.error(
                f"-- -- Error getting models of {corpus_col}. Aborting operation...")
            return

        return results.docs[0]["models"], sc

    def delete_corpus(self,
                      corpus_logical_path: str) -> None:
        """Given the string path of corpus file, it deletes the Solr collection associated with it. Additionally, it removes the document entry of the corpus in the self.corpus_col collection and all the models that have been trained with such a logical corpus.

        Parameters
        ----------
        corpus_logical_path : str
            The path of the logical corpus file to be indexed.
        """

        # 1. Get stem of the logical corpus
        corpus_logical_name = pathlib.Path(corpus_logical_path).stem.lower()

        # 2. Delete corpus collection
        _, sc = self.delete_collection(col_name=corpus_logical_name)
        if sc != 200:
            self.logger.error(
                f"-- -- Error deleting corpus collection {corpus_logical_name}")
            return

        # 3. Get ID and associated models of corpus collection in self.corpus_col
        sc, results = self.execute_query(q='corpus_name:'+corpus_logical_name,
                                         col_name=self.corpus_col,
                                         fl="id,models")
        if sc != 200:
            self.logger.error(
                f"-- -- Error getting corpus ID. Aborting operation...")
            return

        # 4. Delete all models associated with the corpus if any
        if "models" in results.docs[0].keys():
            for model in results.docs[0]["models"]:
                _, sc = self.delete_collection(col_name=model)
                if sc != 200:
                    self.logger.error(
                        f"-- -- Error deleting model collection {model}")
                    return

        # 5. Remove corpus from self.corpus_col
        sc = self.delete_doc_by_id(
            col_name=self.corpus_col, id=results.docs[0]["id"])
        if sc != 200:
            self.logger.error(
                f"-- -- Error deleting corpus from {self.corpus_col}")
        return

    def check_is_corpus(self, corpus_col) -> bool:
        """Checks if the collection given by 'corpus_col' is a corpus collection.

        Parameters
        ----------
        corpus_col : str
            Name of the collection to be checked.

        Returns
        -------
        is_corpus: bool
            True if the collection is a corpus collection, False otherwise.
        """

        corpus_colls, sc = self.list_corpus_collections()
        if corpus_col not in corpus_colls:
            self.logger.error(
                f"-- -- {corpus_col} is not a corpus collection. Aborting operation...")
            return False

        return True

    def check_corpus_has_model(self, corpus_col, model_name) -> bool:
        """Checks if the collection given by 'corpus_col' has a model with name 'model_name'.

        Parameters
        ----------
        corpus_col : str
            Name of the collection to be checked.
        model_name : str
            Name of the model to be checked.

        Returns
        -------
        has_model: bool
            True if the collection has the model, False otherwise.
        """

        corpus_fields, sc = self.get_corpus_coll_fields(corpus_col)
        if 'doctpc_' + model_name not in corpus_fields:
            self.logger.error(
                f"-- -- {corpus_col} does not have the field doctpc_{model_name}. Aborting operation...")
            return False
        return True

    def modify_corpus_SearcheableFields(
        self,
        SearcheableFields: str,
        corpus_col: str,
        action: str
    ) -> None:
        """
        Given a list of fields, it adds them to the SearcheableFields field of the corpus collection given by 'corpus_col' if action is 'add', or it deletes them from the SearcheableFields field of the corpus collection given by 'corpus_col' if action is 'delete'.

        Parameters
        ----------
        SearcheableFields : str
            List of fields to be added to the SearcheableFields field of the corpus collection given by 'corpus_col'.
        corpus_col : str
            Name of the corpus collection whose SearcheableFields field is to be updated.
        action : str
            Action to be performed. It can be 'add' or 'delete'.
        """

        # 1. Get full path
        corpus_path, _ = self.get_corpus_raw_path(corpus_col)

        SearcheableFields = SearcheableFields.split(",")

        # 2. Check that corpus_col is indeed a corpus collection
        if not self.check_is_corpus(corpus_col):
            return

        # 3. Create Corpus object, get SearcheableField and index information in corpus collection
        corpus = Corpus(corpus_path)
        corpus_update, new_SearcheableFields = corpus.get_corpus_SearcheableField_update(
            new_SearcheableFields=SearcheableFields,
            action=action)
        self.logger.info(
            f"-- -- Indexing new SearcheableField information in {corpus_col} collection")
        self.index_documents(corpus_update, corpus_col, self.batch_size)
        self.logger.info(
            f"-- -- Indexing new SearcheableField information in {self.corpus_col} completed.")

        # 4. Get self.corpus_col update
        corpora_id, _ = self.get_id_corpus_in_corpora(corpus_col)
        corpora_update = corpus.get_corpora_SearcheableField_update(
            id=corpora_id,
            field_update=new_SearcheableFields,
            action="set")
        self.logger.info(
            f"-- -- Indexing new SearcheableField information in {self.corpus_col} starts.")
        self.index_documents(corpora_update, self.corpus_col, self.batch_size)
        self.logger.info(
            f"-- -- Indexing new SearcheableField information in {self.corpus_col} completed.")

        return

    # ======================================================
    # MODEL-RELATED OPERATIONS
    # ======================================================

    def index_model(self, model_path: str) -> None:
        """
        Given the string path of a model created with the ITMT (i.e., the name of one of the folders representing a model within the TMmodels folder), it extracts the model information and that of the corpus used for its generation. It then adds a new field in the corpus collection of type 'VectorField' and name 'doctpc_{model_name}, and index the document-topic proportions in it. At last, it index the rest of the model information in the model collection.

        Parameters
        ----------
        model_path : str
            Path to the folder of the model to be indexed.
        """

        # 1. Get stem of the model folder
        model_to_index =  self.path_source / model_path
        model_name = model_to_index.stem.lower()

        # 2. Create collection
        _, err = self.create_collection(
            col_name=model_name, config=self.solr_config)
        if err == 409:
            self.logger.info(
                f"-- -- Collection {model_name} already exists.")
            return
        else:
            self.logger.info(
                f"-- -- Collection {model_name} successfully created.")

        # 3. Create Model object and extract info from the corpus to index
        model = Model(model_to_index)
        json_docs, corpus_name = model.get_model_info_update(action='set')
        if not self.check_is_corpus(corpus_name):
            return
        corpora_id, _ = self.get_id_corpus_in_corpora(corpus_name)
        field_update = model.get_corpora_model_update(
            id=corpora_id, action='add')

        # 4. Add field for the doc-tpc distribution associated with the model being indexed in the document associated with the corpus
        self.logger.info(
            f"-- -- Indexing model information of {model_name} in {self.corpus_col} starts.")

        self.index_documents(field_update, self.corpus_col, self.batch_size)
        self.logger.info(
            f"-- -- Indexing of model information of {model_name} info in {self.corpus_col} completed.")

        # 5. Modify schema in corpus collection to add field for the doc-tpc distribution and the similarities associated with the model being indexed
        model_key = 'doctpc_' + model_name
        sim_model_key = 'sim_' + model_name
        self.logger.info(
            f"-- -- Adding field {model_key} in {corpus_name} collection")
        _, err = self.add_field_to_schema(
            col_name=corpus_name, field_name=model_key, field_type='VectorField')
        self.logger.info(
            f"-- -- Adding field {sim_model_key} in {corpus_name} collection")
        _, err = self.add_field_to_schema(
            col_name=corpus_name, field_name=sim_model_key, field_type='VectorFloatField')

        # 6. Index doc-tpc information in corpus collection
        self.logger.info(
            f"-- -- Indexing model information in {corpus_name} collection")
        self.index_documents(json_docs, corpus_name, self.batch_size)

        self.logger.info(
            f"-- -- Indexing model information in {model_name} collection")
        json_tpcs = model.get_model_info()

        self.index_documents(json_tpcs, model_name, self.batch_size)

        return

    def list_model_collections(self) -> Union[List[str], int]:
        """Returns a list of the names of the model collections that have been created in the Solr server.

        Returns
        -------
        models_lst: List[str]
            List of the names of the model collections that have been created in the Solr server.
        sc: int
            Status code of the request.
        """
        sc, results = self.execute_query(q='*:*',
                                         col_name=self.corpus_col,
                                         fl="models")
        if sc != 200:
            self.logger.error(
                f"-- -- Error getting corpus collections in {self.corpus_col}. Aborting operation...")
            return

        models_lst = [model for doc in results.docs if bool(
            doc) for model in doc["models"]]
        self.logger.info(f"-- -- Models found: {models_lst}")

        return models_lst, sc

    def delete_model(self, model_path: str) -> None:
        """
        Given the string path of a model created with the ITMT (i.e., the name of one of the folders representing a model within the TMmodels folder), 
        it deletes the model collection associated with it. Additionally, it removes the document-topic proportions field in the corpus collection and removes the fields associated with the model and the model from the list of models in the corpus document from the self.corpus_col collection.

        Parameters
        ----------
        model_path : str
            Path to the folder of the model to be indexed.
        """

        # 1. Get stem of the model folder
        model_to_index =  self.path_source / model_path
        model_name = model_to_index.stem.lower()

        # 2. Delete model collection
        _, sc = self.delete_collection(col_name=model_name)
        if sc != 200:
            self.logger.error(
                f"-- -- Error occurred while deleting model collection {model_name}. Stopping...")
            return
        else:
            self.logger.info(
                f"-- -- Model collection {model_name} successfully deleted.")

        # 3. Create Model object and extract info from the corpus associated with the model
        model = Model(model_to_index)
        json_docs, corpus_name = model.get_model_info_update(action='remove')
        sc, results = self.execute_query(q='corpus_name:'+corpus_name,
                                         col_name=self.corpus_col,
                                         fl="id")
        if sc != 200:
            self.logger.error(
                f"-- -- Corpus collection not found in {self.corpus_col}")
            return
        field_update = model.get_corpora_model_update(
            id=results.docs[0]["id"], action='remove')

        # 4. Remove field for the doc-tpc distribution associated with the model being deleted in the document associated with the corpus
        self.logger.info(
            f"-- -- Deleting model information of {model_name} in {self.corpus_col} starts.")
        self.index_documents(field_update, self.corpus_col, self.batch_size)
        self.logger.info(
            f"-- -- Deleting model information of {model_name} info in {self.corpus_col} completed.")

        # 5. Delete doc-tpc information from corpus collection
        self.logger.info(
            f"-- -- Deleting model information from {corpus_name} collection")
        self.index_documents(json_docs, corpus_name, self.batch_size)

        # 6. Modify schema in corpus collection to delete field for the doc-tpc distribution and similarities associated with the model being indexed
        model_key = 'doctpc_' + model_name
        sim_model_key = 'sim_' + model_name
        self.logger.info(
            f"-- -- Deleting field {model_key} in {corpus_name} collection")
        _, err = self.delete_field_from_schema(
            col_name=corpus_name, field_name=model_key)
        self.logger.info(
            f"-- -- Deleting field {sim_model_key} in {corpus_name} collection")
        _, err = self.delete_field_from_schema(
            col_name=corpus_name, field_name=sim_model_key)

        return

    def check_is_model(self, model_col) -> bool:
        """Checks if the model_col is a model collection. If not, it aborts the operation.

        Parameters
        ----------
        model_col : str
            Name of the model collection.

        Returns
        -------
        is_model : bool
            True if the model_col is a model collection, False otherwise.
        """

        model_colls, sc = self.list_model_collections()
        if model_col not in model_colls:
            self.logger.error(
                f"-- -- {model_col} is not a model collection. Aborting operation...")
            return False
        return True

    # ======================================================
    # AUXILIARY FUNCTIONS
    # ======================================================
    def custom_start_and_rows(self, start, rows, col) -> Union[str, str]:
        """Checks if start and rows are None. If so, it returns the number of documents in the collection as the value for rows and 0 as the value for start.

        Parameters
        ----------
        start : str
            Start parameter of the query.
        rows : str
            Rows parameter of the query.
        col : str
            Name of the collection.

        Returns
        -------
        start : str
            Final start parameter of the query.
        rows : str
            Final rows parameter of the query.
        """
        if start is None:
            start = str(0)
        if rows is None:
            numFound_dict, sc = self.do_Q3(col)
            rows = str(numFound_dict['ndocs'])

            if sc != 200:
                self.logger.error(
                    f"-- -- Error executing query Q3. Aborting operation...")
                return

        return start, rows

    # ======================================================
    # QUERIES
    # ======================================================

    def do_Q1(self,
              corpus_col: str,
              doc_id: str,
              model_name: str) -> Union[dict, int]:
        """Executes query Q1.

        Parameters
        ----------
        corpus_col : str
            Name of the corpus collection.
        id : str
            ID of the document to be retrieved.
        model_name : str
            Name of the model to be used for the retrieval.

        Returns
        -------
        thetas: dict
            JSON object with the document-topic proportions (thetas)
        sc : int
            The status code of the response.  
        """

        # 0. Convert corpus and model names to lowercase
        corpus_col = corpus_col.lower()
        model_name = model_name.lower()

        # 1. Check that corpus_col is indeed a corpus collection
        if not self.check_is_corpus(corpus_col):
            return

        # 2. Check that corpus_col has the model_name field
        if not self.check_corpus_has_model(corpus_col, model_name):
            return

        # 3. Execute query
        q1 = self.querier.customize_Q1(id=doc_id, model_name=model_name)
        params = {k: v for k, v in q1.items() if k != 'q'}

        sc, results = self.execute_query(
            q=q1['q'], col_name=corpus_col, **params)

        if sc != 200:
            self.logger.error(
                f"-- -- Error executing query Q1. Aborting operation...")
            return

        # 4. Return -1 if thetas field is not found (it could happen that a document in a collection has not thetas representation since it was not keeped within the corpus used for training the model)
        if 'doctpc_' + model_name in results.docs[0].keys():
            resp = {'thetas': results.docs[0]['doctpc_' + model_name]}
        else:
            resp = {'thetas': -1}

        return resp, sc

    def do_Q2(self, corpus_col: str) -> Union[dict, int]:
        """
        Executes query Q2.

        Parameters
        ----------
        corpus_col: str
            Name of the corpus collection

        Returns
        -------
        json_object: dict
            JSON object with the metadata fields of the corpus collection in the form: {'metadata_fields': [field1, field2, ...]}
        sc: int
            The status code of the response
        """

        # 0. Convert corpus name to lowercase
        corpus_col = corpus_col.lower()

        # 1. Check that corpus_col is indeed a corpus collection
        if not self.check_is_corpus(corpus_col):
            return

        # 2. Execute query (to self.corpus_col)
        q2 = self.querier.customize_Q2(corpus_name=corpus_col)
        params = {k: v for k, v in q2.items() if k != 'q'}
        sc, results = self.execute_query(
            q=q2['q'], col_name=self.corpus_col, **params)

        if sc != 200:
            self.logger.error(
                f"-- -- Error executing query Q2. Aborting operation...")
            return

        # 3. Get Metadatadisplayed fields of corpus_col
        Metadatadisplayed, sc = self.get_corpus_MetadataDisplayed(corpus_col)
        if sc != 200:
            self.logger.error(
                f"-- -- Error getting Metadatadisplayed of {corpus_col}. Aborting operation...")
            return

        # 4. Filter metadata fields to be displayed in the NP
        # meta_fields = [field for field in results.docs[0]
        #               ['fields'] if field in Metadatadisplayed]

        return {'metadata_fields': Metadatadisplayed}, sc

    def do_Q3(self, col: str) -> Union[dict, int]:
        """Executes query Q3.

        Parameters
        ----------
        col : str
            Name of the collection

        Returns
        -------
        json_object : dict
            JSON object with the number of documents in the corpus collection
        sc : int
            The status code of the response
        """

        # 0. Convert collection name to lowercase
        col = col.lower()

        # 1. Check that col is either a corpus or a model collection
        if not self.check_is_corpus(col) and not self.check_is_model(col):
            return

        # 2. Execute query
        q3 = self.querier.customize_Q3()
        params = {k: v for k, v in q3.items() if k != 'q'}

        sc, results = self.execute_query(
            q=q3['q'], col_name=col, **params)

        # 3. Filter results
        if sc != 200:
            self.logger.error(
                f"-- -- Error executing query Q3. Aborting operation...")
            return

        return {'ndocs': int(results.hits)}, sc

    # def do_Q5(self,
    #           corpus_col: str,
    #           model_name: str,
    #           doc_id: str,
    #           start: str,
    #           rows: str) -> Union[dict, int]:
    #     """Executes query Q5.

    #     Parameters
    #     ----------
    #     corpus_col : str
    #         Name of the corpus collection
    #     model_name: str
    #         Name of the model to be used for the retrieval
    #     doc_id: str
    #         ID of the document whose similarity is going to be checked against all other documents in 'corpus_col'
    #      start: str
    #         Offset into the responses at which Solr should begin displaying content
    #     rows: str
    #         How many rows of responses are displayed at a time

    #     Returns
    #     -------
    #     json_object: dict
    #         JSON object with the results of the query.
    #     sc : int
    #         The status code of the response.
    #     """

    #     # 0. Convert corpus and model names to lowercase
    #     corpus_col = corpus_col.lower()
    #     model_name = model_name.lower()

    #     # 1. Check that corpus_col is indeed a corpus collection
    #     if not self.check_is_corpus(corpus_col):
    #         return

    #     # 2. Check that corpus_col has the model_name field
    #     if not self.check_corpus_has_model(corpus_col, model_name):
    #         return

    #     # 3. Execute Q1 to get thetas of document given by doc_id
    #     thetas_dict, sc = self.do_Q1(
    #         corpus_col=corpus_col, model_name=model_name, doc_id=doc_id)
    #     thetas = thetas_dict['thetas']

    #     # 4. Check that thetas are available on the document given by doc_id. If not, infer them
    #     if thetas == -1:
    #         # Get text (lemmas) of the document so its thetas can be inferred
    #         lemmas_dict, sc = self.do_Q15(
    #             corpus_col=corpus_col, doc_id=doc_id)
    #         lemmas = lemmas_dict['lemmas']

    #         inf_resp = self.inferencer.infer_doc(text_to_infer=lemmas,
    #                                              model_for_inference=model_name)
    #         if inf_resp.status_code != 200:
    #             self.logger.error(
    #                 f"-- -- Error attaining thetas from {lemmas} while executing query Q5. Aborting operation...")
    #             return

    #         thetas = inf_resp.results[0]['thetas']
    #         self.logger.info(
    #             f"-- -- Thetas attained in {inf_resp.time} seconds: {thetas}")

    #     # 4. Customize start and rows
    #     start, rows = self.custom_start_and_rows(start, rows, corpus_col)

    #     # 5. Execute query
    #     q5 = self.querier.customize_Q5(
    #         model_name=model_name, thetas=thetas,
    #         start=start, rows=rows)
    #     params = {k: v for k, v in q5.items() if k != 'q'}

    #     sc, results = self.execute_query(
    #         q=q5['q'], col_name=corpus_col, **params)

    #     if sc != 200:
    #         self.logger.error(
    #             f"-- -- Error executing query Q5. Aborting operation...")
    #         return

    #     # 6. Normalize scores
    #     for el in results.docs:
    #         el['score'] *= (100/(self.thetas_max_sum ^ 2))

    #     return results.docs, sc

    def do_Q6(self,
              corpus_col: str,
              doc_id: str) -> Union[dict, int]:
        """Executes query Q6.

        Parameters
        ----------
        corpus_col: str
            Name of the corpus collection
        doc_id: str
            ID of the document whose metadata is going to be retrieved

        Returns
        -------
        json_object: dict
            JSON object with the results of the query.
        sc : int
            The status code of the response.
        """

        # 0. Convert corpus name to lowercase
        corpus_col = corpus_col.lower()

        # 1. Check that corpus_col is indeed a corpus collection
        if not self.check_is_corpus(corpus_col):
            return

        # 2. Get meta fields
        meta_fields_dict, sc = self.do_Q2(corpus_col)
        meta_fields = ','.join(meta_fields_dict['metadata_fields'])

        self.logger.info("-- -- These are the meta fields: " + meta_fields)

        # 3. Execute query
        q6 = self.querier.customize_Q6(id=doc_id, meta_fields=meta_fields)
        params = {k: v for k, v in q6.items() if k != 'q'}

        sc, results = self.execute_query(
            q=q6['q'], col_name=corpus_col, **params)

        if sc != 200:
            self.logger.error(
                f"-- -- Error executing query Q6. Aborting operation...")
            return

        return results.docs, sc

    def do_Q7(self,
              corpus_col: str,
              string: str,
              start: str,
              rows: str) -> Union[dict, int]:
        """Executes query Q7.

        Parameters
        ----------
        corpus_col: str
            Name of the corpus collection
        string: str
            String to be searched in the title of the documents

        Returns
        -------
        json_object: dict
            JSON object with the results of the query.
        sc : int
            The status code of the response.
        """

        # 0. Convert corpus name to lowercase
        corpus_col = corpus_col.lower()

        # 1. Check that corpus_col is indeed a corpus collection
        if not self.check_is_corpus(corpus_col):
            return

        # 2. Get number of docs in the collection (it will be the maximum number of docs to be retireved) if rows is not specified
        if rows is None:
            q3 = self.querier.customize_Q3()
            params = {k: v for k, v in q3.items() if k != 'q'}

            sc, results = self.execute_query(
                q=q3['q'], col_name=corpus_col, **params)

            if sc != 200:
                self.logger.error(
                    f"-- -- Error executing query Q3. Aborting operation...")
                return
            rows = results.hits
        if start is None:
            start = str(0)

        # 2. Execute query
        q7 = self.querier.customize_Q7(
            title_field='SearcheableField',
            string=string,
            start=start,
            rows=rows)
        params = {k: v for k, v in q7.items() if k != 'q'}

        sc, results = self.execute_query(
            q=q7['q'], col_name=corpus_col, **params)

        if sc != 200:
            self.logger.error(
                f"-- -- Error executing query Q7. Aborting operation...")
            return

        return results.docs, sc

    def do_Q8(self,
              model_col: str,
              start: str,
              rows: str) -> Union[dict, int]:
        """Executes query Q8.

        Parameters
        ----------
        model_col: str
            Name of the model collection
        start: str
            Index of the first document to be retrieved
        rows: str
            Number of documents to be retrieved

        Returns
        -------
        json_object: dict
            JSON object with the results of the query.
        sc : int
            The status code of the response.
        """

        # 0. Convert model name to lowercase
        model_col = model_col.lower()

        # 1. Check that model_col is indeed a model collection
        if not self.check_is_model(model_col):
            return

        # 3. Customize start and rows
        start, rows = self.custom_start_and_rows(start, rows, model_col)

        # 4. Execute query
        q8 = self.querier.customize_Q8(start=start, rows=rows)
        params = {k: v for k, v in q8.items() if k != 'q'}

        sc, results = self.execute_query(
            q=q8['q'], col_name=model_col, **params)

        if sc != 200:
            self.logger.error(
                f"-- -- Error executing query Q8. Aborting operation...")
            return

        return results.docs, sc

    def do_Q9(self,
              corpus_col: str,
              model_name: str,
              topic_id: str,
              start: str,
              rows: str) -> Union[dict, int]:
        """Executes query Q9.

        Parameters
        ----------
        corpus_col: str
            Name of the corpus collection on which the query will be carried out
        model_name: str
            Name of the model collection on which the search will be based
        topic_id: str
            ID of the topic whose top-documents will be retrieved
        start: str
            Index of the first document to be retrieved
        rows: str
            Number of documents to be retrieved

        Returns
        -------
        json_object: dict
            JSON object with the results of the query.
        sc : int
            The status code of the response.
        """

        # 0. Convert corpus and model names to lowercase
        corpus_col = corpus_col.lower()
        model_name = model_name.lower()

        # 1. Check that corpus_col is indeed a corpus collection
        if not self.check_is_corpus(corpus_col):
            return

        # 2. Check that corpus_col has the model_name field
        if not self.check_corpus_has_model(corpus_col, model_name):
            return

        # 3. Customize start and rows
        start, rows = self.custom_start_and_rows(start, rows, corpus_col)
        # We limit the maximum number of results since they are top-documnts
        # If more results are needed pagination should be used

        if int(rows) > 100:
            rows = "100"

        # 5. Execute query
        q9 = self.querier.customize_Q9(
            model_name=model_name,
            topic_id=topic_id,
            start=start,
            rows=rows)
        params = {k: v for k, v in q9.items() if k != 'q'}

        sc, results = self.execute_query(
            q=q9['q'], col_name=corpus_col, **params)

        if sc != 200:
            self.logger.error(
                f"-- -- Error executing query Q9. Aborting operation...")
            return

        # 6. Return a dictionary with names more understandable to the end user
        proportion_key = "payload(doctpc_{},t{})".format(model_name, topic_id)
        for dict in results.docs:
            if proportion_key in dict.keys():
                dict["topic_relevance"] = dict.pop(proportion_key)*0.1
            dict["num_words_per_doc"] = dict.pop("nwords_per_doc")

        # 7. Get the topic's top words
        words, sc = self.do_Q10(
            model_col=model_name,
            start=topic_id,
            rows=1,
            only_id=False)
        if sc != 200:
            self.logger.error(
                f"-- -- Error executing query Q10 when using in Q9. Aborting operation...")
            return

        dict_bow, sc = self.do_Q18(
            corpus_col=corpus_col,
            ids=",".join([d['id'] for d in results.docs]),
            words=",".join(words[0]['tpc_descriptions'].split(", ")),
            start=start,
            rows=rows)

        # 7. Merge results
        def replace_payload_keys(dictionary):
            new_dict = {}
            for key, value in dictionary.items():
                match = re.match(r'payload\(bow,(\w+)\)', key)
                if match:
                    new_key = match.group(1)
                else:
                    new_key = key
                new_dict[new_key] = value
            return new_dict

        merged_tpcs = []
        for d1 in results.docs:
            id_value = d1['id']

            # Find the corresponding dictionary in dict2
            d2 = next(item for item in dict_bow if item["id"] == id_value)

            new_dict = {
                "id": id_value,
                "topic_relevance": d1.get("topic_relevance", 0),
                "num_words_per_doc": d1.get("num_words_per_doc", 0),
                "counts": replace_payload_keys({key: d2[key] for key in d2 if key.startswith("payload(bow,")})
            }

            merged_tpcs.append(new_dict)

        return merged_tpcs, sc

    def do_Q10(self,
               model_col: str,
               start: str,
               rows: str,
               only_id: bool) -> Union[dict, int]:
        """Executes query Q10.

        Parameters
        ----------
        model_col: str
            Name of the model collection whose information is being retrieved
        start: str
            Index of the first document to be retrieved
        rows: str
            Number of documents to be retrieved

        Returns
        -------
        json_object: dict
            JSON object with the results of the query.
        sc : int
            The status code of the response.
        """

        # 0. Convert model name to lowercase
        model_col = model_col.lower()

        # 1. Check that model_col is indeed a model collection
        if not self.check_is_model(model_col):
            return

        # 3. Customize start and rows
        start, rows = self.custom_start_and_rows(start, rows, model_col)

        # 4. Execute query
        q10 = self.querier.customize_Q10(
            start=start, rows=rows, only_id=only_id)
        params = {k: v for k, v in q10.items() if k != 'q'}

        sc, results = self.execute_query(
            q=q10['q'], col_name=model_col, **params)

        if sc != 200:
            self.logger.error(
                f"-- -- Error executing query Q10. Aborting operation...")
            return

        return results.docs, sc

    # def do_Q14(self,
    #            corpus_col: str,
    #            model_name: str,
    #            text_to_infer: str,
    #            start: str,
    #            rows: str) -> Union[dict, int]:
    #     """Executes query Q14.

    #     Parameters
    #     ----------
    #     corpus_col : str
    #         Name of the corpus collection
    #     model_name: str
    #         Name of the model to be used for the retrieval
    #     text_to_infer: str
    #         Text to be inferred
    #      start: str
    #         Offset into the responses at which Solr should begin displaying content
    #     rows: str
    #         How many rows of responses are displayed at a time

    #     Returns
    #     -------
    #     json_object: dict
    #         JSON object with the results of the query.
    #     sc : int
    #         The status code of the response.
    #     """

    #     # 0. Convert corpus and model names to lowercase
    #     corpus_col = corpus_col.lower()
    #     model_name = model_name.lower()

    #     # 1. Check that corpus_col is indeed a corpus collection
    #     if not self.check_is_corpus(corpus_col):
    #         return

    #     # 2. Check that corpus_col has the model_name field
    #     if not self.check_corpus_has_model(corpus_col, model_name):
    #         return

    #     # 3. Make request to Inferencer API to get thetas of text_to_infer
    #     inf_resp = self.inferencer.infer_doc(text_to_infer=text_to_infer,
    #                                          model_for_inference=model_name)
    #     if inf_resp.status_code != 200:
    #         self.logger.error(
    #             f"-- -- Error attaining thetas from {text_to_infer} while executing query Q14. Aborting operation...")
    #         return

    #     thetas = inf_resp.results[0]['thetas']
    #     self.logger.info(
    #         f"-- -- Thetas attained in {inf_resp.time} seconds: {thetas}")

    #     # 4. Customize start and rows
    #     start, rows = self.custom_start_and_rows(start, rows, corpus_col)

    #     # 5. Execute query
    #     q14 = self.querier.customize_Q14(
    #         model_name=model_name, thetas=thetas,
    #         start=start, rows=rows)
    #     params = {k: v for k, v in q14.items() if k != 'q'}

    #     sc, results = self.execute_query(
    #         q=q14['q'], col_name=corpus_col, **params)

    #     if sc != 200:
    #         self.logger.error(
    #             f"-- -- Error executing query Q14. Aborting operation...")
    #         return

    #     # 6. Normalize scores
    #     for el in results.docs:
    #         el['score'] *= (100/(self.thetas_max_sum ^ 2))

    #     return results.docs, sc

    def do_Q15(self,
               corpus_col: str,
               doc_id: str) -> Union[dict, int]:
        """Executes query Q15.

        Parameters
        ----------
        corpus_col : str
            Name of the corpus collection.
        id : str
            ID of the document to be retrieved.

        Returns
        -------
        lemmas: dict
            JSON object with the document's lemmas.
        sc : int
            The status code of the response.  
        """

        # 0. Convert corpus and model names to lowercase
        corpus_col = corpus_col.lower()

        # 1. Check that corpus_col is indeed a corpus collection
        if not self.check_is_corpus(corpus_col):
            return

        # 2. Execute query
        q15 = self.querier.customize_Q15(id=doc_id)
        params = {k: v for k, v in q15.items() if k != 'q'}

        sc, results = self.execute_query(
            q=q15['q'], col_name=corpus_col, **params)

        if sc != 200:
            self.logger.error(
                f"-- -- Error executing query Q15. Aborting operation...")
            return

        return {'lemmas': results.docs[0]['lemmas']}, sc

    def do_Q18(self,
               corpus_col: str,
               ids: str,
               words: str,
               start: str,
               rows: str
               ) -> Union[dict, int]:

        # 0. Convert corpus name to lowercase
        corpus_col = corpus_col.lower()

        # 1. Check that corpus_col is indeed a corpus collection
        if not self.check_is_corpus(corpus_col):
            return

        # 2. Execute query
        start, rows = self.custom_start_and_rows(start, rows, corpus_col)

        q18 = self.querier.customize_Q18(
            ids=ids.split(","),
            words=words.split(","),
            start=start,
            rows=rows)
        params = {k: v for k, v in q18.items() if k != 'q'}

        sc, results = self.execute_query(
            q=q18['q'], col_name=corpus_col, **params)

        if sc != 200:
            self.logger.error(
                f"-- -- Error executing query Q18. Aborting operation...")
            return

        return results.docs, sc
    
    def do_Q20(
        self,
        corpus_col:str,
        model_name:str,
        search_word:str,
        start:int,
        rows:int,
        embedding_model:str = "word2vec",
        lang:str = "es",
    ) -> Union[dict,int]:
        
        # 0. Convert corpus and model names to lowercase
        corpus_col = corpus_col.lower()
        model_name = model_name.lower()
        
        # 1. Check that corpus_col is indeed a corpus collection
        if not self.check_is_corpus(corpus_col):
            return
        
        # 2. Check that corpus_col has the model_name field
        if not self.check_corpus_has_model(corpus_col, model_name):
            return

        # 3. Lemmatize and get embedding from search_word
        resp = self.nptooler.get_word_embedding(
            word_to_embed=search_word,
            embedding_model=embedding_model,
            model_for_embedding=model_name,
            lang=lang
        )

        if resp.status_code != 200:
            self.logger.error(
                f"-- -- Error getting embeddings from {search_word} while executing query Q20. Aborting operation...")
            return
        
        # 4. Calculate cosine similarity between the embedding of search_word and the embeddings of the documents in the corpus
        distance = "cosine"
        
        # 5. Find the topic that is most similar to the search_word
        
        # 6. Return the id of the topic and the similarity score