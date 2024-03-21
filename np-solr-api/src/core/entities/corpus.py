"""
This module is a class implementation to manage and hold all the information associated with a logical corpus.

Author: Lorena Calvo-Bartolomé
Date: 27/03/2023
Modifed: 24/01/2024 (Updated for NP-Solr-Service (NextProcurement Proyect))
"""

import configparser
import json
from typing import List
from gensim.corpora import Dictionary
import pathlib
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from src.core.entities.utils import (convert_datetime_to_strftime,
                                     parseTimeINSTANT)


class Corpus(object):
    """
    A class to manage and hold all the information associated with a logical corpus.
    """

    def __init__(self,
                 path_to_raw: pathlib.Path,
                 logger=None,
                 config_file: str = "/config/config.cf") -> None:
        """Init method.

        Parameters
        ----------
        path_to_raw: pathlib.Path
            Path the raw corpus file.
        logger : logging.Logger
            The logger object to log messages and errors.
        config_file: str
            Path to the configuration file.
        """

        if logger:
            self._logger = logger
        else:
            import logging
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('Entity Corpus')

        if not path_to_raw.exists():
            self._logger.error(
                f"Path to raw data {path_to_raw} does not exist."
            )
        self.path_to_raw = path_to_raw
        self.name = path_to_raw.stem.lower()
        self.fields = None

        # Read configuration from config file
        cf = configparser.ConfigParser()
        cf.read(config_file)
        self._logger.info(f"Sections {cf.sections()}")
        if self.name + "-config" in cf.sections():
            section = self.name + "-config"
        elif self.name + "-config" in cf.sections():
            section = self.name + "-config"
        else:
            self._logger.error(
                f"Corpus configuration {self.name} not found in config file.")
        self.id_field = cf.get(section, "id_field")
        self.title_field = cf.get(section, "title_field")
        self.date_field = cf.get(section, "date_field")
        self.MetadataDisplayed = cf.get(
            section, "MetadataDisplayed").split(",")
        self.SearcheableField = cf.get(section, "SearcheableField").split(",")
        if self.title_field in self.SearcheableField:
            self.SearcheableField.remove(self.title_field)
            self.SearcheableField.append("title")
        if self.date_field in self.SearcheableField:
            self.SearcheableField.remove(self.date_field)
            self.SearcheableField.append("date")

        return

    def get_docs_raw_info(self) -> List[dict]:
        """Extracts the information contained in the parquet file associated to the logical corpus and transforms into a list of dictionaries.

        Returns:
        --------
        json_lst: list[dict]
            A list of dictionaries containing information about the corpus.
        """

        ddf = dd.read_parquet(self.path_to_raw).fillna("")
        self._logger.info(ddf.head())

        # If the id_field is in the SearcheableField, remove it and add the id field (new name for the id_field)
        if self.id_field in self.SearcheableField:
            self.SearcheableField.remove(self.id_field)
            self.SearcheableField.append("id")
        self._logger.info(f"SearcheableField {self.SearcheableField}")

        # Rename id-field to id, title-field to title and date-field to date
        ddf = ddf.rename(
            columns={
                self.id_field: "id",
                self.title_field: "title",
                self.date_field: "date"})

        with ProgressBar():
            df = ddf.compute(scheduler='processes')

        self._logger.info(df.columns)

        # Get number of words per document based on the lemmas column
        # NOTE: Document whose lemmas are empty will have a length of 0
        df["nwords_per_doc"] = df["lemmas"].apply(lambda x: len(x.split()))

        # Get BoW representation
        # We dont read from the gensim dictionary that will be associated with the tm models trained on the corpus since we want to have the bow for all the documents, not only those kept after filering extremes in the dictionary during the construction of the logical corpus
        # check none values: df[df.isna()]
        df['lemmas_'] = df['lemmas'].apply(
            lambda x: x.split() if isinstance(x, str) else [])
        dictionary = Dictionary()
        df['bow'] = df['lemmas_'].apply(
            lambda x: dictionary.doc2bow(x, allow_update=True) if x else [])
        df['bow'] = df['bow'].apply(
            lambda x: [(dictionary[id], count) for id, count in x] if x else [])
        df['bow'] = df['bow'].apply(lambda x: None if len(x) == 0 else x)
        df = df.drop(['lemmas_'], axis=1)
        df['bow'] = df['bow'].apply(lambda x: ' '.join(
            [f'{word}|{count}' for word, count in x]).rstrip() if x else None)
        
        # Ger embeddings of the documents
        def get_str_embeddings(vector):
            repr = " ".join(
                [f"e{idx}|{val}" for idx, val in enumerate(vector.split())]).rstrip()

            return repr
        
        df["embeddings"] = df["embeddings"].apply(get_str_embeddings)

        # Convert dates information to the format required by Solr ( ISO_INSTANT, The ISO instant formatter that formats or parses an instant in UTC, such as '2011-12-03T10:15:30Z')
        df, cols = convert_datetime_to_strftime(df)
        df[cols] = df[cols].applymap(parseTimeINSTANT)

        # Create SearcheableField by concatenating all the fields that are marked as SearcheableField in the config file
        df['SearcheableField'] = df[self.SearcheableField].apply(
            lambda x: ' '.join(x.astype(str)), axis=1)

        # Save corpus fields
        self.fields = df.columns.tolist()

        json_str = df.to_json(orient='records')
        json_lst = json.loads(json_str)
        
        return json_lst

    def get_corpora_update(
        self,
        id: int
    ) -> List[dict]:
        """Creates the json to update the 'corpora' collection in Solr with the new logical corpus information.
        """

        # TODO: Update
        fields_dict = [{"id": id,
                        "corpus_name": self.name,
                        "corpus_path": self.path_to_raw.as_posix(),
                        "fields": self.fields,
                        "MetadataDisplayed": self.MetadataDisplayed,
                        "SearcheableFields": self.SearcheableField}]

        return fields_dict

    def get_corpora_SearcheableField_update(
        self,
        id: int,
        field_update: list,
        action: str
    ) -> List[dict]:

        json_lst = [{"id": id,
                    "SearcheableFields": {action: field_update},
                     }]

        return json_lst

    def get_corpus_SearcheableField_update(
        self,
        new_SearcheableFields: str,
        action: str
    ):

        ddf = dd.read_parquet(self.path_to_raw).fillna("")

        # Rename id-field to id, title-field to title and date-field to date
        ddf = ddf.rename(
            columns={self.id_field: "id",
                     self.title_field: "title",
                     self.date_field: "date"})

        with ProgressBar():
            df = ddf.compute(scheduler='processes')

        if action == "add":
            new_SearcheableFields = [
                el for el in new_SearcheableFields if el not in self.SearcheableField]
            if self.title_field in new_SearcheableFields:
                new_SearcheableFields.remove(self.title_field)
                new_SearcheableFields.append("title")
            if self.date_field in new_SearcheableFields:
                new_SearcheableFields.remove(self.date_field)
                new_SearcheableFields.append("date")
            new_SearcheableFields = list(
                set(new_SearcheableFields + self.SearcheableField))
        elif action == "remove":
            if self.title_field in new_SearcheableFields:
                new_SearcheableFields.remove(self.title_field)
                new_SearcheableFields.append("title")
            if self.date_field in new_SearcheableFields:
                new_SearcheableFields.remove(self.date_field)
                new_SearcheableFields.append("date")
            new_SearcheableFields = [
                el for el in self.SearcheableField if el not in new_SearcheableFields]

        df['SearcheableField'] = df[new_SearcheableFields].apply(
            lambda x: ' '.join(x.astype(str)), axis=1)

        not_keeps_cols = [el for el in df.columns.tolist() if el not in [
            "id", "SearcheableField"]]
        df = df.drop(not_keeps_cols, axis=1)

        # Create json from dataframe
        json_str = df.to_json(orient='records')
        json_lst = json.loads(json_str)

        new_list = []
        for d in json_lst:
            d["SearcheableField"] = {"set": d["SearcheableField"]}
            new_list.append(d)

        return new_list, new_SearcheableFields


# if __name__ == '__main__':
#    corpus = Corpus(pathlib.Path("/Users/lbartolome/Documents/GitHub/EWB/data/source/Cordis.json"))
#    json_lst = corpus.get_docs_raw_info()
#    new_list = corpus.get_corpus_SearcheableField_update(["Call"], action="add")
#    fields_dict = corpus.get_corpora_update(1)
