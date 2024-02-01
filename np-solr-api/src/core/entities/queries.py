"""
This module defines a class with the EWB-specific queries used to interact with Solr.


Author: Lorena Calvo-Bartolomé
Date: 19/04/2023
"""


class Queries(object):

    def __init__(self) -> None:

        # ================================================================
        # # Q1: getThetasDocById  ##################################################################
        # # Get document-topic distribution of a selected document in a
        # # corpus collection
        # http://localhost:8983/solr/{col}/select?fl=doctpc_{model}&q=id:{id}
        # ================================================================
        self.Q1 = {
            'q': 'id:{}',
            'fl': 'doctpc_{}',
        }

        # ================================================================
        # # Q2: getCorpusMetadataFields  ##################################################################
        # # Get the name of the metadata fields available for
        # a specific corpus collection (not all corpus have
        # the same metadata available)
        # http://localhost:8983/solr/#/Corpora/query?q=corpus_name:Cordis&q.op=OR&indent=true&fl=fields&useParams=
        # ================================================================
        self.Q2 = {
            'q': 'corpus_name:{}',
            'fl': 'fields',
        }

        # ================================================================
        # # Q3: getNrDocsColl ##################################################################
        # # Get number of documents in a collection
        # http://localhost:8983/solr/{col}/select?q=*:*&wt=json&rows=0
        # ================================================================
        self.Q3 = {
            'q': '*:*',
            'rows': '0',
        }

        # ================================================================
        # # Q5: getDocsWithHighSimWithDocByid
        # ################################################################
        # # Retrieve documents that have a high semantic relationship with
        # # a selected document
        # ---------------------------------------------------------------
        # Previous steps:
        # ---------------------------------------------------------------
        # 1. Get thetas of selected documents
        # 2. Parse thetas in Q1
        # 3. Execute Q4
        # ================================================================
        self.Q5 = {
            'q': "{{!vp f=doctpc_{} vector=\"{}\"}}",
            'fl': "id,score",
            'start': '{}',
            'rows': '{}'
        }

        # ================================================================
        # # Q6: getMetadataDocById
        # ################################################################
        # # Get metadata of a selected document in a corpus collection
        # ---------------------------------------------------------------
        # Previous steps:
        # ---------------------------------------------------------------
        # 1. Get metadata fields of that corpus collection with Q2
        # 2. Parse metadata in Q6
        # 3. Execute Q6
        # ================================================================
        self.Q6 = {
            'q': 'id:{}',
            'fl': '{}'
        }

        # ================================================================
        # # Q7: getDocsWithString
        # ################################################################
        # # Given a corpus collection, it retrieves the ids of the documents whose title contains such a string
        # http://localhost:8983/solr/#/{collection}/query?q=title:{string}&q.op=OR&indent=true&useParams=
        # ================================================================
        self.Q7 = {
            'q': '{}:{}',
            'fl': 'id',
            'start': '{}',
            'rows': '{}'
        }

        # ================================================================
        # # Q8: getTopicsLabels
        # ################################################################
        # # Get the label associated to each of the topics in a given model
        # http://localhost:8983/solr/{model}/select?fl=id%2C%20tpc_labels&indent=true&q.op=OR&q=*%3A*&useParams=
        # ================================================================
        self.Q8 = {
            'q': '*:*',
            'fl': 'id,tpc_labels',
            'start': '{}',
            'rows': '{}'
        }

        # ================================================================
        # # Q9: getTopicTopDocs
        # ################################################################
        # # Get the top documents for a given topic in a model collection
        # http://localhost:8983/solr/cordis/select?indent=true&q.op=OR&q=%7B!term%20f%3D{model}%7Dt{topic_id}&useParams=
        # http://localhost:8983/solr/#/{corpus_collection}/query?q=*:*&q.op=OR&indent=true&fl=doctpc_{model_name},%20nwords_per_doc&sort=payload(doctpc_{model_name},t{topic_id})%20desc,%20nwords_per_doc%20desc&useParams=
        #http://localhost:8983/solr/#/np_all/query?q=*:*&q.op=OR&indent=true&fl=doctpc_np_5tpcs,%20nwords_per_doc&sort=payload(doctpc_np_5tpcs,t0)%20desc,%20nwords_per_doc%20desc&useParams=
        # ================================================================
        self.Q9 = {
            'q': '*:*',
            'sort': 'payload(doctpc_{},t{}) desc, nwords_per_doc desc',
            'fl': 'payload(doctpc_{},t{}), nwords_per_doc, id',
            'start': '{}',
            'rows': '{}'
        }#doctpc_{}

        # ================================================================
        # # Q10: getModelInfo
        # ################################################################
        # # Get the information (chemical description, label, statistics,
        # top docs, etc.) associated to each topic in a model collection
        # ================================================================
        self.Q10 = {
            'q': '*:*',
            'fl': 'id,alphas,top_words_betas,topic_entropy,topic_coherence,ndocs_active,tpc_descriptions,tpc_labels,coords',
            'start': '{}',
            'rows': '{}'
        }

        # ================================================================
        # # Q14: getDocsSimilarToFreeText
        # ################################################################
        # # Get documents that are semantically similar to a free text
        # according to a given model
        # ================================================================
        self.Q14 = self.Q5

        # ================================================================
        # # Q15: getLemmasDocById  ##################################################################
        # # Get lemmas of a selected document in a corpus collection
        # http://localhost:8983/solr/{col}/select?fl=lemmas&q=id:{id}
        #http://localhost:8983/solr/np_all/select?fl=lemmas&q=id:505302
        # ================================================================
        self.Q15 = {
            'q': 'id:{}',
            'fl': 'lemmas',
            'start': '{}',
            'rows': '{}'
        }
        
        self.Q18 = {
            'q': 'id:{}',
            'fl': 'payload(bow,{})',
            'start': '{}',
            'rows': '{}'
        }

        # If adding a new one, start numberation at 20
        
        
    def customize_Q1(self,
                     id: str,
                     model_name: str) -> dict:
        """Customizes query Q1 'getThetasDocById'.

        Parameters
        ----------
        id: str
            Document id.
        model_name: str
            Name of the topic model whose topic distribution is to be retrieved.

        Returns
        -------
        custom_q1: dict
            Customized query Q1.
        """

        custom_q1 = {
            'q': self.Q1['q'].format(id),
            'fl': self.Q1['fl'].format(model_name),
        }
        return custom_q1

    def customize_Q2(self,
                     corpus_name: str) -> dict:
        """Customizes query Q2 'getCorpusMetadataFields'

        Parameters
        ----------
        corpus_name: str
            Name of the corpus collection whose metadata fields are to be retrieved.

        Returns
        -------
        custom_q2: dict
            Customized query Q2.
        """

        custom_q2 = {
            'q': self.Q2['q'].format(corpus_name),
            'fl': self.Q2['fl'],
        }

        return custom_q2

    def customize_Q3(self) -> dict:
        """Customizes query Q3 'getNrDocsColl'

        Returns
        -------
        self.Q3: dict
            The query Q3 (no customization is needed).
        """

        return self.Q3

    def customize_Q5(self,
                     model_name: str,
                     thetas: str,
                     start: str,
                     rows: str) -> dict:
        """Customizes query Q5 'getDocsWithHighSimWithDocByid'

        Parameters
        ----------
        model_name: str
            Name of the topic model whose topic distribution is to be retrieved.
        thetas: str
            Topic distribution of the selected document.
        start: str
            Start value.
        rows: str
            Number of rows to retrieve.

        Returns
        -------
        custom_q5: dict
            Customized query Q5.
        """

        custom_q5 = {
            'q': self.Q5['q'].format(model_name, thetas),
            'fl': self.Q5['fl'].format(model_name),
            'start': self.Q5['start'].format(start),
            'rows': self.Q5['rows'].format(rows),
        }
        return custom_q5

    def customize_Q6(self,
                     id: str,
                     meta_fields: str) -> dict:
        """Customizes query Q6 'getMetadataDocById'


        Parameters
        ----------
        id: str
            Document id.
        meta_fields: str
            Metadata fields of the corpus collection to be retrieved.

        Returns
        -------
        custom_q6: dict
            Customized query Q6.
        """

        custom_q6 = {
            'q': self.Q6['q'].format(id),
            'fl': self.Q6['fl'].format(meta_fields)
        }

        return custom_q6

    def customize_Q7(self,
                     title_field: str,
                     string: str,
                     start: str,
                     rows: str) -> dict:
        """Customizes query Q7 'getDocsWithString'

        Parameters
        ----------
        title_field: str
            Title field of the corpus collection.
        string: str
            String to be searched in the title field.
        start: str
            Start value.
        rows: str
            Number of rows to retrieve.

        Returns
        -------
        custom_q7: dict
            Customized query Q7.
        """

        custom_q7 = {
            'q': self.Q7['q'].format(title_field, string),
            'fl': self.Q7['fl'],
            'start': self.Q7['start'].format(start),
            'rows': self.Q7['rows'].format(rows)
        }

        return custom_q7

    def customize_Q8(self,
                     start: str,
                     rows: str) -> dict:
        """Customizes query Q8 'getTopicsLabels'

        Parameters
        ----------
        rows: str
            Number of rows to retrieve.
        start: str
            Start value.

        Returns
        -------
        self.Q8: dict
            The query Q8
        """

        custom_q8 = {
            'q': self.Q8['q'],
            'fl': self.Q8['fl'],
            'start': self.Q8['start'].format(start),
            'rows': self.Q8['rows'].format(rows),
        }

        return custom_q8

    def customize_Q9(self,
                     model_name: str,
                     topic_id: str,
                     start: str,
                     rows: str) -> dict:
        """Customizes query Q9 'getDocsByTopic'

        Parameters
        ----------
        model_name: str
            Name of the topic model whose topic distribution is going to be used for retreving the top documents for the topic given by 'topic'.
        topic_id: str
            Topic number.
        start: str
            Start value.
        rows: str
            Number of rows to retrieve.

        Returns
        -------
        custom_q9: dict
            Customized query Q9.
        """

        custom_q9 = {
            'q': self.Q9['q'],
            'sort': self.Q9['sort'].format(model_name, topic_id),
            'fl': self.Q9['fl'].format(model_name, topic_id),
            'start': self.Q9['start'].format(start),
            'rows': self.Q9['rows'].format(rows),
        }
        
        return custom_q9

    def customize_Q10(self,
                      start: str,
                      rows: str,
                      only_id: bool) -> dict:
        """Customizes query Q10 'getModelInfo'

        Parameters
        ----------
        start: str
            Start value.
        rows: str

        Returns
        -------
        custom_q10: dict
            Customized query Q10.
        """

        if only_id:
            custom_q10 = {
                'q': self.Q10['q'],
                'fl': 'id',
                'start': self.Q10['start'].format(start),
                'rows': self.Q10['rows'].format(rows),
            }
        else:
            custom_q10 = {
                'q': self.Q10['q'],
                'fl': self.Q10['fl'],
                'start': self.Q10['start'].format(start),
                'rows': self.Q10['rows'].format(rows),
            }

        return custom_q10

    def customize_Q14(self,
                      model_name: str,
                      thetas: str,
                      start: str,
                      rows: str) -> dict:
        """Customizes query Q14 'getDocsSimilarToFreeText'

        Parameters
        ----------
        model_name: str
            Name of the topic model whose topic distribution is to be retrieved.
        thetas: str
            Topic distribution of the user's free text.
        start: str
            Start value.
        rows: str
            Number of rows to retrieve.

        Returns
        -------
        custom_q14: dict
            Customized query Q14.
        """

        custom_q14 = {
            'q': self.Q14['q'].format(model_name, thetas),
            'fl': self.Q14['fl'].format(model_name),
            'start': self.Q14['start'].format(start),
            'rows': self.Q14['rows'].format(rows),
        }
        return custom_q14

    def customize_Q15(self,
                      id: str) -> dict:
        """Customizes query Q15 'getLemmasDocById'.

        Parameters
        ----------
        id: str
            Document id.

        Returns
        -------
        custom_q15: dict
            Customized query Q15.
        """

        custom_q15 = {
            'q': self.Q15['q'].format(id),
            'fl': self.Q15['fl'],
        }
        return custom_q15

    def customize_Q18(self,
                      ids: str,
                      words: str,
                      start:str,
                      rows: str) -> dict:
    
        
        custom_q18 = {
            'q':  self.Q18['q'].format(' & id:'.join(ids)),
            'fl': 'id, ' + ', '.join(self.Q18['fl'].format(word) for word in words),
            'start': self.Q18['start'].format(start),
            'rows': self.Q18['rows'].format(rows),
        }

        return custom_q18