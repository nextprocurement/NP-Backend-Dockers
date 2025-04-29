from locust import HttpUser, task, between, constant
import os
from dotenv import load_dotenv

class QueryEndpointsUser(HttpUser):
    
    load_dotenv()
    
    wait_time =  constant(5) #between(1, 3) #constant(0.1)

    base_params = {
        "corpus_collection": os.getenv("CORPUS_COLLECTION", "test_corpus"),
        "model_collection": os.getenv("MODEL_COLLECTION", "test_model"),
        "model_name": os.getenv("MODEL_NAME", "test_model"),
        "collection": os.getenv("COLLECTION", "test_collection"),
        "doc_id": os.getenv("DOC_ID", "1"),
        "word": os.getenv("WORD", "example"),
        "free_text": os.getenv("FREE_TEXT", "this is a test text"),
        "text_to_infer": os.getenv("TEXT_TO_INFER", "test text"),
        "string": os.getenv("STRING", "search term"),
        "topic_id": os.getenv("TOPIC_ID", "0"),
        "cpv": os.getenv("CPV_CODE", "03"),
    }

    @task
    def get_corpus_metadata_fields(self):
        self.client.get("/queries/getCorpusMetadataFields/", params={"corpus_collection": self.base_params["corpus_collection"]})

    @task
    def get_docs_related_to_word(self):
        self.client.get("/queries/getDocsRelatedToWord/", params={
            "corpus_collection": self.base_params["corpus_collection"],
            "model_collection": self.base_params["model_collection"],
            "word": self.base_params["word"]
        })

    @task
    def get_docs_similar_to_free_text_emb(self):
        self.client.get("/queries/getDocsSimilarToFreeTextEmb/", params={
            "corpus_collection": self.base_params["corpus_collection"],
            "free_text": self.base_params["free_text"]
        })

    @task
    def get_docs_similar_to_free_text_tm(self):
        self.client.get("/queries/getDocsSimilarToFreeTextTM/", params={
            "corpus_collection": self.base_params["corpus_collection"],
            "model_name": self.base_params["model_name"],
            "text_to_infer": self.base_params["text_to_infer"]
        })

    @task
    def get_docs_with_high_sim_with_doc_byid(self):
        self.client.get("/queries/getDocsWithHighSimWithDocByid/", params={
            "corpus_collection": self.base_params["corpus_collection"],
            "model_name": self.base_params["model_name"],
            "doc_id": self.base_params["doc_id"]
        })

    @task
    def get_docs_with_string(self):
        self.client.get("/queries/getDocsWithString/", params={
            "corpus_collection": self.base_params["corpus_collection"],
            "string": self.base_params["string"]
        })

    @task
    def get_metadata_doc_by_id(self):
        self.client.get("/queries/getMetadataDocById/", params={
            "corpus_collection": self.base_params["corpus_collection"],
            "doc_id": self.base_params["doc_id"]
        })

    @task
    def get_model_info(self):
        self.client.get("/queries/getModelInfo/", params={
            "model_collection": self.base_params["model_collection"]
        })

    @task
    def get_nr_docs_coll(self):
        self.client.get("/queries/getNrDocsColl/", params={
            "collection": self.base_params["collection"]
        })

    @task
    def get_thetas_doc_by_id(self):
        self.client.get("/queries/getThetasDocById/", params={
            "corpus_collection": self.base_params["corpus_collection"],
            "doc_id": self.base_params["doc_id"],
            "model_name": self.base_params["model_name"]
        })

    @task
    def get_topic_top_docs(self):
        self.client.get("/queries/getTopicTopDocs/", params={
            "corpus_collection": self.base_params["corpus_collection"],
            "model_name": self.base_params["model_name"],
            "topic_id": self.base_params["topic_id"]
        })

    @task
    def get_topics_labels(self):
        self.client.get("/queries/getTopicsLabels/", params={
            "model_collection": self.base_params["model_collection"]
        })

    @task
    def infer_topic_information(self):
        self.client.get("/queries/inferTopicInformation/", params={
            "text_to_infer": self.base_params["text_to_infer"],
            "cpv": self.base_params["cpv"],
        })
