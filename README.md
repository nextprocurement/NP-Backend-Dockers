# NP-Backend-Dockers

## Overview

``NP-Backend-Dockers`` powers the backend infrastructure of an application designed for efficient indexing, analysis, and retrieval of textual data leveraging a Solr engine. It also seamlessly integrates additional services like topic-based inference and classification in a multi-container setup, enhancing user functionality.

This multi-container application is orchestrated using a docker-compose script, connecting all services through the `np-net` network.

![Python Dockers](https://github.com/nextprocurement/NP-Backend-Dockers/blob/main/static/Images/np_1.png)

## ⚙️ Steps for deployment

1. Clone the repository:

    ```bash
    git clone https://github.com/nextprocurement/NP-Backend-Dockers.git
    ```

2. Initialize submodules:

    ```bash
    git submodule init
    ```

3. Update content of submodules:

    ```bash
    git submodule update
    ```

4. Create folder ``data`` and copy model information into it. It should looks as follows:

    ![Data folder structure](https://github.com/nextprocurement/NP-Backend-Dockers/blob/main/static/Images/np_data_folder_structure.png)

    An example of what should be in the ``data`` folder is available here.

5. Create a network that you can use and replace the ``ml4ds2_net`` in the ``docker-compose.yml`` with the name of your new network:

    ```docker
    networks:
    np-net:
        name: ml4ds2_net
        external: true
    ```

6. Start the services:

    ```bash
    docker-compose up -d
    ```

7. Check all the services are working:

    ```bash
    docker ps
    ```

8. Check that the `NP-solr-dist-plugin` plugin have been mounted properly in Solr. For this, go to Solr (it should be available at [http://your_server_name:8984/solr/#/](http://your_server_name:8984/solr/#/) and create a `test` collection from the following view using the ``np_config`` config set. If everything worked fine, delete the test collection.

    ![Test Solr](https://github.com/nextprocurement/NP-Backend-Dockers/blob/main/static/Images/np_data_folder_structure.png)

    > If you encounter any problems, write an email to [lcalvo@pa.uc3m.es](mailto:lcalvo@pa.uc3m.es).


## 🧩 Main components

### np-solr-api

This RESTful API serves as an entry point for indexing and performing a series of queries to retrieve information from the Solr search engine. It essentially acts as a Python wrapper that encapsulates Solr's fundamental functionalities within a Flask-based framework.

It has dependencies on the ``np_solr`` and ``np-tools`` services and requires access to the following mounted volumes:

- ``./data/source``
- ``./np_config``

### 🔎 Queries

Currently available queries are the following:

| Endpoint                      | Internal name | Returns                                                                                                                              | Format                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ----------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| getThetasDocById              | Q1            | Document-topic distribution of a selected document in a corpus collection.                                                           | {"thetas": thetas}                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| getCorpusMetadataFields       | Q2            | Name of the metadata fields available for a specific corpus collection.                                                              | {"metadata_fields": meta_fields}                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| getNrDocsColl                 | Q3            | Number of documents in a collection.                                                                                                 | {"ndocs": ndocs}                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| getDocsWithHighSimWithDocByid | Q5            | Documents that have a high semantic relationship with a selected document based on the document-topic distribution of a given model. | [{"id": id1, "score": score1 }, {"id": id2, "score": score2 }, ...]                                                                                                                                                                                                                                                                                                                                                                                                            |
| getMetadataDocById            | Q6            | Metadata of a selected document in a corpus collection.                                                                              | {"metadata1": metadata1, "metadata2": metadata2, "metadata3": metadata3, ... }                                                                                                                                                                                                                                                                                                                                                                                                 |
| getDocsWithString             | Q7            | Ids of the documents in a corpus collections in which a given field contains a given string.                                         | [{"id": id1}, {"id": id2}, ...]                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| getTopicsLabels               | Q8            | Label associated to each of the topics in a given topic model.                                                                       | [{"id": id1, "tpc_labels": label1 }, {"id": id2, "tpc_labels": label2}, ...]                                                                                                                                                                                                                                                                                                                                                                                                   |
| getTopicTopDocs               | Q9            | Top documents for a given topic.                                                                                                     | [{"id": id1, "thetas": thetas1, "num_words_per_doc": num_words_per_doc1 }, {"id": id2, thetas": thetas2, "num_words_per_doc": num_words_per_doc2}, ...]                                                                                                                                                                                                                                                                                                                        |
| getModelInfo                  | Q10           | Information (chemical description, label, statistics, top docs, etc.) associated to each topic in a model collection.                | [{"id":id1, "betas": betas1, "alphas": alphas1, "topic_entropy":entropies1, "topic_coherence":cohrs1, "ndocs_active":active1, "tpc_descriptions":desc1, "tpc_labels":labels1, "coords":coords1, "top_words_betas":top_words_betas1,}, {"id":id2, "betas": betas2, "alphas": alphas2, "topic_entropy":entropies2, "topic_coherence":cohrs2, "ndocs_active":active2, "tpc_descriptions":desc2, "tpc_labels":labels2, "coords":coords2, "top_words_betas":top_words_betas2}, ...] |
| getDocsSimilarToFreeTextTM    | Q14           | Documents that are semantically similar to a free text based on the document-topic distribution of a given model.                    | [{"id": id1, "score": score1 }, {"id": id2, "score": score2 }, ...]                                                                                                                                                                                                                                                                                                                                                                                                            |
| getDocsRelatedToWord          | Q20           | Documents related to a word according to a given topic model.                                                                        | {"topic_id":"topic_id_value","topic_str":"topic_string_value","similarity_score":similarity_score_value,"docs":[{"id":"doc_id_value","topic_relevance":topic_relevance_value,"num_words_per_doc":num_words_value},{"id":"doc_id_value","topic_relevance":topic_relevance_value,"num_words_per_doc":num_words_value},...]}                                                                                                                                                      |
| getDocsSimilarToFreeTextEmb   | Q21           | Documents that are semantically similar to a given free text using BERT embeddings.                                                  | [{"id": id1, "title": title1, "score": score1 }, {"id": id2, "title": title2, "score": score2 }, ...]                                                                                                                                                                                                                                                                                                                                                                          |

### np-solr

This service deploys an instance of the Solr search engine using the official Solr image from Docker Hub and relying on the zoo service. It mounts several volumes, including:

- The **Solr data directory** (`./db/data/solr:/var/solr`) for data persistence.
- The **custom Solr plugin** [`NP-solr-dist-plugin`](https://github.com/nextprocurement/NP-solr-dist-plugin), which provides a plugin for performing distance calculations within Solr efficiently.
- The **Solr configuration directory** (`./solr_config:/opt/solr/server/solr`) to access specific Solr schemas for the NextProcurement project data.

### np-solr-initializer

This service is temporary and serves the sole purpose of initializing the mounted volume ``/db/data`` with the necessary permissions required by Solr.

### np-zoo

This service runs Zookeeper, which is essential for Solr to coordinate cluster nodes. It employs the official zookeeper image and mounts two volumes for data and logs.

### np-solr-config

This service handles Solr configuration. It is constructed using the Dockerfile located in the ``solr-config`` directory. This service has dependencies on the Solr and zoo services and mounts the Docker socket and the ``bash_scripts`` directory, which contains a script for initializing the Solr configuration for the NextProcuremetn proyect.

### np-tools

This service deploys a RESTful API with a series of auxiliary endpoints. Right now it contains enpoints to:

- Retrieve embeddings for a given document or word based on a Word2Vec (a precalculated Word2Vec model is assumed) or SBERT.
- Retrieve document-topic representation of a given document based on a trained topic model.
- Retrieve the lemmas of a given document.

It has the same mounted volumes as the ``np-solr-api`` service.

## Requirements

**Python requirements files** available within each "service" folder.

> *Requirements are directly installed in their respective services at the building-up time.*
