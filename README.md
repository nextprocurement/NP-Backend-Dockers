# NP-Backend-Dockers

## Overview

``NP-Backend-Dockers`` drives the backend infrastructure of an application designed for efficient data indexing and analysis. It incorporates outputs from training topic models into a Solr engine, enabling streamlined operations. Accessible via a Swagger page, the backend, powered by a Python-based RESTful application, facilitates basic indexing tasks and queries. Moreover, it supports seamless integration of additional services, such as topic-modeling-based inference and classification, within the multi-container environment, expanding functionality for users.

This multi-container application is orchestrated using a docker-compose script, connecting all services through the `np-net` network.


![Python Dockers](https://github.com/Nemesis1303/NP-Backend-Dockers/blob/main/static/Images/np_1.png)

## Main components

### NP Solr API

RESTful API that utilizes the Solr search engine for data storage and retrieval. It relies on the following services:

1. **np-solr-api**: This Docker image encapsulates the NP Solr API, comprising a Python-based RESTful API. This API connects to a Python-based wrapper encapsulating Solr's fundamental functionalities.

    It has dependencies on the Solr service (``np_solr``) and requires access to the following mounted volumes:
    - ``./data/source``
    - ``./np_config``

2. **np-solr**: This service operates the Solr search engine. It employs the official Solr image from Docker Hub and relies on the zoo service. The service mounts several volumes, including:

   - The **Solr data directory** (``./db/data/solr:/var/solr``) for data persistence.
   - Ad-hoc **custom Solr plugins**, e.g.[solr-ewb-jensen-shanon-distance-plugin](https://github.com/Nemesis1303/solr-ewb-jensen-shanon-distance-plugin) for utilizing the Jensen–Shannon divergence as a vector scoring method.
   - The **Solr configuration directory** (``./solr_config:/opt/solr/server/solr``) to access the specific Solr schemas for EWB.

3. **np-solr-initializer**: This service is temporary and serves the sole purpose of initializing the mounted volume ``/db/data`` with the necessary permissions required by Solr.

4. **np-zoo**: This service runs Zookeeper, which is essential for Solr to coordinate cluster nodes. It employs the official zookeeper image and mounts two volumes for data and logs.

5. **np-solr-config**: This service handles Solr configuration. It is constructed using the Dockerfile located in the ``solr-config`` directory. This service has dependencies on the Solr and zoo services and mounts the Docker socket and the ``bash_scripts`` directory, which contains a script for initializing the Solr configuration for EWB.

### Inference Service

To be defined

### Classification Service

To be defined

## Requirements

**Python requirements files** ([``np-solr-api``](https://github.com/Nemesis1303/NP-Backend-Dockers/blob/main/np-solr-api/requirements.txt)).

> *Note that the requirements are directly installed in their respective services at the building-up time.*
