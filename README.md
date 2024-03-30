# NP-Backend-Dockers

## Overview

``NP-Backend-Dockers`` powers the backend infrastructure of an application designed for efficient indexing, analysis, and retrieval of textual data leveraging a Solr engine. It also seamlessly integrates additional services like topic-based inference and classification in a multi-container setup, enhancing user functionality.

This multi-container application is orchestrated using a docker-compose script, connecting all services through the `np-net` network.


![Python Dockers](https://github.com/nextprocurement/NP-Backend-Dockers/blob/main/static/Images/np_1.png)

## Main components

### np-solr-api

This RESTful API serves as an entry point for indexing and performing a series of queries to retrieve information from the Solr search engine. It essentially acts as a Python wrapper that encapsulates Solr's fundamental functionalities within a Flask-based framework.

It has dependencies on the ``np_solr`` and ``np-tools`` services and requires access to the following mounted volumes:

- ``./data/source``
- ``./np_config``

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

To be defined

## Requirements

**Python requirements files** available within each "service" folder.

> *Requirements are directly installed in their respective services at the building-up time.*
