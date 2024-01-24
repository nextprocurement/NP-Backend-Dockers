# NP-Backend-Dockers

## Overview

## Main components

### NP Solr API

RESTful API that utilizes the Solr search engine for data storage and retrieval. It relies on the following services:

1. **np_solr_api**: This Docker image encapsulates the NP Solr API, comprising a Python-based RESTful API. This API connects to a Python-based wrapper encapsulating Solr's fundamental functionalities.

    It has dependencies on the Solr service (``np_solr``) and requires access to the following mounted volumes:
    - ``./data/source``
    - ``./np_config``

2. **np_solr**: This service operates the Solr search engine. It employs the official Solr image from Docker Hub and relies on the zoo service. The service mounts several volumes, including:

   - The **Solr data directory** (``./db/data/solr:/var/solr``) for data persistence.
   - The following **custom Solr plugins**:
     - [solr-ewb-jensen-shanon-distance-plugin](https://github.com/Nemesis1303/solr-ewb-jensen-shanon-distance-plugin) for utilizing the Jensen–Shannon divergence as a vector scoring method.
   - The **Solr configuration directory** (``./solr_config:/opt/solr/server/solr``) to access the specific Solr schemas for EWB.

3. **np_solr_-_initializer**: This service is temporary and serves the sole purpose of initializing the mounted volume ``/db/data`` with the necessary permissions required by Solr.

4. **np_zoo**: This service runs Zookeeper, which is essential for Solr to coordinate cluster nodes. It employs the official zookeeper image and mounts two volumes for data and logs.

5. **np_solr_config**: This service handles Solr configuration. It is constructed using the Dockerfile located in the ``solr_config`` directory. This service has dependencies on the Solr and zoo services and mounts the Docker socket and the ``bash_scripts`` directory, which contains a script for initializing the Solr configuration for EWB.

### Inference Service

To be defined

### Classification Service

To be defined

## Requirements

**Python requirements files** ([``ewb-tm``](https://github.com/IntelCompH2020/EWB/blob/main/restapi/requirements.txt), [``ewb-inferencer``](https://github.com/IntelCompH2020/EWB/blob/main/inferencer/requirements.txt) and [``ewb-classifier``](https://github.com/IntelCompH2020/EWB/blob/development/classifier/requirements.txt)).

> *Note that the requirements are directly installed in their respective services at the building-up time.*

## Sample data to start using the EWB API Dockers