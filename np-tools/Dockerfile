
# Use official ubuntu image with CUDA support
FROM nvidia/cuda:11.6.2-base-ubuntu20.04

# Set environment variables
ENV TZ=Europe/Madrid
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_HOME=/torch/
# ENV MALLET_HOME /app/Mallet

# Install Vim, Git, Java and Ant
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y vim git openjdk-8-jdk ant

# Install build dependencies for Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    libbz2-dev \
    libenchant-dev \
    libffi-dev \
    libgdbm-dev \
    libncurses5-dev \
    libnss3-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    tzdata \
    zlib1g-dev \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download and install Python 3.11.1
RUN wget https://www.python.org/ftp/python/3.11.1/Python-3.11.1.tgz && \
    tar xzf Python-3.11.1.tgz && \
    cd Python-3.11.1 && \
    ./configure --enable-optimizations && \
    make altinstall && \
    ln -s /usr/local/bin/python3.11 /usr/local/bin/python && \
    ln -s /usr/local/bin/pip3.11 /usr/local/bin/pip && \
    cd .. && \
    rm -rf Python-3.11.1.tgz Python-3.11.1

# Install other necessary dependencies
RUN apt-get update && apt-get install hunspell-es

WORKDIR /np-tools

COPY . .

RUN mkdir -p /data/source

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install "dask[complete]"
RUN python -m spacy download es_dep_news_trf
RUN python -m spacy download en_core_web_trf

# Clone the Mallet repository and build it
RUN mkdir -p /app && \
    git clone https://github.com/mimno/Mallet.git /app/Mallet && \
    cd /app/Mallet && \
    ant
    
# Change back to the working directory
WORKDIR /np-tools

# Download and cache the sentence transformer model
ARG MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${MODEL_NAME}')"
RUN python initialize_st.py

# This is for the CPV model
RUN mkdir -p /models && \
    python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='erick4556/cpv-3digits', filename='model.skops', cache_dir='/models'); \
    hf_hub_download(repo_id='erick4556/cpv-3digits', filename='vectorizer.skops', cache_dir='/models')"

EXPOSE 102

CMD ["python", "app.py"]