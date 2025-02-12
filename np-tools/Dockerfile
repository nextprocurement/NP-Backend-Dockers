# Use official ubuntu image with CUDA support
FROM nvidia/cuda:11.6.2-base-ubuntu20.04

# Set environment variables
ENV TZ=Europe/Madrid
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_HOME=/torch/

# Install basic utilities and dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y vim git openjdk-8-jdk ant wget curl build-essential cmake \
    libbz2-dev libenchant-dev libffi-dev libgdbm-dev libncurses5-dev \
    libnss3-dev libreadline-dev libsqlite3-dev libssl-dev tzdata zlib1g-dev \
    libgl1-mesa-glx libgl1-mesa-dev libpython3-dev python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python 3.11.1
RUN wget https://www.python.org/ftp/python/3.11.1/Python-3.11.1.tgz && \
    tar xzf Python-3.11.1.tgz && \
    cd Python-3.11.1 && \
    ./configure --enable-optimizations && \
    make altinstall && \
    ln -s /usr/local/bin/python3.11 /usr/local/bin/python && \
    ln -s /usr/local/bin/pip3.11 /usr/local/bin/pip && \
    cd .. && \
    rm -rf Python-3.11.1.tgz Python-3.11.1

# Ensure pip is installed correctly
RUN python -m ensurepip --upgrade && \
    python -m pip install --upgrade pip==23.2.1 setuptools wheel

# Install other necessary dependencies
RUN apt-get update && apt-get install -y hunspell-es

WORKDIR /np-tools

COPY . .

RUN mkdir -p /data/source


# Install project requirements
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install "dask[complete]"

# Install spaCy models
RUN python -m spacy download es_dep_news_trf
RUN python -m spacy download en_core_web_trf

# Clone the Mallet repository and build it
RUN mkdir -p /app && \
    git clone https://github.com/mimno/Mallet.git /app/Mallet && \
    cd /app/Mallet && \
    ant

WORKDIR /np-tools

# Download and cache the sentence transformer model
ARG MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${MODEL_NAME}')"

# Initialize the sentence transformer
RUN python initialize_st.py

EXPOSE 102

CMD ["python", "app.py"]
