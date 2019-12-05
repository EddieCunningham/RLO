FROM ubuntu:18.04

USER root

# Install base stuff
RUN apt-get update && \
    apt-get install -y \
    wget \
    ca-certificates \
    git-core \
    pkg-config \
    tree \
    freetds-dev && \
    # clean up
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Anaconda
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh -O ~/anaconda.sh
RUN /bin/bash ~/anaconda.sh -b -p /opt/conda
RUN rm ~/anaconda.sh
ENV PATH /opt/conda/bin:$PATH
RUN conda update conda && conda update anaconda && conda update --all
RUN conda install pip

# Install graphviz
RUN apt-get update && \
    apt-get install -y --no-install-recommends graphviz graphviz-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install the latest versions of these python packages
RUN python -m pip install --upgrade pip && \
    pip uninstall numpy scipy bokeh cython matplotlib -y && \
    pip install --user numpy scipy pandas bokeh cython networkx graphviz \
    pygraphviz PyQt5 matplotlib pymc3 recordclass jaxlib jax tensorflow tensorflow-probability

# Add the path where bokeh gets installed to
RUN export PATH=/root/.local/bin:$PATH

# Set the base directory
WORKDIR /app

# Expose a port for bokeh
EXPOSE 8888
