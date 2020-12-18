FROM nvidia/cuda:10.1-base-ubuntu16.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 

RUN sudo apt update

# Install more basic utilities 
RUN sudo apt install -y tmux vim 

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.6 environment
# RUN /home/user/miniconda/bin/conda create -y --name py36 python=3.6.9 \
#  && /home/user/miniconda/bin/conda clean -ya
# ENV CONDA_DEFAULT_ENV=py36
# ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
# ENV PATH=$CONDA_PREFIX/bin:$PATH
# RUN /home/user/miniconda/bin/conda install conda-build=3.18.9=py36_3 \
#  && /home/user/miniconda/bin/conda clean -ya

# Install packages for lpr environment 
COPY lpr_env.yml /home/user/
RUN /home/user/miniconda/bin/conda env create -f /home/user/lpr_env.yml 
ENV CONDA_DEFAULT_ENV=lpr
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# RUN sudo rm -rf /var/lib/apt/lists/*
RUN conda install -y -c conda-forge opencv 
RUN conda clean -ya 

# Install HDF5 Python bindings
RUN conda install -y h5py \
 && conda clean -ya
# Install Requests, a Python library for making HTTP requests
RUN conda install -y requests \
 && conda clean -ya
 
# Install Graphviz
RUN conda install -y graphviz \
 && conda clean -ya

# Install tqdm 
RUN conda install -y tqdm && conda clean -ya 

# Install apex 
WORKDIR /home/user
RUN git clone https://github.com/NVIDIA/apex 
WORKDIR /home/user/apex 
RUN pip install -v --no-cache-dir ./
WORKDIR /akshayws
