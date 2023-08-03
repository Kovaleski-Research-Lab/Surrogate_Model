FROM nvcr.io/nvidia/pytorch:22.06-py3

RUN apt update && apt install -y tmux vim htop git

WORKDIR /develop/build

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
bash Miniconda3-latest-Linux-x86_64.sh -p /develop/build/miniconda3 -b
ENV PATH=/develop/build/miniconda3/bin:${PATH}
RUN conda update -y conda && \
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia && \
    conda install -c conda-forge pymeep=*=mpi_mpich_* -y && \
    conda install -c pytorch torchvision -y && \
    conda install -c conda-forge pytorch-lightning -y && \
    conda install -c conda-forge jupyterlab matplotlib -y && \
    conda install -c conda-forge ipython tqdm -y && \
    conda install -c conda-forge segmentation-models-pytorch -y && \
    conda install -c conda-forge pandas -y && \
    conda install -c conda-forge ipympl -y && \
    conda install -c conda-forge ffmpeg-python -y
EXPOSE 8888

WORKDIR /develop/results
WORKDIR /develop/data
WORKDIR /develop/code
