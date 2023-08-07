FROM nvcr.io/nvidia/pytorch:22.06-py3

RUN apt update && apt install -y tmux vim htop git 

WORKDIR /develop/build

RUN curl -LO "https://dl.k8s.io/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"

RUN chmod a+x kubectl

RUN mv kubectl /usr/local/bin/

RUN pip install ipython pyyaml jinja2 kubernetes

WORKDIR /develop/results
WORKDIR /develop/data
WORKDIR /develop/code

#RUN git clone https://github.com/Kovaleski-Research-Lab/surrogate_model.git .
#WORKDIR /develop/code/kubernetes
