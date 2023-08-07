# SPIE Journal of Nano Photonics - Fall 2023 

## Installing kubernetes 

Install kubernetes kubernetes on Ubuntu

`sudo snap install kubectl --classic`

## Data Generation Using Kubernetes

Create storage volumes for storing our code and simulations
- `kubectl apply -f kube_files/storage/results.yaml`

Verify creation of storage volumes
- `kubectl get pvc`

Launch data generation 
- `python run_data_generation.py -config config.yaml`

Monitor data generation job
- `kubecel get job` : shows all jobs
  - Each job will spawn a pod (i.e., container) to process a simulation.
- `kubecel get pod` : shows all pods
- `kubectl describe pod [pod_name]` : shows details of the creation process for a pod. Pods take time to launch.
    - `[pod_name]` is the name of the pod shown in `kubectl get pod` above
   
- Use an interactive pod to check what simulation files are being generated
  - `kubectl apply -f monitor_pod.yaml` : create the pod. Monitor creation with `kubectl describe pod monitor`.
  - `kubectl exec -it monitor -- /bin/bash` : enter the monitor pod as interactive root user.
  - `ls /develop/results` : show created simulation files

## Repo Details

Folder descriptions:
- `kubenetes` : contains files for data generation via kubernetes.
- `build` : contains Dockerfile for running meep simulations.
- `src` : contains python scripts for generating meep data.
- `_3x3Pillars.py` : contains the class which serves as a wrapper for meep, specifically for running our simulations.
- `utils` : contains the parameter manager, which manages all of our meta atom parameters, which are initialized in config.yaml.
