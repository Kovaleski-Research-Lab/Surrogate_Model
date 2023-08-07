# SPIE Journal of Nano Photonics - Fall 2023 

## Installing kubernetes 

Install kubernetes kubernetes on Ubuntu

`sudo snap install kubectl --classic`

## Data Generation Using Kubernetes

Create storage volumes for storing our code and simulations
- `kubectl apply -f kube_files/storage/code.yaml`
- `kubectl apply -f kube_files/storage/results.yaml`

Verify creation of storage volumes
- `kubectl get pvc`

Launch data generation 
- `kubectl apply -f launch_jobs.yaml`

Monitor data generation job
- `kubecel get job` : shows all jobs
- `kubectl describe job meep-controller` : shows details of the creation process for job meep-controller. Jobs take time to launch.
  - This job lauches a pod. Monitor that pod progress with `kubectl describe pod [pod_name]`
    - `[pod_name]` is the name of the pod shown in `kubectl describe job meep-controller` above
   
- Use an interactive pod to check what simulation files are being generated
  - `kubectl apply -f monitor_pod.yaml` : create the pod. Monitor creation with `kubectl describe pod monitor`.
  - `kubectl exec -it monitor -- /bin/bash` : enter the monitor pod as interactive root user.
  - `cat /develop/results/log.out` : show i/o from python job launcher
  - `ls /develop/results/simulations` : show created simulation files
  - `ls /develop/results/sim_job_files` : show created job files

## Repo Details

Folder descriptions:
- `kubenetes` : contains files for data generation via kubernetes.
- `build` : contains Dockerfile for running meep simulations.
- `src` : contains python scripts for generating meep data.
- `_3x3Pillars.py` : contains the class which serves as a wrapper for meep, specifically for running our simulations.
- `utils` : contains the parameter manager, which manages all of our meta atom parameters, which are initialized in config.yaml.
