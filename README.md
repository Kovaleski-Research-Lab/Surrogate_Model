# SPIE Journal of Nano Photonics - Fall 2023 

## Data Generation Using Kubernetes

Create storage volumes for storing our code and simulations
- `kubectl apply -f kube_files/storage/code.yaml`
- `kubectl apply -f kube_files/storage/results.yaml`

Verify creation of storage volumes
- `kubectl get pvc`

Launch data generation 
- `kubectl apply -f launch_jobs.yaml`

## Repo Details

Folder descriptions:
- `kubenetes` : contains files for data generation via kubernetes.
- `build` : contains Dockerfile for running meep simulations.
- `src` : contains python scripts for generating meep data.
- `_3x3Pillars.py` : contains the class which serves as a wrapper for meep, specifically for running our simulations.
- `utils` : contains the parameter manager, which manages all of our meta atom parameters, which are initialized in config.yaml.
