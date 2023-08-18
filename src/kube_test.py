import os
import sys
import yaml
import time
import shutil
import atexit
import datetime
import subprocess

from dateutil.tz import tzutc
from kubernetes import client, config
from jinja2 import Environment, FileSystemLoader

def exit_handler(): # always run this script after this file ends.

    config.load_kube_config()   # python can see the kube config now. now we can run API commands.

    v1 = client.CoreV1Api()   # initializing a tool to do kube stuff.

    pod_list = v1.list_namespaced_pod(namespace = params["namespace"])    # get all pods currently running (1 pod generates a single meep sim) 

    current_group = [ele.metadata.owner_references[0].name for ele in pod_list.items if(params["kill_tag"] in ele.metadata.name)]    # getting the name of the pod

    current_group = list(set(current_group))    # remove any duplicates

    for job_name in current_group:
        subprocess.run(["kubectl", "delete", "job", job_name])    # delete the kube job (a.k.a. pod)

    print("\nCleaned up any jobs that include tag : %s\n" % params["kill_tag"])   


if __name__ == "__main__":

    args = parse_args(sys.argv)
    
    params = load_config(args["config"]) 

    atexit.register(exit_handler)  # this is how we clean up jobs. 
    from IPython import embed; embed(); exit()
    #run_generation(params)
#    run_training(params)
