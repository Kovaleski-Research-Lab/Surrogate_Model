import os
import sys
import yaml
import time
import shutil
import atexit
import datetime
import subprocess
from IPython import embed

from dateutil.tz import tzutc
from kubernetes import client, config
from jinja2 import Environment, FileSystemLoader

def exit_handler(): # always run this script after this file ends.

    config.load_kube_config()   # python can see the kube config now. now we can run API commands.

    v1 = client.CoreV1Api()   # initializing a tool to do kube stuff.

    pod_list = v1.list_namespaced_pod(namespace = params["namespace"])    # get all pods currently running (1 pod generates a single meep sim) 
    
    # getting the name of the pod
    current_group = [ele.metadata.owner_references[0].name for ele in pod_list.items if(params["kill_tag"] in ele.metadata.name)]    
    current_group = list(set(current_group))    # remove any duplicates

    for job_name in current_group:
        subprocess.run(["kubectl", "delete", "job", job_name])    # delete the kube job (a.k.a. pod)

    print("\nCleaned up any jobs that include tag : %s\n" % params["kill_tag"])   

def load_config(argument):

    try:
        return yaml.load(open(argument), Loader = yaml.FullLoader) 

    except Exception as e:
        print("\nError: Loading YAML Configuration File") 
        print("\nSuggestion: Using YAML file? Check File For Errors\n")
        print(e)
        exit()        

def parse_args(all_args, tags = ["--", "-"]):

    all_args = all_args[1:]

    if(len(all_args) % 2 != 0):
        print("Argument '%s' not defined" % all_args[-1])
        exit()

    results = {}

    i = 0
    while(i < len(all_args) - 1):
        arg = all_args[i].lower()
        for current_tag in tags:
            if(current_tag in arg):
                arg = arg.replace(current_tag, "")                
        results[arg] = all_args[i + 1]
        i += 2

    return results

def load_file(path):

    data_file = open(path, "r")
    
    info = ""

    for line in data_file:
        info += line

    data_file.close()

    return info

def create_folder(path):

    if(os.path.exists(path)):
        shutil.rmtree(path)

    os.makedirs(path)

def save_file(path, data):

    data_file = open(path, "w")
   
    data_file.write(data) 

    data_file.close()

def run_test(params):

    template = load_file(params["path_template"])

    tag = params["path_template"].split("/")[-1]
    folder = params["path_template"].replace("/%s" % tag, "")
    environment = Environment(loader = FileSystemLoader(folder))
    template = environment.get_template(tag)
    # this is where we'll dump the .yaml files that contain job information
    # maybe this can be deleted - we used it to keep track of meep sim jobs when lots of jobs were running in parallel :/

    create_folder(params["path_train_job_files"])

    job_name = "%s-%s" % (params["kill_tag"], str(params['experiment']).zfill(6))

    template_info = {"job_name": job_name, 
                     #"n_index": str(counter),
                     "num_cpus": str(params["num_cpus_per_op"]),
                     "num_mem_lim": str(params["num_mem_lim"]),
                     "num_mem_req": str(params["num_mem_req"]),
                     "path_sims": params["path_simulations"], "path_image": params["path_image"], "path_logs": params["path_logs"]}
    filled_template = template.render(template_info)

    path_job = os.path.join(params["path_train_job_files"], job_name + ".yaml") 

    # --- Save job file

    save_file(path_job, filled_template)
    #from IPython import embed; embed(); exit()
    # --- Launch job

    print(f"running kubernetes subprocess...{path_job}")
    subprocess.run(["kubectl", "apply", "-f", path_job])
    
if __name__ == "__main__":

    args = parse_args(sys.argv)
    
    params = load_config(args["config"]) 

    #atexit.register(exit_handler)  # this is how we clean up jobs. 
    run_test(params)
