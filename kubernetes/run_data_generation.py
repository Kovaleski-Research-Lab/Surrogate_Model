# Import: Basic Python Libraries

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

# Create: Response, Program Exit

def exit_handler(): # always run this script after this file ends.

    config.load_kube_config()   # python can see the kube config now. now we can run API commands.

    v1 = client.CoreV1Api()   # initializing a tool to do kube stuff.

    pod_list = v1.list_namespaced_pod(namespace = params["namespace"])    # get all pods currently running (1 pod generates a single meep sim) 

    current_group = [ele.metadata.owner_references[0].name for ele in pod_list.items if(params["kill_tag"] in ele.metadata.name)]    # getting the name of the pod

    current_group = list(set(current_group))    # remove any duplicates

    for job_name in current_group:
        subprocess.run(["kubectl", "delete", "job", job_name])    # delete the kube job (a.k.a. pod)

    print("\nCleaned up any jobs that include tag : %s\n" % params["kill_tag"])   

# Create: Results Folders

def create_folder(path):

    if(os.path.exists(path)):
        shutil.rmtree(path)

    os.makedirs(path)

# Save: Template File

def save_file(path, data):

    data_file = open(path, "w")
   
    data_file.write(data) 

    data_file.close()

# Load: Template File

def load_file(path):

    data_file = open(path, "r")
    
    info = ""

    for line in data_file:
        info += line

    data_file.close()

    return info

# Run: Parallelized Physics Simulatiion

def run_generation(params):

    # Load job template

    template = load_file(params["path_template"])

    tag = params["path_template"].split("/")[-1]
    folder = params["path_template"].replace("/%s" % tag, "")
    environment = Environment(loader = FileSystemLoader(folder))
    template = environment.get_template(tag)

    # Launch Simulation Jobs

    create_folder(params["path_sim_job_files"])

    # - Begin data generation

    print("\nLaunching Data Generation Jobs")

    group_id, parallel_id = params["start_group_id"], 1   # group id is job, parallel id is the batch

    while(group_id < params["num_sims"]):

        print("\nCurrent Group: %s\n" % parallel_id)

        # - Launch simulation job group

        current_group = [] # list that holds all the job names

        for i in range(params["num_parallel_ops"]): 

            if(group_id + i > params["num_sims"]):
                break

            # -- Configure simulation job

            job_name = "%s-%s" % (params["kill_tag"], str(group_id + i).zfill(6))

            current_group.append(job_name)

            # jinja stuff
            template_info = {"job_name": job_name, 
                             "n_index": str(group_id + i),
                             "num_cpus": str(params["num_cpus_per_op"]),
                             "num_mem_lim": str(params["num_mem_lim"]),
                             "num_mem_req": str(params["num_mem_req"]),
                             "path_results": params["path_simulations"], "path_image": params["path_image"]}

            # this is a jinja template.
            filled_template = template.render(template_info)

            path_job = os.path.join(params["path_sim_job_files"], job_name + ".yaml") 

            if(sys.platform == "win32"):
                path_job = path_job.replace("\\", "/").replace("/", "\\")

            # -- Save simulation job file

            save_file(path_job, filled_template)

            # -- Launch simulation job

            subprocess.run(["kubectl", "apply", "-f", path_job])

        # - Wait until current simulation pod group is completed. Checks every minute.

        k = 0
        wait_time_sec = 60

        while(1):

            time.sleep(wait_time_sec)

            if(k % 2 == 0): # every 2 min including the 1st min, check progress

                config.load_kube_config()
                v1 = client.CoreV1Api()
                pod_list = v1.list_namespaced_pod(namespace = params["namespace"], timeout_seconds = 300)
            
                ##########
 
                if(k == 0):
                    print()

                pod_list = [item for item in pod_list.items if(params["kill_tag"] in item.metadata.name)]

                pod_name = [item.metadata.name for item in pod_list]
                pod_statuses = [item.status.phase for item in pod_list]

                pod_progress = [1 if(phase == "Succeeded") else 0 for phase in pod_statuses]
                hit_list =  [i for i, phase in enumerate(pod_statuses) if(phase == "Pending")]

                ##########

                for index in hit_list:

                    for i, item in enumerate(pod_list):

                        if(i == index):

                            job_name = current_group[i]
                            then = item.status.start_time
                            now = datetime.datetime.now(tzutc())

                            try:
                                diff_time = (now - then).total_seconds() / 60
                            except:
                                diff_time = 0

                            if(diff_time >= params["kill_time_min"]):
                                print("\nRemoving job: %s\n" % job_name)
                                subprocess.run(["kubectl", "delete", "job", job_name])
                                current_group.pop(i)
                                pod_statuses.pop(i)
                                pod_progress.pop(i)
                
                #print("Progressing group %s: Elapsed Time = %s minutes, Completion = %s / %s" % (parallel_id, (wait_time_sec * (k + 1)) / 60, sum(pod_progress), params["num_parallel_ops"]))

                print("Progressing group %s: Elapsed Time = %s minutes, Completion = %s / %s" % (parallel_id, (wait_time_sec * (k + 1)) / 60, sum(pod_progress), len(pod_statuses)))

                #if(sum(pod_progress) == params["num_parallel_ops"]):  # we break out when this gets to 32
                
                if(sum(pod_progress) == len(pod_statuses)):  # we break out when this gets to 32
                    print()
                    break
            
            k += 1

        # - Clean up job group

        for job_name in current_group:
            subprocess.run(["kubectl", "delete", "job", job_name])

        group_id += params["num_parallel_ops"]
        parallel_id += 1

    print("\nData Generation Complete\n")

# Validate: Configuration File

def load_config(argument):

    try:
        return yaml.load(open(argument), Loader = yaml.FullLoader) 

    except Exception as e:
        print("\nError: Loading YAML Configuration File") 
        print("\nSuggestion: Using YAML file? Check File For Errors\n")
        print(e)
        exit()        

# Parse: Command-line Arguments

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

# Main: Load Configuration File

if __name__ == "__main__":

    args = parse_args(sys.argv)

    params = load_config(args["config"]) 

    atexit.register(exit_handler)  # this is how we clean up jobs. 

    run_generation(params)
    
