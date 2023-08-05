# Import: Basic Python Libraries

import os
import sys
import yaml
import shutil
import subprocess

from jinja2 import Environment, FileSystemLoader

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

    # Load template

    template = load_file(params["path_template"])

    tag = params["path_template"].split("/")[-1]
    folder = params["path_template"].replace("/%s" % tag, "")
    environment = Environment(loader = FileSystemLoader(folder))
    template = environment.get_template(tag)

    # Launch Simulation Jobs

    # - Create results folder

    sim_folder = os.path.join(params["path_results"], "simulations")
    job_folder = os.path.join(params["path_results"], "sim_job_files")

    create_folder(sim_folder)
    create_folder(job_folder)

    # - Begin data generation

    print("\nLaunching Data Generation Jobs")

    group_id, parallel_id = 1, 1

    while(group_id < params["num_sims"]):

        print("\nCurrent Group: %s\n" % parallel_id)

        # - Launch simulation job group

        current_group = []

        for i in range(params["num_parallel_ops"]):

            if(group_id + i > params["num_sims"]):
                break

            # -- Configure simulation job

            job_name = "sim-%s" % (str(group_id + i).zfill(6))

            current_group.append(job_name)

            template_info = {"job_name": job_name, 
                             "n_index": group_id + i,
                             "num_cpus": params["num_cpus_per_op"],
                             "num_mem_lim": params["num_mem_lim"], 
                             "num_mem_req": params["num_mem_req"], 
                             "path_results": sim_folder, 
                             "path_image": params["path_image"], 
                             "path_code": params["path_code"]}

            filled_template = template.render(template_info)

            path_job = os.path.join(job_folder, job_name + ".yaml")

            # -- Launch simulation job
            
            subprocess.run(["kubectl", "apply", "-f", path_job])

            # -- Save simulation job file

            save_file(path_job, filled_template)

        # - Wait until current simulation job group is completed. Checks every minute.

        k, wait_time_sec = 0, 60

        while(len(os.listdir(sim_folder)) < (params["num_parallel_ops"] * parallel_id)):

            time.sleep(wait_time_sec)

            if(k % 5 == 0):
                print("Elapsed time (group %s): %s minutes" % (parallel_id, (wait_time_sec * k) / 60))
            
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

    run_generation(params)
    
