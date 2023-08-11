import psutil
import subprocess
import time
import pickle
from IPython import embed
import tracemalloc

    
def main():
    tracemalloc.start()
    #program_command = ["mpirun", "-np", "32", "python3", "run_sim.py", "-neighbor_index", "0",
    #                    "-resim", "0", "-folder_name", "None", "-dataset", "None",
    #                    "-path_out_sim", " .", "-path_out_log", "."]
    
    program_command = ["mpirun", "-np", "32", "python3", "run_sim.py", "-index", "0",
                        "-resim", "0"]
    process = subprocess.Popen(program_command)
    results = []
    baseline = psutil.virtual_memory()[3]/1000000000 
    print('RAM memory used before process:', baseline)
    while process.poll() is None:
        time.sleep(10)
        # Getting % usage of virtual_memory ( 3rd field)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        # Getting usage of virtual_memory in GB ( 4th field)
        print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)    
        results.append(psutil.virtual_memory()[3]/1000000000)
        print(f"max = {max(results)}, min = {min(results)}")
    
    print(f"subprocess completed. baseline = {baseline}, max = {max(results)}.")
if __name__ == "__main__":
    main()
 
