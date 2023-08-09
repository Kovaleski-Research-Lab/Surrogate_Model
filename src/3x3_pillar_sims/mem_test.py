import psutil
import subprocess
import time
import pickle
from IPython import embed

def get_memory_usage(pid):
    process = psutil.Process(pid)
    memory_info = process.memory_info()
    return memory_info.rss

def main():
    program_command = ["mpirun", "-np", "32", "python3", "run_sim.py", "-neighbor_index", "0",
                        "-resim", "0", "-folder_name", "None", "-dataset", "None",
                        "-path_output", " ."]
    process = subprocess.Popen(program_command)
    
    try:
        i = 0
        res = []
        while True:
            memory_usage = get_memory_usage(process.pid) / 1e9 # MB
            print(f"Memory usage: {memory_usage} MB")
            time.sleep(10)
            res.append(memory_usage)
                  
    except KeyboardInterrupt:
        print(f"max = {max(res)}")
        print(f"min = {min(res)}")
        
        with open("res.pkl", "wb") as f:
            pickle.dump(res, f)

        process.terminate()
        process.wait()
    
if __name__ == "__main__":
    main()
 
