import datetime
import subprocess

from dateutil.tz import tzutc
from kubernetes import client, config

def clean_via_pods():

    config.load_kube_config()

    v1 = client.CoreV1Api()

    pod_list = v1.list_namespaced_pod(namespace = "gpn-mizzou-muem")

    #pod_phases = [item.status.phase for item in pod_list.items]
    #output = [1 for ele in pod_phases if(ele == "Succeeded")]

    current_group = [ele.metadata.owner_references[0].name for ele in pod_list.items if("sim" in ele.metadata.name)]

    current_group = list(set(current_group))

    for job_name in current_group:
        subprocess.run(["kubectl", "delete", "job", job_name])

    print("\nCleaned up any jobs that include tag : %s\n" % "sim")

def clean_via_jobs():

    config.load_kube_config()

    v1 = client.BatchV1Api()

    job_list = v1.list_namespaced_job(namespace = "gpn_mizzou-meum")

    from IPython import embed
    embed()

    print("\nCleaned up any jobs that include tag : %s\n" % "sim")

#clean_via_jobs()

config.load_kube_config()

v1 = client.CoreV1Api()

pod_list = v1.list_namespaced_pod(namespace = "gpn-mizzou-muem")

pod_names = [item.metadata.name for item in pod_list.items]
pod_phases = [item.status.phase for item in pod_list.items]

pod_times_min = []
for item in pod_list.items:

    then = item.status.start_time
    now = datetime.datetime.now(tzutc())

    print("\n")
    print(item.metadata.name)
    print("\n")
    print(then, now)

    
    try:
        pod_times_min.append((now - then).total_seconds() / 60)
    except:
        from IPython import embed
        embed()
        exit()

print("Pod Activity")

for n, p, t in zip(pod_names, pod_phases, pod_times_min):
    print("Name = %s, Phase = %s, Time (Min) = %s" % (n, p, t))

#output = [1 for ele in pod_phases if(ele == "Succeeded")]

#print(pod_phases, len(pod_phases))
#print(output, len(output))
#print(sum(output))

