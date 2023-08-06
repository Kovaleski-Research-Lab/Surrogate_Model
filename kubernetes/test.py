import subprocess

from kubernetes import client, config

config.load_kube_config()

v1 = client.CoreV1Api()

pod_list = v1.list_namespaced_pod(namespace = "gpn-mizzou-muem")

current_group = [ele.metadata.owner_references[0].name for ele in pod_list.items if("sim" in ele.metadata.name)]

current_group = list(set(current_group))

for job_name in current_group:
    subprocess.run(["kubectl", "delete", "job", job_name])

print("\nCleaned up any jobs that include tag : %s\n" % "sim")

