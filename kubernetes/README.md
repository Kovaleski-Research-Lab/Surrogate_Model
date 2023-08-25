# Using S3 buckets with rclone to move data from a PV (persistent volume on kubernetes) to your local machine:

1. Request access to S3 Resources: https://github.com/MU-HPDI/nautilus/wiki/Using-Nautilus-S3
    - This resource also shows you how to create your S3 bucket and copy data to/from.
    - You'll want to set up local S3 integration on your local compute using the instructions at the link.
2. You'll need to have an interactive pod mounted to your PV whose docker image has rclone installed.
    - -v ; curl https://rclone.org/install.sh | bash https://rclone.org/install/
    - the pod needs to have environment variables set like this one: https://github.com/MU-HPDI/nautilus/blob/main/kube/cloudstorage/rclone_pod.yml
3. When you enter into the pod (kubectl exec -it {podName} -- /bin/bash) you will need to set up local S3 Integration, like you did on your local machine, only this time set your endpoint to http://rook-ceph-rgw-centrals3.rook-central.
    - Note: There's probbaly a better way. We should figure out how to put this config stuff in the yaml file that your pod uses so we don't have to set up local S3 Integration every time.
4. Run `rclone lsd nautilus:` both locally and from your pod to make sure your bucket exists.
5. From your interactive pod, copy data from your PV to the bucket: `rclone copy --progress --copy-links {path/to/your/data} nautilus:{your-bucket}`
    - Note: If you build directories in your bucket, make sure you include the path to the data your grabbing: `rclone copy --progress --copy-links {/path/to/your/data} nautilus:{your-bucket/your/path}`
6. From your local machine, copy data from the bucket to a local path: `rclone copy --progress --copy-links nautilus:{your-bucketi/and/path/if/it/exists} {your/local/destination/path}`

# Opening an jupyter notebook with kubernetes:

1. open a port on the machine whose browser you'll be using: ssh -N -f -L localhost:{port_number}:localhost:{port_number} {pawprint}@{ip_adress}
    - Example: ssh -N -f -L localhost:10099:localhost:10099 agkgd4@128.206.23.4 (I'm connecting MINDFUL port 10099 to WOPR port 10099)
2. on the machine you're using to interface with kubernetes: create the pod using jupyter_pod.yaml: kubectl apply -f jupyter_pod.yaml
                                                             and enter into the pod: kubectl exec -it jupyter -- /bin/bash
3. in another pane, create a port forward connection:  kubectl port-forward {pod-name} {machine-connected-to-kube-port}:{host-port} -n {name-space}
    - Example: kubectl port-forward jupyter-testing 10097:8888 -n gpn-mizzou-muem
4. go back to the pod and run jupyter lab command. Open a browser and enter the local ip and port number in the addresss bar
    - Example: 127.0.0.1:1099
   You'll be routed to a jupyter page that asks for a token. Use the long hash given in the pod window (copy and paste).

# There's a Mizzou-HPC slack 

1. Ask Dr. Scott to give you access. (Ask Andy first. I might be able to invite you) Then you can tag Alex Hurt and Anes Ouadou with kubernetes questions.
2. Also, kubernetes support: https://element.nrp-nautilus.io/#/room/#general:matrix.nrp-nautilus.io (You have an account if you completed setp 1 in https://github.com/MU-HPDI/nautilus/wiki/Using-Nautilus-S3.

