# Using S3 buckets with rclone to move data from a PV (persistent volume on kubernetes) to your local machine:

1. Request access to S3 Resources: https://github.com/MU-HPDI/nautilus/wiki/Using-Nautilus-S3
    - This resource also shows you how to create your S3 bucket and copy data to/from.
    - Set up local S3 integration on your local compute using the instructions at the link.
2. You'll need to have an interactive pod mounted to your PV whose docker image has rclone installed. If the docker image does NOT have rclone installed, you can install it manually. Note that this will not persist, however, and monitor pods have an "expiration" of 6 hours. Here's how you install rclone manually:
```bash
-v ; curl https://rclone.org/install.sh | bash
```

- set up local s3 configuration as you did in step one, but this time set your endpoint to http://rook-ceph-rgw-centrals3.rook-central. Follow this process:
    
```bash
$ rclone config
No remotes found, make a new one?
n) New remote
s) Set configuration password
q) Quit config
n/s/q> n

Enter name for new remote.
name> nautilus

Option Storage.
Type of storage to configure.
Choose a number from below, or type in your own value.
 1 / 1Fichier
   \ (fichier)
 2 / Akamai NetStorage
   \ (netstorage)
 3 / Alias for an existing remote
   \ (alias)
 4 / Amazon Drive
   \ (amazon cloud drive)
 5 / Amazon S3 Compliant Storage Providers including AWS, Alibaba, Ceph, China Mobile, Cloudflare, ArvanCloud, Digital Ocean, Dreamhost, Huawei OBS, IBM COS, IDrive e2, Lyve Cloud, Minio, Netease, RackCorp, Scaleway, SeaweedFS, StackPath, Storj, Tencent COS and Wasabi
   \ (s3)
 6 / Backblaze B2
   \ (b2)
 7 / Better checksums for other remotes
   \ (hasher)
 8 / Box
   \ (box)
 9 / Cache a remote
   \ (cache)
10 / Citrix Sharefile
   \ (sharefile)
11 / Combine several remotes into one
   \ (combine)
12 / Compress a remote
   \ (compress)
13 / Dropbox
   \ (dropbox)
14 / Encrypt/Decrypt a remote
   \ (crypt)
15 / Enterprise File Fabric
   \ (filefabric)
16 / FTP
   \ (ftp)
17 / Google Cloud Storage (this is not Google Drive)
   \ (google cloud storage)
18 / Google Drive
   \ (drive)
19 / Google Photos
   \ (google photos)
20 / HTTP
   \ (http)
21 / Hadoop distributed file system
   \ (hdfs)
22 / HiDrive
   \ (hidrive)
23 / Hubic
   \ (hubic)
24 / In memory object storage system.
   \ (memory)
25 / Internet Archive
   \ (internetarchive)
26 / Jottacloud
   \ (jottacloud)
27 / Koofr, Digi Storage and other Koofr-compatible storage providers
   \ (koofr)
28 / Local Disk
   \ (local)
29 / Mail.ru Cloud
   \ (mailru)
30 / Mega
   \ (mega)
31 / Microsoft Azure Blob Storage
   \ (azureblob)
32 / Microsoft OneDrive
   \ (onedrive)
33 / OpenDrive
   \ (opendrive)
34 / OpenStack Swift (Rackspace Cloud Files, Memset Memstore, OVH)
   \ (swift)
35 / Pcloud
   \ (pcloud)
36 / Put.io
   \ (putio)
37 / QingCloud Object Storage
   \ (qingstor)
38 / SSH/SFTP
   \ (sftp)
39 / Sia Decentralized Cloud
   \ (sia)
40 / Storj Decentralized Cloud Storage
   \ (storj)
41 / Sugarsync
   \ (sugarsync)
42 / Transparently chunk/split large files
   \ (chunker)
43 / Union merges the contents of several upstream fs
   \ (union)
44 / Uptobox
   \ (uptobox)
45 / WebDAV
   \ (webdav)
46 / Yandex Disk
   \ (yandex)
47 / Zoho
   \ (zoho)
48 / premiumize.me
   \ (premiumizeme)
49 / seafile
   \ (seafile)
Storage> 5

Option provider.
Choose your S3 provider.
Choose a number from below, or type in your own value.
Press Enter to leave empty.
 1 / Amazon Web Services (AWS) S3
   \ (AWS)
 2 / Alibaba Cloud Object Storage System (OSS) formerly Aliyun
   \ (Alibaba)
 3 / Ceph Object Storage
   \ (Ceph)
 4 / China Mobile Ecloud Elastic Object Storage (EOS)
   \ (ChinaMobile)
 5 / Cloudflare R2 Storage
   \ (Cloudflare)
 6 / Arvan Cloud Object Storage (AOS)
   \ (ArvanCloud)
 7 / Digital Ocean Spaces
   \ (DigitalOcean)
 8 / Dreamhost DreamObjects
   \ (Dreamhost)
 9 / Huawei Object Storage Service
   \ (HuaweiOBS)
10 / IBM COS S3
   \ (IBMCOS)
11 / IDrive e2
   \ (IDrive)
12 / Seagate Lyve Cloud
   \ (LyveCloud)
13 / Minio Object Storage
   \ (Minio)
14 / Netease Object Storage (NOS)
   \ (Netease)
15 / RackCorp Object Storage
   \ (RackCorp)
16 / Scaleway Object Storage
   \ (Scaleway)
17 / SeaweedFS S3
   \ (SeaweedFS)
18 / StackPath Object Storage
   \ (StackPath)
19 / Storj (S3 Compatible Gateway)
   \ (Storj)
20 / Tencent Cloud Object Storage (COS)
   \ (TencentCOS)
21 / Wasabi Object Storage
   \ (Wasabi)
22 / Any other S3 compatible provider
   \ (Other)
provider> 22

Option env_auth.
Get AWS credentials from runtime (environment variables or EC2/ECS meta data if no env vars).
Only applies if access_key_id and secret_access_key is blank.
Choose a number from below, or type in your own boolean value (true or false).
Press Enter for the default (false).
 1 / Enter AWS credentials in the next step.
   \ (false)
 2 / Get AWS credentials from the environment (env vars or IAM).
   \ (true)
env_auth> 1

Option access_key_id.
AWS Access Key ID.
Leave blank for anonymous access or runtime credentials.
Enter a value. Press Enter to leave empty.
access_key_id> JPKIHDAZY41Q5IPG1SPM

Option secret_access_key.
AWS Secret Access Key (password).
Leave blank for anonymous access or runtime credentials.
Enter a value. Press Enter to leave empty.
secret_access_key> 2AZNOB1ivWEGKV1UYK6HDb5KuEJJ0s5dxDBxQPEE

Option region.
Region to connect to.
Leave blank if you are using an S3 clone and you don't have a region.
Choose a number from below, or type in your own value.
Press Enter to leave empty.
   / Use this if unsure.
 1 | Will use v4 signatures and an empty region.
   \ ()
   / Use this only if v4 signatures don't work.
 2 | E.g. pre Jewel/v10 CEPH.
   \ (other-v2-signature)
region>

Option endpoint.
Endpoint for S3 API.
Required when using an S3 clone.
Enter a value. Press Enter to leave empty.
endpoint> http://rook-ceph-rgw-centrals3.rook-central

Option location_constraint.
Location constraint - must be set to match the Region.
Leave blank if not sure. Used when creating buckets only.
Enter a value. Press Enter to leave empty.
location_constraint>

Option acl.
Canned ACL used when creating buckets and storing or copying objects.
This ACL is used for creating objects and if bucket_acl isn't set, for creating buckets too.
For more info visit https://docs.aws.amazon.com/AmazonS3/latest/dev/acl-overview.html#canned-acl
Note that this ACL is applied when server-side copying objects as S3
doesn't copy the ACL from the source but rather writes a fresh one.
Choose a number from below, or type in your own value.
Press Enter to leave empty.
   / Owner gets FULL_CONTROL.
 1 | No one else has access rights (default).
   \ (private)
   / Owner gets FULL_CONTROL.
 2 | The AllUsers group gets READ access.
   \ (public-read)
   / Owner gets FULL_CONTROL.
 3 | The AllUsers group gets READ and WRITE access.
   | Granting this on a bucket is generally not recommended.
   \ (public-read-write)
   / Owner gets FULL_CONTROL.
 4 | The AuthenticatedUsers group gets READ access.
   \ (authenticated-read)
   / Object owner gets FULL_CONTROL.
 5 | Bucket owner gets READ access.
   | If you specify this canned ACL when creating a bucket, Amazon S3 ignores it.
   \ (bucket-owner-read)
   / Both the object owner and the bucket owner get FULL_CONTROL over the object.
 6 | If you specify this canned ACL when creating a bucket, Amazon S3 ignores it.
   \ (bucket-owner-full-control)
acl> 1

Edit advanced config?
y) Yes
n) No (default)
y/n> n

Configuration complete.
Options:
- type: s3
- provider: Other
- access_key_id: YOUR_KEY_ID_HERE
- secret_access_key: YOUR_SECRET_KEY_HERE
- endpoint: https://s3-central.nrp-nautilus.io
- acl: private
Keep this "nautilus" remote?
y) Yes this is OK (default)
e) Edit this remote
d) Delete this remote
y/e/d> y

Current remotes:

Name                 Type
====                 ====
nautilus             s3

e) Edit existing remote
n) New remote
d) Delete remote
r) Rename remote
c) Copy remote
s) Set configuration password
q) Quit config
e/n/d/r/c/s/q> q
```

-Note: There's probabaly a better way. We should figure out how to put this config stuff in the yaml file that your pod uses so we don't have to set up local S3 Integration every time.

5. Run
   ```bash
   rclone lsd nautilus:
   ```
   both locally and from your pod to make sure your bucket exists.
7. From your interactive pod, copy data from your PV to the bucket:
   ```bash
   rclone copy --progress --copy-links {path/to/your/data} nautilus:{your-bucket}
   ```
9. From your local machine, copy data from the bucket to a local path:
    ```bash
   rclone copy --progress --copy-links nautilus:{your-bucketi/and/path/if/it/exists} {your/local/destination/path}
    ```

-To see what's in your bucket:
```
rclone ls nautilus:{bucket-name}
```

-To remove a file from the bucket:
```
rclone delete nautilus:{bucket-name}/{filename}
```

If you want to remove more than one file, you can specify the minimum file size, for example, to delete all files larger than 6MB:
```
rclone --min-size 6M delete nautilus:{bucket-name}
```

## Resizing your PVC
The easiest way to resize an existing PVC is to enter this command from your local:
```bash
kubectl edit pvc {pvc_name}
```

and edit the storage request values. save and exit. To confirm, run:
```bash
kubectl get pvc
```
You should see your volume size change here. Note that you might be allocated a little bit less than what you requested, but you can always request more.

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

