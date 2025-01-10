# Deploying an Optimum TPU Instance with Google Kubernetes Engine (GKE)


## Context

This is a short document that explains the steps to deploy Optimum TPU on [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine). Google Kubernetes Engine (GKE) is a solution that offers containers orchestration withing Google Cloud Platform. For more information on GKE, refer to the [official documentation](https://cloud.google.com/kubernetes-engine/docs) and the documentation about [TPUs on GKE](https://cloud.google.com/kubernetes-engine/docs/concepts/tpus).

We also provide [examples of other GKE deployments](https://github.com/huggingface/Google-Cloud-Containers/tree/main/examples) on the Hugging Face's Google Cloud Containers repository.


We assume TPU quotas have been [requested](https://console.cloud.google.com/iam-admin/quotas) to be able to use GKE.
We will also need few tools:

* Google Cloud CLI. If you have not installed it before, please follow the links right after to
[install](https://cloud.google.com/sdk/docs/install) and [setup](https://cloud.google.com/sdk/docs/initializing).
* The `kubectl` command-line tool allows to interact with GKE. To install it, use this [link](https://kubernetes.io/docs/tasks/tools/#kubectl).

We will need to login to Google Cloud Platform (GCP) and choose the active project to avoid errors on this:

```bash
export PROJECT_ID="your-project-id"
gcloud auth login
gcloud config set project $PROJECT_ID
```

Make sure also that you [enabled the Artifact Registry and the GKE APIs](https://console.cloud.google.com/flows/enableapi?apiid=artifactregistry.googleapis.com%2Ccontainer.googleapis.com).
If you prefer to do it with gcloud, you can do:

```bash
gcloud services enable container.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable containerfilesystem.googleapis.com
```

Additionally, in order to use `kubectl` with the GKE Cluster credentials, we also need to install the `gke-gcloud-auth-plugin`, that can be installed with `gcloud` as follows:

```bash
gcloud components install gke-gcloud-auth-plugin
```

> [!NOTE]
> Installing the `gke-gcloud-auth-plugin` does not need to be installed via `gcloud` specifically, to read more about the alternative installation methods, please visit [Install `kubectl` and configure cluster access](https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl).


## Plan the GKE Cluster

First we need to [plan what TPUs we want to use](https://cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus).  To make an example, we will consider these choices:

- We will use the autopilot mode, to simplify the deployment. On this mode, we will select the `tpu-v5-lite-podslice` as it is the single-host solution for `v5e`.
- We are going to select the 2x2 topology. For now, Optimum TPU only support single-host TPUs. We can see the available topologies [here](https://cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus#topology).
- Validate where you want to [locate](https://cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus#availability) our TPU. We picked a `us-central1` from the Google Cloud regions where out TPUs choice is available. Note that we are going to choose a region. To list all regions we can type
```bash
gcloud compute zones list
```

For convenience, we will set variables that reflect our choices

```bash
export CLUSTER_NAME=my-tpu-cluster
export MACHINE_TYPE=tpu-v5-lite-podslice
export LOCATION=us-central1
export TOPOLOGY=2x2
```

## Creating the GKE Cluster

We can now create the GKE Cluster, that will contain a single TPU node. In order to deploy it, we will use the "Autopilot" mode, which is the recommended one for most of the workloads, since the underlying infrastructure is managed by Google. Alternatively, we can also use the "Standard" mode. To check the differences between Autopilot and Standard mode for TPU workloads check either [Deploy TPU workloads in GKE Autopilot](https://cloud.google.com/kubernetes-engine/docs/how-to/tpus-autopilot) or [Deploy TPU workloads in GKE Standard](https://cloud.google.com/kubernetes-engine/docs/how-to/tpus), respectively.

```bash
gcloud container clusters create-auto $CLUSTER_NAME \
    --project=$PROJECT_ID \
    --location=$LOCATION \
    --release-channel=stable \
    --cluster-version=1.29
```

This step can take 5 minutes or more.


## Secrets in GKE

We can now proceed to the deployment, but we might want to create Kubernetes secrets to hold Hugging Face Hub security token, that might be necessary to access some gated models (check [this link](https://huggingface.co/docs/hub/en/security-tokens) for more info). To do that, first get the credentials for the cluster, so it will be possible to access it via `kubectl`:


```bash
gcloud container clusters get-credentials $CLUSTER_NAME --location=$LOCATION
```

It is now possible to create the secret and apply it:

```bash
kubectl create secret generic hf-secret \
    --from-literal=hf_token=$HF_TOKEN \
    --dry-run=client -o yaml | kubectl apply -f -
```

More information on how to set Kubernetes secrets in a GKE Cluster at [Use Secret Manager add-on with Google Kubernetes Engine](https://cloud.google.com/secret-manager/docs/secret-manager-managed-csi-component).


# Deploy Text Generation Inference on TPU

[Text Generation Inference (TGI)](https://huggingface.co/docs/text-generation-inference/en/index) is the text generation serving solution created by Hugging Face. To serve a model on TPU, first we need to select a model, e.g.: `meta-llama/Llama-3.2-1B-Instruct`. We need to make sure we added the Hugging Face token to access this gated model, please check previous section to do that.

Configuration files for the deployment are available [here](https://github.com/huggingface/optimum-tpu/tree/main/examples/gke/configs-tgi). Supposing we downloaded the whole directory, we can fine three configuration files:

* `deployment.yaml`: contains the deployment details of the pod including the reference to the Hugging Face LLM DLC setting the `MODEL_ID` to `meta-llama/Llama-3.2-1B-Instruct`. Note that we provided a container compatible with google services that contains the Optimum TPU's TGI runtime.
* `service.yaml`: contains the service details of the pod, exposing the port 8080 for the TGI service.
* (optional) `ingress.yaml`: contains the ingress details of the pod, exposing the service to the external world so that it can be accessed via the ingress IP.

We can adjust the seeings and apply:

```bash
kubectl apply -f configs-tgi/
```

This might take few minutes. To check the status, we can run this command:

```bash
kubectl get pods
```

# Test TGI Deployment

Since we configured ingress in the `ingress.yaml` file, we can now access the exposed external port, that we will store in a shell variable:

```bash
TGI_IP=`kubectl get ingress tgi-ingress -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

It is now going to be able to run an inference on the TGI server with a simple `cURL` command:

```bash
curl http://$TGI_IP/generate \
    -X POST \
    -d '{"inputs":"<bos><start_of_turn>user\nWhat is 40+2?<end_of_turn>\n<start_of_turn>model\n","parameters":{"temperature":0.7, "top_p": 0.95, "max_new_tokens": 50}}' \
    -H 'Content-Type: application/json'
```

For other ways to submit requests to TGI, refer to this [documentation](https://huggingface.co/docs/text-generation-inference/en/basic_tutorials/consuming_tgi).


# Delete GKE Cluster

Once we are done using TGI in the GKE Cluster, we can safely delete the cluster we've just created to avoid incurring in unnecessary costs.

```bash
gcloud container clusters delete $CLUSTER_NAME --location=$LOCATION
```

