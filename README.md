# Scikit-learn Model Serving with KServe and MLflow

A production-oriented local MLOps project demonstrating how to train, register, and serve a scikit-learn model using:

* Python
* scikit-learn
* MLflow
* DagsHub Hosted MLflow
* Kubernetes
* KServe
* MLServer
* KServe Models Web App
* Docker
* kind

The project uses the diabetes regression dataset provided by scikit-learn and deploys a `RandomForestRegressor` through KServe.

The model artifact is stored in DagsHub-hosted MLflow. KServe retrieves the registered model through a custom MLflow storage initializer and serves it using KServe's MLServer runtime.

---

## Architecture
<img width="2172" height="1984" alt="kserve-mlops" src="https://github.com/user-attachments/assets/ad5eb32d-8068-411f-a508-142ed1b17106" />


---

# 1. Project Goals

This project demonstrates the complete path from model training to Kubernetes-based inference:

```text
Dataset
   |
   v
Train model
   |
   v
Track experiment with MLflow
   |
   v
Register model
   |
   v
Deploy model with KServe
   |
   v
Download model from MLflow
   |
   v
Load model with MLServer
   |
   v
Expose inference endpoint
   |
   v
Test prediction
   |
   v
Manage model endpoint through KServe Models Web App
```

The project intentionally separates:

| Component                  | Responsibility                            |
| -------------------------- | ----------------------------------------- |
| scikit-learn               | Model training                            |
| MLflow                     | Experiment tracking and model registry    |
| DagsHub                    | Hosted MLflow server and artifact storage |
| Kubernetes                 | Container orchestration                   |
| KServe                     | Model serving orchestration               |
| MLServer                   | Model inference runtime                   |
| Custom Storage Initializer | Downloads MLflow model artifacts          |
| Kubernetes Secret          | Stores MLflow credentials                 |
| Models Web App             | Web UI for KServe                         |

---

# 2. Versions

This project uses the following versions.

| Component      | Version                                  |
| -------------- | ---------------------------------------- |
| Python         | 3.12                                     |
| MLflow         | 2.10.2                                   |
| scikit-learn   | 1.4.1.post1                              |
| KServe         | 0.16.0                                   |
| Models Web App | 0.16.1                                   |
| Kubernetes     | 1.33.x                                   |
| kind           | Local Kubernetes cluster                 |
| Docker         | Required                                 |
| MLServer       | Provided through KServe MLServer runtime |

The versions are intentionally pinned to keep the environment reproducible.

---

# 3. Prerequisites

Install the following tools:

* Docker
* kubectl
* kind
* Python 3.12
* Git

Verify:

```bash
docker --version
kubectl version --client
kind version
python3.12 --version
git --version
```

You also need a DagsHub account and a DagsHub repository with MLflow enabled.

---

# 4. Clone the Repository

Clone the project:

```bash
git clone https://github.com/Prasadchaskar/sklearn-kserve-mlops.git
cd sklearn-kserve-mlops
```

---

# 5. Create the Python Virtual Environment

Create the virtual environment:

```bash
python3.12 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Upgrade packaging tools:

```bash
pip install --upgrade pip setuptools wheel
```

---

# 6. Install Python Dependencies

Install the required packages:

```bash
pip install \
  "mlflow==2.10.2" \
  "scikit-learn==1.4.1.post1" \
  "python-dotenv" \
  "PyYAML"
```

Install a compatible setuptools version:

```bash
pip install "setuptools<81"
```

Verify MLflow and scikit-learn:

```bash
python -c "import mlflow, sklearn; print('MLflow:', mlflow.__version__); print('scikit-learn:', sklearn.__version__)"
```

Expected:

```text
MLflow: 2.10.2
scikit-learn: 1.4.1.post1
```

---

# 7. Configure DagsHub MLflow

Create the environment file:

```bash
cp .env.example .env
```

Edit it:

```bash
nano .env
```

Set:

```env
MLFLOW_TRACKING_URI=<YOUR_DAGSHUB_MLFLOW_URI>
MLFLOW_TRACKING_USERNAME=<YOUR_DAGSHUB_USERNAME>
MLFLOW_TRACKING_PASSWORD=<YOUR_DAGSHUB_TOKEN_OR_PASSWORD>
```

Do not commit `.env`.

The `.gitignore` should contain:

```gitignore
.env
.venv/
__pycache__/
*.pyc
```

The `.env.example` file should contain only placeholders:

```env
MLFLOW_TRACKING_URI=
MLFLOW_TRACKING_USERNAME=
MLFLOW_TRACKING_PASSWORD=
```

---

# 8. Project Structure

The project structure is:

```text
sklearn-kserve-mlops/
│
├── configs/
│   └── training.yaml
│
├── kubernetes/
│   ├── diabetes-regressor.yaml
│   └── mlflow-storage-container.yaml
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── inspect_model.py
│   │
│   └── serving/
│       ├── __init__.py
│       ├── mlflow_storage_initializer.py
│       └── Dockerfile
│
├── tests/
│   └── .gitkeep
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

---

# 9. Train the Model

Run:

```bash
python -m src.training.train
```

The training script:

1. Loads the diabetes dataset.
2. Splits the dataset into training and testing data.
3. Creates a `RandomForestRegressor`.
4. Trains the model.
5. Calculates evaluation metrics.
6. Logs parameters to MLflow.
7. Logs metrics to MLflow.
8. Logs the model artifact to MLflow.
9. Registers the model in MLflow.

The registered model name is:

```text
diabetes-regressor
```

The model uses MLflow's sklearn flavor and is stored using a pickle/cloudpickle-compatible format supported by the MLflow runtime used in this project.

---

# 10. Verify the Registered Model

Run:

```bash
python -m src.training.inspect_model
```

The model should be registered in DagsHub MLflow.

The expected registered model is:

```text
diabetes-regressor
```

The project uses model version:

```text
2
```

The resulting MLflow model URI is:

```text
models:/diabetes-regressor/2
```

---

# 11. Create the Kubernetes Cluster

Create a local kind cluster:

```bash
kind create cluster --name my-local-cluster
```

Verify:

```bash
kubectl cluster-info
```

Check the node:

```bash
kubectl get nodes
```

Expected:

```text
my-local-cluster-control-plane   Ready
```

Verify the Kubernetes server version:

```bash
kubectl version
```

---

## 12. Install KServe

This project uses **KServe v0.16.0** with **Standard deployment mode**.

Standard mode uses Kubernetes Deployments and Services to serve machine learning models without requiring Knative Serving.

### 12.1 Install KServe

Install KServe using the official v0.16.0 release manifests.

```bash
# Install KServe CRDs and controller
kubectl apply --server-side -f \
  https://github.com/kserve/kserve/releases/download/v0.16.0/kserve.yaml

# Install default ClusterServingRuntimes
kubectl apply --server-side -f \
  https://github.com/kserve/kserve/releases/download/v0.16.0/kserve-cluster-resources.yaml
```

The `--server-side` option is used when applying the KServe manifests because some KServe CRDs are large.

### 12.2 Configure Standard Deployment Mode

KServe supports multiple deployment modes. Configure Standard as the default mode for this project.

```bash
kubectl patch configmap/inferenceservice-config \
  -n kserve \
  --type=merge \
  -p '{"data":{"deploy":"{\"defaultDeploymentMode\":\"Standard\"}"}}'
```

Verify the deployment mode:

```bash
kubectl get configmap inferenceservice-config \
  -n kserve \
  -o jsonpath='{.data.deploy}'
```

Expected configuration:

```json
{
  "defaultDeploymentMode": "Standard"
}
```

### 12.3 Verify KServe Installation

Verify that the KServe namespace exists:

```bash
kubectl get namespace kserve
```

Verify the KServe controller:

```bash
kubectl get pods -n kserve
```

The KServe controller pod should reach:

```text
Running
```

Verify the KServe CRDs:

```bash
kubectl get crd | grep serving.kserve.io
```

You should see resources including:

```text
inferenceservices.serving.kserve.io
```

Verify the available ClusterServingRuntimes:

```bash
kubectl get clusterservingruntimes
```

The cluster should contain the MLServer runtime:

```text
kserve-mlserver
```

KServe is now installed and ready for deploying the project's InferenceService.

---

# 13. Create the KServe Namespace

Create a dedicated namespace for the model:

```bash
kubectl create namespace sklearn-kserve
```

Verify:

```bash
kubectl get namespace sklearn-kserve
```

---

# 14. Create the MLflow Kubernetes Secret

The model-serving infrastructure needs credentials to access DagsHub MLflow.

Create the Secret from `.env`:

```bash
kubectl create secret generic dagshub-mlflow \
  --namespace sklearn-kserve \
  --from-env-file=.env
```

Verify the Secret exists:

```bash
kubectl get secret dagshub-mlflow -n sklearn-kserve
```

Verify only the keys:

```bash
kubectl get secret dagshub-mlflow \
  -n sklearn-kserve \
  -o jsonpath='{.data}' | tr ',' '\n'
```

The Secret must contain:

```text
MLFLOW_TRACKING_URI
MLFLOW_TRACKING_USERNAME
MLFLOW_TRACKING_PASSWORD
```

Do not print or commit the actual credential values.

---

# 15. Build the Custom MLflow Storage Initializer

KServe's storage initializer is responsible for downloading model artifacts before the inference runtime starts.

The default storage initializer does not handle the MLflow:

```text
models:/
```

URI used by this project.

Therefore, this project provides a custom storage initializer.

The initializer:

1. Receives the model URI from KServe.
2. Reads MLflow credentials from environment variables.
3. Connects to DagsHub MLflow.
4. Downloads the model artifact.
5. Copies the model into `/mnt/models`.
6. Exits successfully.
7. Allows the MLServer container to load the model.

The entrypoint accepts:

```text
<model_uri> <destination_path>
```

For example:

```text
models:/diabetes-regressor/2 /mnt/models
```

---

# 16. Build the Storage Initializer Image

From the project root:

```bash
docker build \
  -t sklearn-kserve-mlflow-storage:v1 \
  -f src/serving/Dockerfile \
  .
```

Verify:

```bash
docker images | grep sklearn-kserve-mlflow-storage
```

---

# 17. Load the Image into kind

Because the Kubernetes cluster is running with kind, load the image into the kind node:

```bash
kind load docker-image \
  sklearn-kserve-mlflow-storage:v1 \
  --name my-local-cluster
```

Verify the image is available to kind:

```bash
docker exec \
  my-local-cluster-control-plane \
  crictl images | grep sklearn-kserve-mlflow-storage
```

---

# 18. Register the Custom Storage Initializer with KServe

The file:

```text
kubernetes/mlflow-storage-container.yaml
```

defines a KServe:

```text
ClusterStorageContainer
```

The important configuration is:

```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: ClusterStorageContainer

metadata:
  name: mlflow-storage-initializer
```

The container uses:

```text
sklearn-kserve-mlflow-storage:v1
```

and receives the MLflow credentials from:

```text
dagshub-mlflow
```

The supported URI is:

```text
models:/
```

Apply it:

```bash
kubectl apply \
  -f kubernetes/mlflow-storage-container.yaml
```

Verify:

```bash
kubectl get clusterstoragecontainers
```

You should see:

```text
mlflow-storage-initializer
```

---

# 19. Deploy the Model with KServe

The model configuration is:

```text
kubernetes/diabetes-regressor.yaml
```

The important section is:

```yaml
spec:
  predictor:
    model:
      modelFormat:
        name: mlflow
      protocolVersion: v2
      storageUri: "models:/diabetes-regressor/2"
```

Apply it:

```bash
kubectl apply \
  -f kubernetes/diabetes-regressor.yaml
```

---

# 20. Check the InferenceService

Run:

```bash
kubectl get inferenceservice \
  -n sklearn-kserve
```

Expected:

```text
NAME                 READY
diabetes-regressor   True
```

You can also run:

```bash
kubectl describe inferenceservice \
  diabetes-regressor \
  -n sklearn-kserve
```

---

# 21. Check the Predictor Pod

Run:

```bash
kubectl get pods \
  -n sklearn-kserve
```

The predictor pod should eventually show:

```text
1/1 Running
```

For example:

```text
diabetes-regressor-predictor-xxxxxxxxxx-xxxxx
```

---

# 22. Verify Storage Initializer Logs

Find the predictor pod:

```bash
kubectl get pods -n sklearn-kserve
```

Then check the storage initializer logs:

```bash
kubectl logs \
  -n sklearn-kserve \
  <POD_NAME> \
  -c storage-initializer
```

The logs should show:

```text
Configuring MLflow tracking.
Model URI: models:/diabetes-regressor/2
Destination: /mnt/models
Downloading MLflow model artifact.
Model downloaded to temporary location.
MLflow model successfully copied to: /mnt/models
Temporary model directory cleaned up.
```

This confirms that:

```text
DagsHub MLflow
       |
       v
Custom Storage Initializer
       |
       v
/mnt/models
```

is working.

---

# 23. Verify MLServer

Check the main container logs:

```bash
kubectl logs \
  -n sklearn-kserve \
  <POD_NAME>
```

The important message is:

```text
Loaded model 'diabetes-regressor' successfully.
```

The server should also report that it is listening on:

```text
HTTP: 8080
gRPC: 9000
```

---

# 24. Test the Model Locally

Forward the KServe predictor service:

```bash
kubectl port-forward \
  -n sklearn-kserve \
  svc/diabetes-regressor-predictor \
  8080:80
```

Keep this terminal running.

Open another terminal.

---

# 25. Test the MLflow Inference Endpoint

Use:

```bash
curl -X POST \
  http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
      "columns": [
        "age",
        "sex",
        "bmi",
        "bp",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6"
      ],
      "data": [
        [
          0.03807591,
          0.05068012,
          0.06169621,
          0.02187235,
          -0.0442235,
          -0.03482076,
          -0.04340085,
          -0.00259226,
          0.01990749,
          -0.01764613
        ]
      ]
    }
  }'
```

Expected response format:

```json
{
  "predictions": [
    225.08212318237315
  ]
}
```

The exact prediction may vary if the training configuration or model version changes.

---

# 26. Understanding the Complete Prediction Flow

When the request is sent:

```text
curl
 |
 v
KServe Service
 |
 v
MLServer
 |
 v
MLflow model loaded from /mnt/models
 |
 v
RandomForestRegressor
 |
 v
Prediction
```

The model is not downloaded from MLflow for every prediction.

The model is downloaded when the serving pod is initialized.

After that, the model is loaded into the inference runtime.

---

# 27. Deploy KServe Models Web App

The KServe Models Web App is a separate application from the KServe controller.

It provides a browser-based UI for KServe model-serving resources.

Clone the Models Web App repository locally:

```bash
cd ..
git clone https://github.com/kserve/models-web-app.git
cd models-web-app
```

Checkout the version used by this project:

```bash
git checkout v0.16.1
```

---

# 28. Configure the Models Web App for Local Port-Forwarding

The Models Web App normally expects to be exposed through an ingress and `/models/` URL prefix.

For local port-forwarding, configure:

```yaml
APP_PREFIX="/"
APP_DISABLE_AUTH="True"
APP_SECURE_COOKIES="False"
```

Edit:

```bash
nano config/base/kustomization.yaml
```

Under:

```yaml
configMapGenerator:
- literals:
```

make sure these settings are present:

```yaml
- APP_PREFIX="/"
- APP_DISABLE_AUTH="True"
- APP_SECURE_COOKIES="False"
```

The application image should be pinned to:

```yaml
images:
- name: kserve/models-web-app
  newName: ghcr.io/kserve/models-web-app
  newTag: 0.16.1
```

The Models Web App uses these settings specifically for local `kubectl port-forward` access: it needs the `/` prefix, authorization disabled because there is no logged-in user, and secure cookies disabled because local port-forwarding uses plain HTTP.

---

# 29. Deploy the Models Web App

For this local setup, do not deploy the Istio resource because the cluster is not using Istio for this UI.

The base Kustomization should contain:

```yaml
resources:
- rbac.yaml
- service.yaml
- deployment.yaml
```

and should not contain:

```yaml
- istio.yaml
```

Apply the application:

```bash
kubectl apply -k config/base
```

---

# 30. Verify Models Web App

Check the pod:

```bash
kubectl get pods -n kserve
```

The Models Web App pod should show:

```text
1/1 Running
```

Check the service:

```bash
kubectl get svc \
  kserve-models-web-app \
  -n kserve
```

Expected:

```text
kserve-models-web-app   ClusterIP
```

---

# 31. Access the Models Web App

Create a port-forward:

```bash
kubectl port-forward \
  -n kserve \
  svc/kserve-models-web-app \
  8081:80
```

Open:

```text
http://localhost:8081/
```

The KServe Models Web App should load.

---

# 32. Verify the Models Web App from the Terminal

You can verify that the application is serving HTTP:

```bash
curl -i http://localhost:8081/
```

You should receive:

```text
HTTP/1.1 200 OK
```

and HTML content for the Models Web App.

---

# 33. Verify the KServe Model Through the UI

The Models Web App should be able to discover the KServe resources running in the Kubernetes cluster.

The model deployed earlier is:

```text
diabetes-regressor
```

Namespace:

```text
sklearn-kserve
```

The UI provides a graphical way to inspect and manage KServe model-serving resources.

---

# 34. Useful Kubernetes Commands

Check all KServe resources:

```bash
kubectl get inferenceservices \
  -A
```

Check model pods:

```bash
kubectl get pods \
  -n sklearn-kserve
```

Check services:

```bash
kubectl get svc \
  -n sklearn-kserve
```

Check KServe runtimes:

```bash
kubectl get clusterservingruntimes
```

Check storage containers:

```bash
kubectl get clusterstoragecontainers
```

Check the KServe controller:

```bash
kubectl get pods \
  -n kserve
```

Check model events:

```bash
kubectl describe inferenceservice \
  diabetes-regressor \
  -n sklearn-kserve
```

Check model logs:

```bash
kubectl logs \
  -n sklearn-kserve \
  <POD_NAME>
```

---

# 35. Important Files

## `configs/training.yaml`

Contains training configuration:

```yaml
experiment:
  name: sklearn-diabetes-regression

model:
  name: diabetes-regressor

data:
  test_size: 0.2
  random_state: 42

training:
  model_type: random_forest
  n_estimators: 100
  max_depth: 10
  random_state: 42
```

---

## `src/config.py`

Loads required environment variables and prevents the application from starting when required configuration is missing.

The important variables are:

```text
MLFLOW_TRACKING_URI
MLFLOW_TRACKING_USERNAME
MLFLOW_TRACKING_PASSWORD
```

---

## `src/training/train.py`

Responsible for:

```text
Dataset loading
       |
Data splitting
       |
Model training
       |
Evaluation
       |
MLflow logging
       |
Model registration
```

---

## `src/serving/mlflow_storage_initializer.py`

Responsible for:

```text
Receive MLflow model URI
       |
Read MLflow credentials
       |
Connect to MLflow
       |
Download artifact
       |
Copy artifact to /mnt/models
```

It does not perform inference.

---

## `src/serving/Dockerfile`

Builds the custom storage initializer container.

The image contains:

```text
Python
MLflow
Storage initializer code
```

It does not contain the trained model.

---

## `kubernetes/mlflow-storage-container.yaml`

Registers the custom storage initializer with KServe.

It maps:

```text
models:/
```

to:

```text
sklearn-kserve-mlflow-storage:v1
```

---

## `kubernetes/diabetes-regressor.yaml`

Defines the KServe `InferenceService`.

It tells KServe:

```text
Model format = MLflow

Model URI =
models:/diabetes-regressor/2

Protocol =
V2
```

---

# 36. Security

Never commit:

```text
.env
```

Never put credentials directly into:

```text
Dockerfile
Git repository
InferenceService YAML
Python source code
```

Credentials are injected through Kubernetes Secrets.

For production deployments, the Kubernetes Secret should normally be integrated with an enterprise secrets-management solution such as:

```text
AWS Secrets Manager
Azure Key Vault
Google Secret Manager
HashiCorp Vault
External Secrets Operator
```

depending on the platform.

The `APP_DISABLE_AUTH=True` setting is only for this local learning environment.

Production deployments should use proper authentication, authorization and HTTPS.

---

# 37. Local Development vs Production

This project intentionally runs locally using:

```text
kind
kubectl port-forward
```

A production environment would normally replace these with:

```text
Managed Kubernetes / Production Kubernetes
        |
        v
KServe
        |
        v
Ingress / Gateway
        |
        v
HTTPS
        |
        v
Authentication / Authorization
```

The model registry and artifact storage would also normally use the organization's managed MLflow and object-storage architecture.

---

# 38. Why MLflow and KServe Are Separate

MLflow and KServe have different responsibilities.

### MLflow

MLflow manages:

```text
Experiments
Runs
Parameters
Metrics
Artifacts
Model versions
Model registry
```

### KServe

KServe manages:

```text
Model serving
Kubernetes deployment
Serving lifecycle
Inference endpoints
Serving runtimes
Scaling and traffic management
```

### MLServer

MLServer is responsible for:

```text
Loading the model
Receiving inference requests
Executing model prediction
Returning predictions
```

### Models Web App

The Models Web App provides:

```text
Web UI
KServe resource visibility
Model endpoint management
```

The architecture is therefore:

```text
MLflow
  |
  | model version
  v
KServe
  |
  | model artifact
  v
MLServer
  |
  | prediction
  v
Client
```

---

# 39. Why Use a Custom Storage Initializer?

The custom storage initializer exists because the model location is:

```text
models:/diabetes-regressor/2
```

This is an MLflow model URI.

The initializer translates:

```text
MLflow model URI
        |
        v
MLflow API
        |
        v
Downloaded model artifact
        |
        v
/mnt/models
```

KServe then uses the existing MLServer runtime.

This means the custom component is responsible only for model acquisition; it is not a replacement for KServe's inference runtime.

---

# 40. Final Result

At the end of this project you have:

```text
                 DagsHub
                    |
                    | MLflow
                    |
                    v
        diabetes-regressor:v2
                    |
                    v
              KServe
                    |
                    v
          Custom Storage
           Initializer
                    |
                    v
              /mnt/models
                    |
                    v
              MLServer
                    |
                    v
           Prediction API
                    |
                    v
              Client / UI
```

The important MLOps separation is:

```text
Model Lifecycle
      |
      v
    MLflow

Model Serving
      |
      v
    KServe

Inference Runtime
      |
      v
   MLServer

User Interface
      |
      v
Models Web App
```

This project demonstrates a complete local Kubernetes-based ML serving workflow without embedding the model into the serving image and without introducing an unnecessary object-storage layer when the hosted MLflow service already provides model artifact storage.
