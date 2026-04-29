# Getting Started with Kubernetes — An End-to-End ML Project

This is a hands-on, beginner-friendly tutorial that takes you all the way from training a tiny machine-learning model to running it on a **local Kubernetes cluster**. Every step is explained, every command can be copy-pasted, and the final result is a containerised FastAPI service serving live predictions.

By the end you will have:

1. Trained a scikit-learn classifier on the classic Iris dataset.
2. Wrapped it in a **FastAPI** REST service with `/predict` and `/health` endpoints.
3. Packaged everything into a **Docker** image.
4. Deployed two replicas of that image to a local **Kubernetes** cluster (minikube), exposed it through a Service, scaled it up and down, and performed a rolling update.

No cloud account is required — everything runs on your laptop.

---

## Table of Contents

1. [What is Kubernetes (and why bother)?](#1-what-is-kubernetes-and-why-bother)
2. [Prerequisites](#2-prerequisites)
3. [Project Structure](#3-project-structure)
4. [Step 1 — Set Up the Python Environment](#step-1--set-up-the-python-environment)
5. [Step 2 — Train the ML Model](#step-2--train-the-ml-model)
6. [Step 3 — Understand the FastAPI Service](#step-3--understand-the-fastapi-service)
7. [Step 4 — Run the API Locally](#step-4--run-the-api-locally)
8. [Step 5 — Dockerize the Application](#step-5--dockerize-the-application)
9. [Step 6 — Run the Container](#step-6--run-the-container)
10. [Step 7 — Install Kubernetes Tools](#step-7--install-kubernetes-tools)
11. [Step 8 — Kubernetes Core Concepts](#step-8--kubernetes-core-concepts)
12. [Step 9 — Make the Image Available to Minikube](#step-9--make-the-image-available-to-minikube)
13. [Step 10 — Apply the Manifests](#step-10--apply-the-manifests)
14. [Step 11 — Inspect the Cluster](#step-11--inspect-the-cluster)
15. [Step 12 — Call the Service](#step-12--call-the-service)
16. [Step 13 — Scale the Deployment](#step-13--scale-the-deployment)
17. [Step 14 — Rolling Update with a New Model](#step-14--rolling-update-with-a-new-model)
18. [Step 15 — Cleanup](#step-15--cleanup)
19. [Troubleshooting](#troubleshooting)
20. [Where to Go Next](#where-to-go-next)

---

## 1. What is Kubernetes (and why bother)?

**Kubernetes** (often abbreviated **k8s** — there are 8 letters between the "k" and the "s") is a system that runs and manages containers across one or more machines. You describe the *desired state* of your application in YAML files ("I want 2 replicas of this image, exposed on port 80"), and Kubernetes takes responsibility for **making reality match that description** — restarting crashed containers, replacing failed nodes, rolling out new versions without downtime, and load-balancing traffic.

For an ML engineer, the practical wins are:

- **Reproducible deployments** — the same manifests work on your laptop, on staging, and on a production cluster.
- **Self-healing** — if a Pod crashes (out-of-memory, bad input), Kubernetes spins up a new one.
- **Scaling** — change one number and you have 10 replicas instead of 2.
- **Zero-downtime updates** — push a new model and old Pods are replaced gradually.

In this tutorial we use **minikube**, which runs a complete single-node cluster inside a VM or container on your machine.

---

## 2. Prerequisites

You will need the following installed. Versions in parentheses are what this tutorial was written against; newer versions almost certainly work as well.

| Tool | Why | Check with |
|------|-----|-----------|
| Python (≥ 3.10) | Train the model and run FastAPI locally | `python3 --version` |
| Docker Desktop (or Docker Engine + Docker CLI) | Build and run the container image | `docker --version` |
| kubectl (≥ 1.28) | The Kubernetes command-line client | `kubectl version --client` |
| minikube (≥ 1.32) | Local Kubernetes cluster | `minikube version` |
| curl | Send HTTP requests for testing | `curl --version` |

Installation pointers:

- **macOS** — `brew install python kubectl minikube` and install Docker Desktop from docker.com.
- **Linux** — use your package manager for Python and Docker, then follow the official kubectl and minikube install pages.
- **Windows** — install Docker Desktop, then `winget install Kubernetes.kubectl Kubernetes.minikube` (or use Chocolatey).

> **Tip:** Docker Desktop also ships its own Kubernetes feature. You *can* use that instead of minikube, but the commands in [Step 9](#step-9--make-the-image-available-to-minikube) about loading images differ slightly. Stick with minikube for the smoothest experience.

---

## 3. Project Structure

```
kubernetes_get_started/
├── README.md              ← you are here
├── requirements.txt       ← Python dependencies
├── train.py               ← trains and saves the ML model
├── Dockerfile             ← container build recipe
├── .dockerignore
├── .gitignore
├── app/
│   ├── __init__.py
│   ├── main.py            ← FastAPI service
│   └── model.joblib       ← created by train.py (not committed)
└── k8s/
    ├── deployment.yaml    ← describes the Pods
    └── service.yaml       ← exposes the Pods to the network
```

Create that directory layout exactly — every command below assumes it.

---

## Step 1 — Set Up the Python Environment

A virtual environment isolates your project's packages from the rest of your system.

```bash
cd kubernetes_get_started

python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins the libraries we need: FastAPI (web framework), Uvicorn (ASGI server), scikit-learn (training), joblib (model serialisation), numpy, and pydantic (request validation).

> Why pin versions? In containers and in Kubernetes you want the *exact same* libraries every time the image is built. Floating versions (`fastapi>=0.100`) can produce different behaviour on different days.

---

## Step 2 — Train the ML Model

The `train.py` script trains a logistic-regression classifier on the Iris dataset (a 150-row toy dataset built into scikit-learn). The trained model is wrapped in a `Pipeline` together with a `StandardScaler`, then saved to `app/model.joblib` using `joblib.dump`.

Run:

```bash
python train.py
```

Expected output:

```
Test accuracy: 0.9667
Model saved to .../app/model.joblib
```

Take a moment to read [train.py](train.py):

- We load `iris` and split it into train/test sets (`train_test_split`).
- The `Pipeline` chains pre-processing (`StandardScaler`) and the classifier (`LogisticRegression`) so they are persisted together as a single object.
- We persist not just the model but also the human-readable `target_names` (`setosa`, `versicolor`, `virginica`) so the API can return the predicted class name.

> **Why save into the `app/` folder?** Because the Docker build copies the whole `app/` directory into the image. Anything saved there gets baked into the container.

---

## Step 3 — Understand the FastAPI Service

Open [app/main.py](app/main.py). The important pieces are:

- **`IrisFeatures`** — a Pydantic model. FastAPI automatically validates incoming JSON against it and returns a 422 error if a field is missing or the wrong type. No manual parsing required.
- **`@app.on_event("startup")`** — this hook runs once when the process starts. It loads the joblib bundle into `app.state` so each request avoids the cost of re-reading from disk.
- **`/health`** — a tiny endpoint that returns `{"status": "ok"}`. Kubernetes will hit it constantly to decide whether the Pod is alive and ready (more on that in [Step 8](#step-8--kubernetes-core-concepts)).
- **`/predict`** — accepts the four flower measurements and returns the predicted class name, the class id, and per-class probabilities.
- **`/`** — returns the hostname (`os.getenv("HOSTNAME")`). Inside Kubernetes, the hostname equals the **Pod name**, which makes it easy to see *which* replica handled your request when we scale up.

---

## Step 4 — Run the API Locally

Before containerising, prove the service works on your machine:

```bash
uvicorn app.main:app --reload --port 8000
```

`--reload` restarts Uvicorn whenever you edit a file — handy during development, but never use it in production.

Open the auto-generated docs at <http://localhost:8000/docs>. FastAPI gives you a Swagger UI for free. Try `POST /predict` with this body:

```json
{
  "sepal_length": 5.1,
  "sepal_width":  3.5,
  "petal_length": 1.4,
  "petal_width":  0.2
}
```

Or from a second terminal:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

You should see something like:

```json
{
  "predicted_class": "setosa",
  "predicted_class_id": 0,
  "probabilities": [0.97, 0.03, 0.0]
}
```

Stop the server with `Ctrl+C` once you are satisfied.

---

## Step 5 — Dockerize the Application

A **Docker image** is a self-contained snapshot of your application *plus everything it needs to run* — the right Python version, the right libraries, your code, the trained model. A **container** is an instance of that image, running as an isolated process on your machine (or your Kubernetes cluster).

Look at the [Dockerfile](Dockerfile):

```Dockerfile
FROM python:3.11-slim                  # 1. Start from an official, minimal Python image

ENV PYTHONUNBUFFERED=1 \               # 2. Sensible Python defaults for containers
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /code                          # 3. Everything below runs inside /code

COPY requirements.txt ./               # 4. Copy + install deps FIRST (better cache)
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app                         # 5. Then copy the application code (and the model)

EXPOSE 8000                            # 6. Document that the container listens on 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Two things deserve highlighting:

- **Layer caching.** Each instruction creates a layer. By copying `requirements.txt` and running `pip install` *before* copying the app code, Docker reuses the cached layer with all the installed wheels every time you change application code but not the dependencies. This makes rebuilds dramatically faster.
- **`--host 0.0.0.0`.** Inside a container, `127.0.0.1` is *only* reachable from inside the container itself. To expose the server to the outside world you must bind to `0.0.0.0`.

> **Important:** The image bakes in `app/model.joblib`. Make sure you ran `python train.py` in [Step 2](#step-2--train-the-ml-model) so the file exists before you build.

Build the image:

```bash
docker build -t iris-api:1.0.0 .
```

`-t iris-api:1.0.0` tags the image with a name (`iris-api`) and a version (`1.0.0`). Verify:

```bash
docker images | grep iris-api
```

---

## Step 6 — Run the Container

```bash
docker run --rm -p 8000:8000 --name iris iris-api:1.0.0
```

- `--rm` removes the container automatically when it exits.
- `-p 8000:8000` maps host port → container port.
- `--name iris` gives it a friendly name.

In a second terminal, repeat the `curl` from [Step 4](#step-4--run-the-api-locally) and confirm you still get a prediction. Then stop the container with `Ctrl+C`.

Congratulations — you now have a portable, reproducible artefact. Anyone with Docker can run your model with one command, regardless of OS or local Python setup.

---

## Step 7 — Install Kubernetes Tools

If you haven't already:

```bash
# macOS
brew install kubectl minikube

# Linux (one of many options — see the official docs for your distro)
# kubectl: https://kubernetes.io/docs/tasks/tools/
# minikube: https://minikube.sigs.k8s.io/docs/start/
```

Start the cluster:

```bash
minikube start --driver=docker
```

The first start downloads a Kubernetes node image (~1 GB) and may take a few minutes. Subsequent starts are fast. When it finishes:

```bash
kubectl get nodes
```

You should see a single node named `minikube` with status `Ready`. `kubectl` is already pointed at the cluster — minikube updated your kube-config automatically.

---

## Step 8 — Kubernetes Core Concepts

Before we deploy, here are the four objects we will use. Understanding these unlocks everything else.

### Pod
A **Pod** is the smallest unit Kubernetes runs. It wraps one (occasionally several) containers that share a network and storage. Pods are *cattle, not pets*: you never name a single Pod or care about it individually — if it dies, another takes its place.

### Deployment
You almost never create Pods directly. A **Deployment** says: *"keep N Pods of this image alive, and when I change the spec, replace them gracefully."* It owns a **ReplicaSet**, which owns the Pods. When you do a rolling update, the Deployment creates a new ReplicaSet and slowly shifts traffic.

### Service
Pods come and go, so their IP addresses change. A **Service** gives you a *stable* virtual IP (and DNS name) that load-balances across whichever Pods currently match its label selector. Types we care about:

- `ClusterIP` (default) — only reachable inside the cluster.
- `NodePort` — exposes the service on a fixed port of every node. Perfect for local development. We use this.
- `LoadBalancer` — provisions a real cloud load balancer (only meaningful on cloud-managed clusters).

### Probes
Each container can declare two health probes:

- **Readiness probe** — "is this Pod ready to receive traffic?" If it fails, the Service stops routing to that Pod (but the Pod is *not* killed). Useful while the model is still loading.
- **Liveness probe** — "is this Pod still healthy?" If it fails, Kubernetes kills and restarts the container.

Both probes in our manifest hit `GET /health` on port 8000.

### Manifests we will apply

[k8s/deployment.yaml](k8s/deployment.yaml):

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: iris-api
spec:
  replicas: 2                       # ← run two Pods
  selector:
    matchLabels:
      app: iris-api                 # ← Deployment manages Pods with this label
  template:
    metadata:
      labels:
        app: iris-api               # ← Pods are tagged with this label
    spec:
      containers:
        - name: iris-api
          image: iris-api:1.0.0     # ← the image we built in Step 5
          imagePullPolicy: IfNotPresent   # ← do NOT try to pull from Docker Hub
          ports:
            - containerPort: 8000
          resources:                # ← guard rails so one Pod cannot eat the whole node
            requests: { cpu: "100m", memory: "128Mi" }
            limits:   { cpu: "500m", memory: "256Mi" }
          readinessProbe: { httpGet: { path: /health, port: 8000 }, initialDelaySeconds: 5,  periodSeconds: 5  }
          livenessProbe:  { httpGet: { path: /health, port: 8000 }, initialDelaySeconds: 15, periodSeconds: 20 }
```

`imagePullPolicy: IfNotPresent` is critical: it tells Kubernetes "if the image is already on the node, use it; do not try to pull it from a remote registry." Since our image lives only on our laptop, that is exactly what we want.

[k8s/service.yaml](k8s/service.yaml):

```yaml
apiVersion: v1
kind: Service
metadata:
  name: iris-api
spec:
  type: NodePort
  selector:
    app: iris-api                   # ← routes to all Pods with this label
  ports:
    - port: 80                      # ← the Service's port (cluster-internal)
      targetPort: 8000              # ← the Pod's port
      nodePort: 30080               # ← the port exposed on the node
```

---

## Step 9 — Make the Image Available to Minikube

Here is the most common gotcha for beginners.

Minikube runs in its own Docker daemon (or VM). When you ran `docker build` in [Step 5](#step-5--dockerize-the-application), the image landed in your *host's* Docker daemon — minikube cannot see it. We must move the image *into* the minikube node.

The simplest one-liner:

```bash
minikube image load iris-api:1.0.0
```

This copies the image from your host daemon into minikube's image store. Verify:

```bash
minikube image ls | grep iris-api
```

> Alternative: you can also point your shell's Docker CLI directly at minikube's daemon (`eval $(minikube docker-env)`) and run `docker build` again — that builds the image straight inside minikube and skips the load step. Either approach works.

---

## Step 10 — Apply the Manifests

```bash
kubectl apply -f k8s/
```

`kubectl apply` is **declarative**: it diffs what is in the cluster against your YAML and reconciles the difference. Running it twice with the same files is a no-op.

Expected output:

```
deployment.apps/iris-api created
service/iris-api created
```

---

## Step 11 — Inspect the Cluster

This is the part where Kubernetes starts to feel useful. A few essential commands:

```bash
kubectl get deployments              # high-level: how many replicas are ready?
kubectl get pods -o wide             # individual Pods, their node, IPs, and status
kubectl get services                 # services with their ports
kubectl describe deployment iris-api # full event log + spec
kubectl logs -l app=iris-api --tail=50  # logs from all matching Pods
```

Wait until both Pods are `Running` and `READY 1/1`. The first time, this can take 30-60 seconds because the readiness probe waits 5 seconds before its first check.

If something is off, `kubectl describe pod <pod-name>` is your friend — the **Events** section at the bottom usually pinpoints the problem.

---

## Step 12 — Call the Service

Get the URL minikube assigns to the NodePort:

```bash
minikube service iris-api --url
```

This prints something like `http://192.168.49.2:30080`. Use it:

```bash
URL=$(minikube service iris-api --url)

curl -X POST $URL/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":6.7,"sepal_width":3.0,"petal_length":5.2,"petal_width":2.3}'
```

Expected response:

```json
{
  "predicted_class": "virginica",
  "predicted_class_id": 2,
  "probabilities": [0.0, 0.06, 0.94]
}
```

Hit the root endpoint a few times in a row:

```bash
for i in $(seq 1 6); do curl -s $URL/ ; echo; done
```

Notice the `pod` field changes — Kubernetes is load-balancing requests across both replicas.

> **If `minikube service` is hanging or your driver doesn't expose NodePorts** (common on Docker driver on macOS), use port-forwarding instead:
>
> ```bash
> kubectl port-forward service/iris-api 8000:80
> ```
>
> Then call `http://localhost:8000/predict`.

---

## Step 13 — Scale the Deployment

Bump the number of replicas to 5:

```bash
kubectl scale deployment iris-api --replicas=5
kubectl get pods -w           # watch new Pods come up; Ctrl+C when done
```

Repeat the curl loop from Step 12 — you should now see up to five different pod names in the responses.

Scale back down:

```bash
kubectl scale deployment iris-api --replicas=2
```

The two surplus Pods are gracefully terminated.

---

## Step 14 — Rolling Update with a New Model

Let's simulate releasing version 2. Edit [train.py](train.py) — for example, change `LogisticRegression(max_iter=200, ...)` to `LogisticRegression(max_iter=500, C=2.0, ...)` — and rerun training:

```bash
python train.py
```

Build a new image *with a new tag*:

```bash
docker build -t iris-api:1.1.0 .
minikube image load iris-api:1.1.0
```

> Why a new tag? If you reuse `1.0.0`, Kubernetes thinks the spec is unchanged and does not roll out anything. Tags are how Kubernetes detects "this is a new version."

Trigger the rollout:

```bash
kubectl set image deployment/iris-api iris-api=iris-api:1.1.0
kubectl rollout status deployment/iris-api
```

Kubernetes creates Pods with the new image, waits for their readiness probe, then terminates the old ones — one or two at a time depending on the rollout strategy. Throughout the whole thing the Service keeps serving traffic.

To roll back if something is wrong:

```bash
kubectl rollout undo deployment/iris-api
```

---

## Step 15 — Cleanup

When you're done playing:

```bash
kubectl delete -f k8s/        # remove Deployment + Service
minikube stop                 # stop the cluster (keep state for next time)
# minikube delete             # nuke the cluster entirely
```

To remove the local images:

```bash
docker rmi iris-api:1.0.0 iris-api:1.1.0
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|------|
| `ErrImagePull` / `ImagePullBackOff` in `kubectl get pods` | The image isn't in minikube's daemon | Run `minikube image load iris-api:1.0.0` and confirm with `minikube image ls` |
| `CrashLoopBackOff` | The container starts but exits | `kubectl logs <pod>` — most often "Model file not found" because you forgot to run `python train.py` before building the image |
| Pod stays in `Pending` | Cluster has no resources | `kubectl describe pod <name>` will show `Insufficient cpu/memory`. Lower the `requests` in the Deployment or increase minikube resources: `minikube start --cpus=4 --memory=4096` |
| `minikube service` hangs | Driver doesn't expose NodePorts cleanly | Use `kubectl port-forward service/iris-api 8000:80` instead |
| `connection refused` from inside the cluster but not host | App bound to `127.0.0.1` instead of `0.0.0.0` | Ensure the Dockerfile's `CMD` includes `--host 0.0.0.0` |
| Readiness probe fails | App takes too long to load the model | Increase `initialDelaySeconds` in the Deployment |
| You changed the Deployment YAML but nothing happens | Kubernetes only re-rolls when *something it tracks* changes | Either change the image tag, or run `kubectl rollout restart deployment/iris-api` |

Useful debugging commands:

```bash
kubectl get events --sort-by=.lastTimestamp
kubectl describe pod <pod-name>
kubectl logs <pod-name> --previous            # logs from the previous crash
kubectl exec -it <pod-name> -- /bin/sh        # shell inside the Pod
```

---

## Where to Go Next

You now have a working foundation. Natural next steps:

- **ConfigMaps and Secrets** — externalise environment variables (model path, log level, API keys) without rebuilding the image.
- **Ingress** — a more production-friendly way to expose HTTP services than NodePort. Try `minikube addons enable ingress`.
- **Horizontal Pod Autoscaler (HPA)** — scale replicas automatically based on CPU or custom metrics.
- **Helm** — package your manifests so they can be installed with one command and parameterised per environment.
- **Persistent Volumes** — store the model on a volume instead of inside the image, so updates don't require a rebuild.
- **Observability** — add Prometheus + Grafana for metrics, and structured logging.
- **CI/CD** — build images and apply manifests automatically on every git push (GitHub Actions, GitLab CI, ArgoCD, Flux).

Happy shipping!
