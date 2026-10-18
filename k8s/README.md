# Kubernetes deployment

The default manifests deploy the API and Kafka scoring worker only. PostgreSQL,
Kafka, and Cassandra are external services; their operators/managed-service lifecycle
is intentionally separate from this stateless application deployment. The API image
contains the existing real model artifact. Do not use the synthetic model image.

## Local kind workflow

1. Install Docker Desktop with WSL2, `kind`, and `kubectl`.
2. Build the API image and the streaming-enabled worker image:

   ```powershell
   docker build -t fraud-platform-api:local .
   docker build --build-arg INSTALL_STREAMING=true -t fraud-platform-worker:local .
   kind create cluster --name fraud-platform
   kind load docker-image fraud-platform-api:local fraud-platform-worker:local --name fraud-platform
   ```

3. Create the namespace, then create the credential Secret without putting credentials
   in YAML. Set a real database URL and API key in your PowerShell session first:

   ```powershell
   kubectl apply -f k8s/namespace.yaml
   kubectl create secret generic fraud-platform-secrets -n fraud-platform `
     --from-literal=FRAUD_API_KEY="$env:FRAUD_API_KEY" `
     --from-literal=DATABASE_URL="$env:DATABASE_URL"
   ```

4. Set the non-secret broker/service endpoints for the target cluster by editing a
   local ConfigMap overlay or applying the ConfigMap with those values. Required keys
   for the worker are `KAFKA_BOOTSTRAP_SERVERS` and `CASSANDRA_HOSTS`.
5. Validate and apply:

   ```powershell
   kubectl kustomize k8s
   kubectl apply -k k8s
   kubectl -n fraud-platform rollout status deployment/fraud-api
   kubectl -n fraud-platform get pods,services,hpa
   ```

The API readiness probe waits for PostgreSQL migrations and connectivity. Apply
Alembic migrations as a controlled release step before the API rollout:

```powershell
docker run --rm --env-file .env `
  -e DATABASE_URL="$env:DATABASE_URL" fraud-platform-api:local alembic upgrade head
```

The `k8s/ingress.yaml` is deliberately excluded from the default Kustomize resources.
Apply it only after installing an ingress controller and configuring TLS/authentication.
The HPA requires the cluster Metrics Server. CPU scaling is configured; Kafka consumer
lag is not currently exported and is not used for autoscaling.

Rollback the API deployment with:

```powershell
kubectl -n fraud-platform rollout undo deployment/fraud-api
kubectl -n fraud-platform rollout status deployment/fraud-api
```

These manifests are statically reviewable configuration only. No Kubernetes cluster
or `kubectl` is available in the development workspace, so a successful deployment has
not been verified here.
