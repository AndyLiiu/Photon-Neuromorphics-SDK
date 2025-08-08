# Deployment Guide

## 🚀 Production Deployment of Photon Neuromorphics SDK

This guide covers deployment strategies for the Photon Neuromorphics SDK in production environments, from single-node installations to large-scale distributed systems.

## Quick Start

### Local Installation
```bash
# Standard installation
pip install photon-neuromorphics

# With hardware interface support
pip install photon-neuromorphics[hardware]

# Development installation
git clone https://github.com/danieleschmidt/Photon-Neuromorphics-SDK.git
cd Photon-Neuromorphics-SDK
pip install -e ".[dev,hardware]"
```

### Docker Deployment
```bash
# Pull latest image
docker pull photonneuro/photon-neuromorphics:latest

# Run interactive container
docker run -it --rm photonneuro/photon-neuromorphics:latest python

# Run with GPU support (if available)
docker run --gpus all -it photonneuro/photon-neuromorphics:latest
```

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   API Gateway   │    │   Backend       │
│   (Browser)     │────│   (NGINX)       │────│   (Python)      │
│   - WASM Module │    │   - Load Balance│    │   - PhotonNeuro │
│   - Visualize   │    │   - SSL Term.   │    │   - Hardware IF │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                    ┌─────────────────┐
                    │   Database      │
                    │   (PostgreSQL)  │
                    │   - Experiments │
                    │   - Calibration │
                    └─────────────────┘
```

## 🐳 Docker Deployment

### Basic Docker Setup

**Dockerfile** (already provided in repository):
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    libfftw3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .
RUN pip install -e .

# Expose port for API
EXPOSE 8000

# Default command
CMD ["python", "-m", "photon_neuro.api.server"]
```

### Docker Compose for Multi-Service Deployment

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  photon-neuro:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/photon_neuro
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped

  db:
    image: postgres:13
    environment:
      POSTGRES_DB: photon_neuro
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - photon-neuro
    restart: unless-stopped

volumes:
  postgres_data:
```

### Advanced Docker Configuration

**Multi-stage build for production**:
```dockerfile
# Build stage
FROM python:3.9 as builder

WORKDIR /build
COPY requirements*.txt ./
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

# Production stage
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopenblas0 \
    libfftw3-3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash photon
USER photon
WORKDIR /home/photon

# Install Python packages
COPY --from=builder /build/wheels /tmp/wheels
RUN pip install --user --no-cache /tmp/wheels/* && rm -rf /tmp/wheels

# Copy application
COPY --chown=photon:photon . /home/photon/app
WORKDIR /home/photon/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import photon_neuro; print('OK')" || exit 1

CMD ["python", "-m", "photon_neuro.api.server", "--host", "0.0.0.0", "--port", "8000"]
```

## ☸️ Kubernetes Deployment

### Namespace and ConfigMap

**namespace.yaml**:
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: photon-neuro
  labels:
    name: photon-neuro
```

**configmap.yaml**:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: photon-neuro-config
  namespace: photon-neuro
data:
  DATABASE_URL: "postgresql://postgres:password@postgres:5432/photon_neuro"
  REDIS_URL: "redis://redis:6379/0"
  LOG_LEVEL: "INFO"
  WASM_SIMD_ENABLED: "true"
  MAX_WORKERS: "4"
```

### Application Deployment

**deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: photon-neuro
  namespace: photon-neuro
  labels:
    app: photon-neuro
spec:
  replicas: 3
  selector:
    matchLabels:
      app: photon-neuro
  template:
    metadata:
      labels:
        app: photon-neuro
    spec:
      containers:
      - name: photon-neuro
        image: photonneuro/photon-neuromorphics:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: photon-neuro-config
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: logs-volume
          mountPath: /app/logs
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: photon-neuro-data
      - name: logs-volume
        persistentVolumeClaim:
          claimName: photon-neuro-logs
```

### Service and Ingress

**service.yaml**:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: photon-neuro-service
  namespace: photon-neuro
spec:
  selector:
    app: photon-neuro
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: ClusterIP
```

**ingress.yaml**:
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: photon-neuro-ingress
  namespace: photon-neuro
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.photon-neuro.io
    secretName: photon-neuro-tls
  rules:
  - host: api.photon-neuro.io
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: photon-neuro-service
            port:
              number: 80
```

### Horizontal Pod Autoscaler

**hpa.yaml**:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: photon-neuro-hpa
  namespace: photon-neuro
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: photon-neuro
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
```

### Persistent Storage

**pvc.yaml**:
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: photon-neuro-data
  namespace: photon-neuro
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: nfs-client
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: photon-neuro-logs
  namespace: photon-neuro
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 50Gi
  storageClassName: nfs-client
```

## ☁️ Cloud Platform Deployment

### AWS EKS Deployment

**terraform/aws-eks.tf**:
```hcl
provider "aws" {
  region = var.aws_region
}

module "eks" {
  source = "terraform-aws-modules/eks/aws"
  
  cluster_name    = "photon-neuro-cluster"
  cluster_version = "1.27"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  node_groups = {
    main = {
      desired_capacity = 3
      max_capacity     = 10
      min_capacity     = 1
      
      instance_types = ["m5.large"]
      
      k8s_labels = {
        Environment = var.environment
        Application = "photon-neuro"
      }
    }
  }
}

# RDS for database
resource "aws_db_instance" "photon_neuro_db" {
  identifier = "photon-neuro-db"
  
  engine         = "postgres"
  engine_version = "13.7"
  instance_class = "db.t3.micro"
  
  allocated_storage = 20
  storage_encrypted = true
  
  db_name  = "photon_neuro"
  username = "postgres"
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = true
}

# ElastiCache for Redis
resource "aws_elasticache_subnet_group" "main" {
  name       = "photon-neuro-cache-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_elasticache_cluster" "photon_neuro_redis" {
  cluster_id           = "photon-neuro-redis"
  engine               = "redis"
  node_type            = "cache.t3.micro"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379
  subnet_group_name    = aws_elasticache_subnet_group.main.name
  security_group_ids   = [aws_security_group.elasticache.id]
}
```

### Google Cloud GKE Deployment

**gcp-gke.yaml**:
```yaml
# Cloud Build configuration
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-t'
      - 'gcr.io/$PROJECT_ID/photon-neuromorphics:$COMMIT_SHA'
      - '.'
  
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'push'
      - 'gcr.io/$PROJECT_ID/photon-neuromorphics:$COMMIT_SHA'
  
  - name: 'gcr.io/cloud-builders/gke-deploy'
    args:
      - 'run'
      - '--filename=k8s/'
      - '--image=gcr.io/$PROJECT_ID/photon-neuromorphics:$COMMIT_SHA'
      - '--location=us-central1-a'
      - '--cluster=photon-neuro-cluster'
```

### Azure AKS Deployment

**azure-pipelines.yml**:
```yaml
trigger:
  branches:
    include:
    - main

pool:
  vmImage: 'ubuntu-latest'

variables:
  containerRegistry: 'photonneuro.azurecr.io'
  imageRepository: 'photon-neuromorphics'
  dockerfilePath: '$(Build.SourcesDirectory)/Dockerfile'
  tag: '$(Build.BuildId)'

stages:
- stage: Build
  displayName: Build and push stage
  jobs:
  - job: Build
    displayName: Build
    steps:
    - task: Docker@2
      displayName: Build and push image
      inputs:
        command: buildAndPush
        repository: $(imageRepository)
        dockerfile: $(dockerfilePath)
        containerRegistry: $(dockerRegistryServiceConnection)
        tags: |
          $(tag)

- stage: Deploy
  displayName: Deploy stage
  dependsOn: Build
  jobs:
  - deployment: Deploy
    displayName: Deploy
    environment: 'production'
    strategy:
      runOnce:
        deploy:
          steps:
          - task: KubernetesManifest@0
            displayName: Deploy to Kubernetes cluster
            inputs:
              action: deploy
              manifests: |
                $(Build.SourcesDirectory)/k8s/*.yaml
              containers: |
                $(containerRegistry)/$(imageRepository):$(tag)
```

## 📊 Monitoring and Observability

### Prometheus Monitoring

**prometheus-config.yaml**:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: photon-neuro
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    
    rule_files:
      - /etc/prometheus/rules/*.yml
    
    scrape_configs:
    - job_name: 'photon-neuro'
      kubernetes_sd_configs:
      - role: pod
        namespaces:
          names: [photon-neuro]
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: photon-neuro
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
```

### Grafana Dashboard

**grafana-dashboard.json** (excerpt):
```json
{
  "dashboard": {
    "id": null,
    "title": "Photon Neuromorphics Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(photon_neuro_requests_total[5m])",
            "legendFormat": "{{method}} {{status}}"
          }
        ]
      },
      {
        "title": "WASM Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "photon_neuro_wasm_execution_time",
            "legendFormat": "{{operation}}"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "process_resident_memory_bytes",
            "legendFormat": "Resident Memory"
          }
        ]
      }
    ]
  }
}
```

### Logging Configuration

**logging.yaml**:
```yaml
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
  json:
    class: pythonjsonlogger.jsonlogger.JsonFormatter
    format: '%(asctime)s %(name)s %(levelname)s %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: json
    stream: ext://sys.stdout
  
  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: standard
    filename: /app/logs/photon_neuro.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

loggers:
  photon_neuro:
    level: DEBUG
    handlers: [console, file]
    propagate: false
  
  photon_neuro.performance:
    level: INFO
    handlers: [console, file]
    propagate: false

root:
  level: INFO
  handlers: [console]
```

## 🔒 Security Configuration

### Network Policies

**network-policy.yaml**:
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: photon-neuro-network-policy
  namespace: photon-neuro
spec:
  podSelector:
    matchLabels:
      app: photon-neuro
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    - podSelector:
        matchLabels:
          app: photon-neuro
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS outbound
    - protocol: TCP
      port: 53   # DNS
    - protocol: UDP
      port: 53   # DNS
```

### Pod Security Policy

**pod-security-policy.yaml**:
```yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: photon-neuro-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

## 📈 Performance Optimization

### Resource Allocation Guidelines

| Component | CPU Request | CPU Limit | Memory Request | Memory Limit |
|-----------|-------------|-----------|----------------|--------------|
| API Server | 250m | 1000m | 512Mi | 2Gi |
| WASM Worker | 500m | 2000m | 1Gi | 4Gi |
| Database | 500m | 1000m | 1Gi | 2Gi |
| Redis | 100m | 500m | 256Mi | 1Gi |

### Auto-scaling Configuration

```yaml
# Custom metrics for photonic workloads
apiVersion: v2
kind: HorizontalPodAutoscaler
metadata:
  name: photon-neuro-custom-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: photon-neuro
  minReplicas: 2
  maxReplicas: 50
  metrics:
  - type: Pods
    pods:
      metric:
        name: photonic_operations_per_second
      target:
        type: AverageValue
        averageValue: "1000"
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## 🚨 Troubleshooting

### Common Issues

1. **WASM Module Loading Failures**
   ```bash
   # Check WASM support
   kubectl exec -it photon-neuro-pod -- python -c "from photon_neuro.wasm import initialize_wasm; print('WASM OK')"
   
   # Debug WASM initialization
   kubectl logs photon-neuro-pod | grep -i wasm
   ```

2. **Hardware Interface Problems**
   ```bash
   # Check hardware connectivity
   kubectl exec -it photon-neuro-pod -- python -c "import pyvisa; rm = pyvisa.ResourceManager(); print(rm.list_resources())"
   ```

3. **Performance Issues**
   ```bash
   # Monitor resource usage
   kubectl top pods -n photon-neuro
   
   # Check HPA status
   kubectl get hpa -n photon-neuro
   
   # View detailed metrics
   kubectl describe hpa photon-neuro-hpa -n photon-neuro
   ```

### Debug Commands

```bash
# Get pod status
kubectl get pods -n photon-neuro -o wide

# View pod logs
kubectl logs -f deployment/photon-neuro -n photon-neuro

# Execute debug commands
kubectl exec -it photon-neuro-pod -n photon-neuro -- /bin/bash

# Check service endpoints
kubectl get endpoints -n photon-neuro

# View resource usage
kubectl top nodes
kubectl top pods -n photon-neuro --containers
```

## 🎯 Production Checklist

### Pre-Deployment
- [ ] Security scan completed (Bandit, Safety)
- [ ] Performance benchmarks meet requirements
- [ ] Database migrations tested
- [ ] Backup and recovery procedures verified
- [ ] Monitoring and alerting configured
- [ ] SSL certificates installed and tested
- [ ] Network policies applied
- [ ] Resource limits configured
- [ ] Health checks implemented
- [ ] Documentation updated

### Post-Deployment
- [ ] Application health verified
- [ ] Performance metrics baseline established
- [ ] Monitoring dashboards accessible
- [ ] Log aggregation working
- [ ] Auto-scaling tested
- [ ] Backup verification completed
- [ ] Disaster recovery plan tested
- [ ] Team access and permissions verified
- [ ] Runbook created and tested

---

For additional support, consult the [troubleshooting guide](TROUBLESHOOTING.md) or contact the development team.