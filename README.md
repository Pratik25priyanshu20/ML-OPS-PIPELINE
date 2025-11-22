 Bank Marketing ML API — Production-Grade MLOps Pipeline on Kubernetes

A fully containerized, autoscaling, production-ready Machine Learning Prediction API deployed on Kubernetes with:
	•	FastAPI inference server
	•	Scikit-learn ML model + metadata
	•	Kubernetes Deployment + Service + HPA
	•	Prometheus metrics scraping
	•	Grafana dashboards (business, model, system)
	•	End-to-end observability
	•	Load-tested autoscaling
	•	Clean repository structure
	•	Production monitoring workflow



Project Overview

This system exposes a real-time ML inference API that predicts whether a customer will subscribe to a term deposit (based on the UCI Bank Marketing dataset).

The project is fully productionized using:
	•	Docker + Kubernetes
	•	Horizontal Pod Autoscaler (HPA)
	•	Prometheus Operator
	•	Grafana dashboards
	•	Prometheus FastAPI instrumentation
	•	Load testing & autoscaling validation



flowchart TD

A[Client / Load Generator] -->|HTTP Requests| B[FastAPI ML Inference API]

B -->|Prometheus Instrumentation| C[/metrics Endpoint/]

C -->|scrape| D[Prometheus Operator]

D -->|Query| E[Grafana Dashboards]

B -->|CPU Load| F[Horizontal Pod Autoscaler]

F -->|Scale Up/Down| G[Bank Marketing API Deployment]

G --> H[Multiple API Pods]



├── src/
│   ├── serve/api.py               # FastAPI ML inference server
│   ├── data/feature_engineering.py
│   ├── models/train.py
│   └── ...
├── models/
│   ├── best_model.pkl
│   └── best_model_metadata.yaml
├── data/features/
│   ├── scaler.pkl
│   └── feature_columns.pkl
├── k8s/
│   ├── api-deployment.yaml
│   ├── api-service.yaml
│   └── api-hpa.yaml
├── monitoring/
│   ├── grafana/
│   │   ├── dashboards/
│   │   │   ├── business_metrics.json
│   │   │   ├── model_performance.json
│   │   │   └── system_metrics.json
│   │   ├── dashboards-configmap.yaml
│   │   └── values.yaml
│   └── prometheus/
│       └── scrape-configs.yaml
└── README.md



 Machine Learning Pipeline

Preprocessing
	•	Clean raw CSV
	•	Handle categorical variables
	•	One-hot encode with drop_first=True
	•	StandardScaler for numeric features
	•	Save feature_columns.pkl
	•	Save scaler.pkl

Model
	•	XGBoost or RandomForest classifier
	•	Stored as best_model.pkl
	•	Metadata stored in best_model_metadata.yaml:
	•	model_name
	•	model_type
	•	metrics
	•	saved_at
	•	version

Prediction Flow
	1.	Request validated using Pydantic
	2.	Convert to training feature schema
	3.	FeatureEngineer → engineer_features
	4.	One-hot encoding
	5.	Align columns with training
	6.	Apply scaler
	7.	Run model inference
	8.	Return prediction + probability + confidence



REST API Endpoints

Health Check

GET /health
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "2025-11-19",
  "uptime_seconds": 1203.23
}






Monitoring and Observability

This project includes a complete, production-grade monitoring and observability system built using:
	•	Prometheus (metrics collection and time-series storage)
	•	Grafana (dashboards and visualization)
	•	kube-prometheus-stack (Kubernetes-native monitoring)
	•	Custom application metrics exposed by the FastAPI inference service

The monitoring layer provides visibility across application performance, model behavior, and system health.


1. Application-Level Metrics (FastAPI)

The ML inference API exposes Prometheus-compatible metrics at: GET /metrics

These metrics are collected using the prometheus_client library.


| Metric Name                                         | Type       | Purpose                                 |
|-----------------------------------------------------|------------|------------------------------------------|
| `predictions_total{prediction="yes/no", model_version}` | Counter    | Counts predictions by class label        |
| `predictions_created`                               | Gauge      | Timestamp when the counter was created   |
| `prediction_latency_seconds`                        | Histogram  | Inference latency distribution           |
| `http_requests_total{handler, method, status}`      | Counter    | Tracks API requests by handler & status  |
| `http_request_duration_seconds`                     | Histogram  | Endpoint-specific latency measurements   |




These metrics allow tracking of:
	•	Number of predictions performed
	•	Ratio of positive vs negative predictions
	•	Latency behavior of the ML model
	•	API throughput and performance



2. Kubernetes Monitoring Stack

The monitoring system is deployed using: helm install monitoring prometheus-community/kube-prometheus-stack


kube-prometheus-stack includes:
	•	Prometheus
	•	Grafana
	•	Alertmanager
	•	Node exporter
	•	Pod/Container scraping via ServiceMonitor and PodMonitor
	•	Automatic discovery of annotated services

This provides end-to-end visibility into Kubernetes workloads, cluster nodes, and API behavior.




3. Prometheus Scraping Configuration

The FastAPI inference service is automatically scraped by Prometheus due to the following annotations applied on the Kubernetes Service:

yaml
metadata:
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8000"
    prometheus.io/path: "/metrics"






4. Grafana Dashboard Provisioning (GitOps-Friendly)

Grafana dashboards in this project are fully provisioned, meaning:
	•	Dashboards are stored as JSON files in the repository
	•	No manual UI changes are needed after deployment
	•	Dashboards are loaded automatically into Grafana via a ConfigMap
	•	Dashboards are version-controlled

Directory Structure:

monitoring/grafana/
  ├── dashboards/
  │      business_metrics.json
  │      model_performance.json
  │      system_metrics.json
  ├── dashboards-configmap.yaml
  └── values.yaml



Grafana Sidecar Auto-Loader

values.yaml enables automatic loading:
grafana:
  sidecar:
    dashboards:
      enabled: true
      label: grafana_dashboard
      folder: Bank Marketing API

Grafana reads dashboards from the ConfigMap automatically whenever it restarts.


5. Overview of Dashboards

5.1 Business Metrics Dashboard

Provides high-level business performance indicators:
	•	Total predictions
	•	Yes predictions
	•	No predictions
	•	Conversion rate (Yes / Total)
	•	Predictions over time

5.2 Model Performance Dashboard

Focuses on machine learning model behavior:
	•	P50, P90, P99 latency (derived from histogram buckets)
	•	Latency distribution
	•	Request throughput
	•	Model degradation or drift indicators

5.3 System Metrics Dashboard

Monitors cluster and infrastructure health:
	•	CPU usage per pod
	•	Memory usage per pod
	•	Horizontal Pod Autoscaler replicas
	•	Pod restarts

⸻

6. Updating Dashboards (GitOps Workflow)

Because dashboards are provisioned, updates must follow these steps:

Step 1 — Make changes in Grafana UI

Using the “Edit” button on a panel.

Step 2 — Export dashboard JSON

Click: Save > Save JSON to file

Step 3 — Replace file in repository

Overwrite the corresponding file:
monitoring/grafana/dashboards/{dashboard_name}.json

Step 4 — Regenerate ConfigMap

kubectl create configmap bank-marketing-api-dashboards \
  -n monitoring \
  --from-file=monitoring/grafana/dashboards/business_metrics.json \
  --from-file=monitoring/grafana/dashboards/model_performance.json \
  --from-file=monitoring/grafana/dashboards/system_metrics.json \
  -o yaml --dry-run=client > monitoring/grafana/dashboards-configmap.yaml


Step 5 — Apply changes

kubectl apply -f monitoring/grafana/dashboards-configmap.yaml -n monitoring
kubectl delete pod -n monitoring -l app.kubernetes.io/name=grafana





7. Validation

7.1 Prometheus Web UI

Port forward Prometheus:
kubectl port-forward svc/monitoring-kube-prometheus-prometheus -n monitoring 9090

visit: http://localhost:9090


verify metrics:
 predictions_total
prediction_latency_seconds_bucket
http_requests_total



7.2 Grafana

Port forward Grafana: kubectl port-forward svc/monitoring-grafana -n monitoring 3000:80


open : http://localhost:3000

dashboards appear under : Dashboards > Bank Marketing API