# AI-Proactive Road Safety System (APRSS)

A predictive AI system that anticipates road accidents **before they happen**. APRSS combines a CARLA/SUMO co-simulation environment, YOLOv8 visual perception, and an XGBoost risk classifier to detect and score dangerous driving situations in real time, then serves that intelligence through a FastAPI backend and a live Streamlit dashboard.

> Simulate danger → learn from it → predict it → visualize it.

---

## Table of Contents

- [Why This Project Exists](#why-this-project-exists)
- [System Architecture](#system-architecture)
- [Phase 1 — Simulation & Dataset Generation](#phase-1--simulation--dataset-generation)
- [Phase 2 — Exploratory Data Analysis](#phase-2--exploratory-data-analysis)
- [Phase 3 — Model Training](#phase-3--model-training)
- [Phase 4 — Backend & Live Inference (Docker)](#phase-4--backend--live-inference-docker)
- [Phase 5 — Streamlit Dashboard](#phase-5--streamlit-dashboard)
- [Phase 6 — Cloud Deployment (Azure)](#phase-6--cloud-deployment-azure)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Live Demo](#live-demo)
- [Screenshots](#screenshots)
- [Getting Started](#getting-started)
- [Results](#results)
- [Future Work](#future-work)

---

## Why This Project Exists

Most road-safety systems only react after a near-miss has already happened. Real-world accident data is also scarce, unlabeled, and impossible to safely reproduce for training purposes — you can't ask real drivers to almost cause a pedestrian collision so you can capture the sensor data.

APRSS solves both problems at once by **generating its own labeled danger**: a co-simulated CARLA + SUMO environment deliberately injects realistic hazardous scenarios (cut-ins, sudden braking, jaywalking, etc.), captures exactly what the car's sensors would see, and scores exactly how dangerous each moment was. That synthetic-but-physically-grounded dataset is what makes the downstream perception and risk models possible.

---

## System Architecture

```
CARLA (3D world) ─┐
                   ├── Co-Simulation ──► Sensor Capture ──► Risk Labeling ──► Dataset
SUMO (traffic)  ───┘                     (RGB / Depth /      (TTC, proximity,
                                           LiDAR)              scenario, pedestrian
                                                                weighting)
                                                                     │
                                                                     ▼
                                                        EDA + XGBoost Training
                                                         (SMOTE, class weights,
                                                          MLflow/DagsHub tracking)
                                                                     │
                                                                     ▼
                                          FastAPI Backend  ◄── YOLOv8 Perception
                                     (WebSocket / HTTP inference)
                                                                     │
                                                                     ▼
                                                        Streamlit Dashboard (HUD)
```

The project is built in two connected halves:

1. **Offline / training-time**: CARLA + SUMO generate a labeled dataset of frames, depth maps, LiDAR point clouds, and risk metadata; that dataset is used to train the XGBoost risk classifier.
2. **Online / live-inference-time**: the same YOLOv8 perception logic runs on live camera frames (with no simulator ground truth available), and the trained XGBoost model classifies risk in real time, streamed to a dashboard.

---

## Phase 1 — Simulation & Dataset Generation

**Goal:** build a realistic, co-simulated environment and use it to generate a large, labeled, multimodal dataset that reflects genuine road hazards — not just normal driving.

### 1.1 Building the road network

We started from a real-world map exported from **OpenStreetMap** (`trial2.osm`). Raw OSM data isn't usable by a traffic simulator on its own, so we converted it into a structured directed graph using SUMO's `netconvert` CLI, explicitly enabling pedestrian infrastructure guessing so that sidewalks and crossings would be generated automatically:

```bash
netconvert --osm-files trial2.osm \
  --output-file final_with_crossings.net.xml \
  --crossings.guess true --walkingareas true
```

**Why:** without `--crossings.guess`, SUMO has no concept of where pedestrians are legally allowed to cross, which would make pedestrian-hazard scenarios physically implausible later on.

### 1.2 Generating traffic demand

Using SUMO's `randomTrips.py`, we generated one hour (3600s) of randomized vehicle and pedestrian demand:

```bash
python randomTrips.py -n final_with_crossings.net.xml -o veh_routes.rou.xml --prefix v ...
python randomTrips.py -n final_with_crossings.net.xml -o ped_routes.rou.xml --prefix p --persontrips True ...
```

**Why:** random trip generation gives statistically realistic origin-destination traffic rather than a handful of scripted paths, so the ambient traffic behaves like a real city instead of a test track.

### 1.3 Bridging SUMO and CARLA

SUMO only models traffic *abstractly* — it has no 3D geometry, cameras, or physics. To let CARLA visualize and sense that same traffic, the network was exported to the **OpenDRIVE** (`.xodr`) standard:

```bash
netconvert --sumo-net-file final_with_crossings.net.xml \
  --opendrive-output final_with_crossings.xodr
```

**Why:** OpenDRIVE guarantees CARLA's coordinate space, lane geometry, and junction logic match SUMO's *exactly*, which is what makes true co-simulation (rather than two disconnected simulators) possible.

### 1.4 Running the co-simulation (`carla_sumo.py`)

The core generation script connects to a running CARLA server and a SUMO instance via TraCI, and on every simulation tick:

1. Spawns a Tesla Model 3 as the **ego vehicle**, driving on autopilot at a reduced speed to give sensors more time to capture interesting moments.
2. Attaches three sensors to the ego car — **RGB camera** (800×600, 90° FOV), **depth camera** (16-bit millimeter precision), and a **64-channel LiDAR** (100m range, 1M+ points/sec) — each streaming on its own thread so no frames are dropped.
3. Discards the first 80 steps as warm-up so traffic has time to build up and the ego car reaches normal speed before recording starts.
4. Mirrors SUMO's live vehicle/pedestrian lists into physical CARLA actors each step, positioned in a spatial grid around the ego car.
5. **Deliberately injects danger** at a configurable rate (default 35% of frames): cut-ins, sudden braking, head-on encounters, pedestrian crossings, and jaywalking — each built by finding a *geometrically plausible* candidate actor (e.g., the nearest vehicle ahead and in-lane for a brake scenario) and evolving its motion gradually rather than teleporting it, so the resulting behavior looks physically real.
6. Computes a **risk score (0–1)** for every actor using time-to-collision, in-lane proximity, an opposing-direction bonus for head-on cases, and a 1.5× pedestrian multiplier — then maps the highest score in the frame to one of four levels: **Safe / Medium / High / Critical**.
7. Saves the RGB image, depth map, LiDAR point cloud, and a full JSON metadata file (risk score/level, active scenario, every actor's position/speed) for every recorded frame, and groups frames into overlapping 60-frame sequences for time-series modeling.

**Why this design:** random simulation alone produces mostly "safe" driving, which would train a model to just predict the majority class. Scripted-but-natural-looking hazard injection guarantees enough high-risk examples to actually learn from, while the physically grounded scenario logic (nearest plausible vehicle, gradual motion) keeps the danger looking like real driving rather than an obvious scripted event.

### 1.5 Pushing the dataset to Kaggle

The generated multimodal dataset (frames, depth, LiDAR, and metadata) was uploaded to Kaggle so it could be pulled programmatically into the training notebook via the Kaggle CLI, keeping the pipeline reproducible and shareable without manual uploads.

📦 **Kaggle Dataset:** https://www.kaggle.com/datasets/mennaset/road-safety-simulation-data

---

## Phase 2 — Exploratory Data Analysis

Before modeling, the raw simulation logs were cleaned and explored:

- **Cleaning:** numeric risk scores were converted into human-readable labels (Safe/Medium/High/Critical), administrative/tracking columns were dropped, and rows missing critical ground-truth fields (safety distance, reaction time) were removed to guarantee a fully complete training set.
- **Target distribution:** Medium risk turned out to be the *most* common class, with Safe frames actually the rarest — meaning the simulation maintains a consistently challenging traffic environment rather than being mostly safe. This finding directly shaped the class-imbalance strategy in Phase 3.
- **Scenario analysis:** pedestrian-related scenarios (jaywalking, crossing) showed the highest average risk scores and the largest share of Critical frames, while normal driving stayed mostly Medium.
- **Risk relationships:** higher ego speed, shorter time-to-collision, smaller minimum distance, and more detected threats were all consistently linked to higher risk — confirmed by correlation analysis (risk score is strongly negatively correlated with TTC and minimum distance, and positively correlated with speed and threat count).
- **Feature engineering:** kinematic, spatial, and historical features were engineered on top of the raw logs — distance-to-speed ratios, threat density, kinetic-energy proxies, and rolling variance — to give the model richer collision-risk signals than raw distance/speed alone.

**Why:** the EDA findings (particularly the unusual class balance and the strength of TTC/distance as predictors) directly informed which features to engineer and how aggressively to correct for class imbalance in training.

---

## Phase 3 — Model Training

### 3.1 Perception (YOLOv8)

A `PerceptionPipeline` class wraps a YOLOv8 model and turns each raw camera frame into structured tabular data:

- **Monocular 3D estimation:** since a single RGB camera has no depth sensor, an object's horizontal position in the frame is mapped to lateral offset, and its vertical position is mapped to an estimated forward distance — giving every detection an approximate (x, y, z) location even without ground truth.
- **Ground-truth fusion (training only):** each detection is matched to the nearest known simulator actor within an 8-meter threshold; matched detections get the simulator's precise position/velocity/yaw instead of the rough vision estimate, and any actor YOLO completely missed is still logged (so the dataset reflects real perception gaps).
- **Nodes, edges, and frame features:** every frame produces per-object nodes, pairwise distance/closing-speed/TTC edges between every actor pair, and one frame-level summary row combining all of it — the exact tabular input the XGBoost classifier needs.

**Why fuse vision with ground truth at training time:** pure vision-only distance estimates are noisy. Fusing in ground truth when it's available produces far more accurate training labels, while the pipeline still gracefully falls back to vision-only estimates at live-inference time when no simulator metadata exists.

### 3.2 Handling class imbalance

The EDA confirmed that High and Critical risk frames were a minority. To prevent the model from simply learning to predict the dominant class:

1. **SMOTE (Synthetic Minority Over-sampling Technique)** generated synthetic minority-class samples by interpolating between neighboring real instances — not by duplicating existing rows.
2. **Balanced class weights** were computed *after* SMOTE to further emphasize minority classes during training.

**Why combine both:** SMOTE alone can still leave the loss function under-weighting rare classes; class weighting alone can overfit to a handful of duplicated points. Together, they gave the model materially better recall on rare-but-critical events without one technique's weakness dominating.

### 3.3 Training & model selection

An XGBoost multi-class classifier was trained to predict the four risk levels. Several tree counts (50, 100, 150, 200, 250 estimators) were swept to find the right complexity, using regularization, subsampling, and early stopping to control overfitting.

Model selection used a **custom score**: the average of **Macro Recall** (treats all classes equally) and **Critical Recall** (specifically measures how well the Critical class is detected — the category that matters most for actual safety). Overfitting was checked by comparing train vs. test Macro Recall.

**Why this metric, not accuracy:** in a safety system, missing a Critical event is far more costly than a false alarm. Optimizing for accuracy alone would reward a model for nailing the abundant "Medium" class while quietly missing rare Critical events — exactly the failure mode this project exists to prevent.

The final balanced XGBoost model reached **75% overall accuracy**, with meaningfully improved recall on the Safe, High, and Critical minority classes compared to an unbalanced baseline.

### 3.4 Experiment tracking

Every training run — hyperparameter tweak, tree-count variant, and evaluation metric — was logged with **MLflow**, hosted via **DagsHub**, and the final champion model was saved and versioned there.

**Why:** reproducibility. With dozens of tree-count/regularization combinations tested, tracking was the only way to reliably identify and reproduce the best-performing configuration later.

---

## Phase 4 — Backend & Live Inference (Docker)

### `model.py` — `LiveRiskPipeline`

The production wrapper used by the backend. On every live frame it:

1. Runs the same `PerceptionPipeline` used in training, but with no simulator ground truth (this is live inference), producing detections from YOLO alone.
2. Assembles the exact feature vector the trained XGBoost model expects, with sensible fallbacks wherever ground-truth values aren't available.
3. Loads the model via XGBoost's **native Booster API** rather than the scikit-learn wrapper — the sklearn wrapper doesn't restore the attributes needed for `predict_proba` after loading a saved model, so the native API avoids missing-attribute errors entirely.
4. **Temporal smoothing:** averages raw class probabilities over a rolling window (default 5 frames) so a single noisy frame can't suddenly flip the displayed risk level.
5. **Hysteresis logic:** even after smoothing, the displayed label only changes once a new class has led for several consecutive frames — *except* escalations to a more dangerous level, which are allowed through immediately, since a safety warning should never be delayed. De-escalations require the full confirmation streak.
6. Returns both a `risk_level` (always matches the raw smoothed probabilities — good for analytics) and a `stable_risk_level` (the flicker-resistant, driver-facing HUD label), plus detections for the client to draw bounding boxes.

**Why smoothing *and* hysteresis:** smoothing alone still allows label flapping near a decision boundary; hysteresis alone can't fix single noisy spikes. Combined, they produce a HUD label that's both fast to escalate and stable once displayed — the two properties an actual driver-facing warning system needs simultaneously.

### `main.py` — FastAPI Gateway

Exposes two endpoints, each backed by its own `LiveRiskPipeline` instance to avoid state collisions between sessions:

- `POST /predict/frame` — single-image HTTP inference; resets pipeline history before every call so each request is fully independent.
- `WS /live/carla` — continuous WebSocket stream; keeps rolling state across frames so temporal smoothing and hysteresis actually work, resetting only when a connection starts or ends.

Images are accepted as base64-encoded JPEG (preferred) or raw uint8 arrays, decoded with OpenCV, and run through the risk pipeline; predictions are returned as JSON.

### Docker Images

Two separate images keep the backend and frontend independently scalable:

- **`Dockerfile`** — lightweight Python 3.10 image running the FastAPI service with Uvicorn on port `8000`, installing only the OpenCV system libraries needed for headless operation.
- **`Dockerfile.streamlit`** — a matching Python 3.10 image running the Streamlit dashboard on port `8501`.

**`docker-compose.yml`** orchestrates both: the dashboard container depends on the API being ready, and both restart automatically unless explicitly stopped — a clean separation between "compute the risk" and "visualize the risk."

---

## Phase 5 — Streamlit Dashboard

A multi-page Streamlit dashboard renders the live risk feed with a cyberpunk/HUD-style visual theme, covering object-detection analytics, driver behavior, accident-prevention simulation, scenario testing, historical analytics, AI performance metrics, and reporting.

🖥️ **Live Dashboard:** https://streamlit-dashboard.bluebay-ca82a8fb.eastus.azurecontainerapps.io/

---

## Phase 6 — Cloud Deployment (Azure)

The Dockerized backend + dashboard stack was deployed to **Azure** so the system is reachable outside the local environment — the same `docker-compose` configuration used locally is what runs in the cloud, keeping local and production environments identical.

---

## Project Structure

```
AI-Proactive-Road-Safety-System/
├── SUMO/                                  # Traffic network & route definitions
│   ├── trial2.osm                         # Raw OpenStreetMap export
│   ├── final_with_crossings.net.xml       # SUMO network (with pedestrian crossings)
│   ├── final_with_crossings.xodr          # OpenDRIVE export consumed by CARLA
│   ├── final.net.xml / final.sumocfg      # Simulation configuration
│   ├── veh_routes.rou.xml                 # Generated vehicle demand
│   ├── ped_routes.rou.xml                 # Generated pedestrian demand
│   └── routes.rou.xml / all_routes.rou.xml
│
├── carla-env/                             # Python virtual environment for the CARLA client
│
├── dataset/                                # Generated multimodal dataset (see Kaggle link)
│
├── road_safety_azure_deploy_package/
│   └── deploy_package/
│       ├── app/
│       │   ├── __init__.py
│       │   ├── main.py                    # FastAPI gateway (HTTP + WebSocket)
│       │   └── model.py                   # PerceptionPipeline + LiveRiskPipeline
│       ├── Dockerfile                      # Backend API container
│       ├── Dockerfile.streamlit            # Dashboard container
│       ├── docker-compose.yml              # Orchestrates both services
│       ├── requirements.txt
│       ├── best_xgboost_model.json         # Trained XGBoost model
│       ├── yolov8n.pt                      # YOLOv8 detection weights
│       └── dashboard.py                    # Streamlit dashboard entry point
│
├── carla_sumo.py                          # CARLA/SUMO co-simulation dataset generator
├── Final_of_Road_safety_system.ipynb      # EDA + model training notebook
├── Project Documentation (1).pdf          # Full technical write-up
└── README.md
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Simulation | CARLA, SUMO, OpenStreetMap, netconvert, randomTrips.py |
| Perception | YOLOv8 (Ultralytics), OpenCV |
| Modeling | XGBoost, scikit-learn, SMOTE, pandas, NumPy |
| Experiment Tracking | MLflow, DagsHub |
| Backend | FastAPI, Uvicorn, WebSockets |
| Frontend | Streamlit |
| Deployment | Docker, Docker Compose, Azure |
| Dataset Hosting | Kaggle |

---

## Dataset

The dataset was generated entirely through the CARLA/SUMO co-simulation pipeline described in [Phase 1](#phase-1--simulation--dataset-generation) — synchronized RGB, depth, and LiDAR captures, each paired with a full risk-metadata JSON (risk score, risk level, scenario label, and every actor's position/speed). It was uploaded to Kaggle for reproducible, shareable access.

📦 **Kaggle Dataset:** https://www.kaggle.com/datasets/mennaset/road-safety-simulation-data

---

## Live Demo

🖥️ **Streamlit Dashboard:** https://streamlit-dashboard.bluebay-ca82a8fb.eastus.azurecontainerapps.io/

---

## Screenshots


### Streamlit Dashboard
<img width="1264" height="638" alt="image" src="https://github.com/user-attachments/assets/e79613ea-e7ac-46de-a22d-c536e79117ad" />

<img width="873" height="762" alt="Screenshot 2026-07-11 103911" src="https://github.com/user-attachments/assets/eec86ee8-9032-4ac6-a69e-ced89a2b4720" />


---

## Getting Started

### Prerequisites

- CARLA 0.9.15 and SUMO installed locally (for regenerating the dataset)
- Docker & Docker Compose (for running the inference API + dashboard)
- Python 3.10

### Run the backend + dashboard

```bash
git clone https://github.com/Basant-Tarik01/AI-Proactive-Road-Safety-System.git
cd AI-Proactive-Road-Safety-System/road_safety_azure_deploy_package/deploy_package
docker-compose up --build
```

- API available at `http://localhost:8000`
- Dashboard available at `http://localhost:8501`

### Regenerate the dataset

```bash
python carla_sumo.py --sumo-cfg SUMO/final.sumocfg --steps 600 --danger-ratio 0.35 --enable-pedestrians
```

---

## Results

- **75% overall accuracy** on the 4-class risk classifier (Safe / Medium / High / Critical)
- Meaningfully improved recall on minority Safe, High, and Critical classes after SMOTE + class weighting
- Stable, flicker-resistant live HUD output via temporal smoothing + hysteresis
- Fully reproducible training pipeline, tracked end-to-end in MLflow/DagsHub

---

## Future Work

- Expand the Critical-class dataset further to push Critical Recall higher
- Explore temporal/sequence models (LSTM/Transformer) on the 60-frame sequence index
- Add multi-camera coverage for blind-spot detection
- Extend cloud deployment with autoscaling for concurrent live sessions
