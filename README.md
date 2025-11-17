# OrganicHAR: Towards Activity Discovery in Organic Settings for Privacy Preserving Sensors Using Efficient Video Analysis

[[paper (IMWUT 2025)](https://doi.org/10.1145/3770674)]


**Abstract:**
Deploying human activity recognition (HAR) at home is still rare because sensor signals vary wildly across houses, people,
and time, essentially requiring in-situ data collection and training. Prior approaches use cameras to generate training labels
for privacy-preserving sensors (LiDAR, RADAR, Thermal), but this forces sensors to detect predefined activities that cameras
can see yet the sensors themselves cannot reliably distinguish. In this work, we introduce OrganicHAR, an activity discovery
framework that inverts this relationship by placing sensor capabilities at the center of activity discovery. Our approach
identifies naturally occurring signal patterns using privacy-preserving sensors, leverages Vision Language Models (VLMs) only
during these key moments for scene understanding, and discovers discrete activity labels at granularities that these sensors
can reliably detect. Our evaluation with 12 participants demonstrates OrganicHAR's effectiveness: it achieves 79% accuracy
for coarse (4-5) activities using only basic ambient sensors (radar, lidar, thermal arrays), and 73% accuracy for fine-grained
(8-9) activities when a wearable IMU, depth, and pose sensor are added. OrganicHAR maintains 77% accuracy on average
across configurations while discovering 4-8 categories per user (15 across all users) tailored to each environment and sensor
capabilities. By triggering video processing only at key moments identified by local sensors, we reduce queries to VLM by
90%, enabling practical and privacy-preserving activity recognition in natural settings.

---

## Overview

OrganicHAR is a sensor-first activity discovery framework that enables privacy-preserving human activity recognition in home environments. The system:

1. **Identifies key moments** using privacy-preserving sensors (Doppler radar, LiDAR, thermal arrays, IMU)
2. **Discovers activities** by processing only ~10% of video data at key moments using VLMs
3. **Trains activity models** that operate solely on privacy-preserving sensors during deployment
4. **Adapts to individual environments** by discovering 4-8 activity categories per user tailored to sensor capabilities

---

## Repository Structure

The codebase is organized into five sequential stages (s1-s5) that correspond to the OrganicHAR pipeline:

```
OrganicHAR-Public/
├── s1.create_session_data/    # Data preprocessing and session segmentation
├── s2.create_clusters/         # Key moment identification via clustering
├── s3.generate_labels/         # Activity label discovery using VLMs/LLMs
├── s4.train_activity_models/   # HAR model training
├── s5.evaluation/              # Performance evaluation
├── src/                        # Core utilities and modules
│   ├── featurization/          # Sensor-specific feature extraction
│   ├── key_moment_detection/   # Clustering and change detection
│   ├── label_discovery/        # VLM/LLM-based label generation
│   └── evaluation/             # Evaluation metrics
└── complete_paper/             # LaTeX source for the IMWUT paper
```

---

## Setup

### Prerequisites

- Python 3.8+
- Raw sensor data organized in the expected directory structure
- API keys for VLM services (e.g., OpenAI GPT-4, Google Vertex AI)

### Environment Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` to configure paths for your setup:
   ```bash
   # Base Directories
   BASE_RAW_DIR=./raw_data              # Your raw sensor data directory
   BASE_PROCESSED_DIR=./processed_data  # Processed data output directory
   BASE_RESULTS_DIR=./results           # Results and models directory

   # Model Cache Directories (download models as needed)
   POSE_CACHE_DIR=./models/pose
   DEPTH_MODEL_PATH=./models/monodepth/DepthAnythingV2SmallF16.mlpackage
   DINOV2_CACHE_DIR=./models/dinov2
   AST_COREML_CACHE_DIR=./models/audio
   ```

3. Load environment variables:
   ```bash
   source .env  # or use python-dotenv in scripts
   ```

---

## Pipeline Stages

### Stage 1: Session Data Creation (`s1.create_session_data/`)

**Purpose**: Process raw sensor streams and segment them into analysis sessions.

**Scripts**:
- `1a_get_sessions_from_android.py` - Extract sessions from Android sensor data
- `1b_get_sessions_from_iwatch.py` - Extract sessions from Apple Watch data
- `2a_process_birdseye.py` - Process overhead camera/video data
- `2b_process_depth.py` - Process depth sensor data
- `2c_process_thermalcam.py` - Process thermal camera data
- `3c_extract_window_data.py` - Extract 5-second windows with 0.5s stride (ambient sensors)
- `3d_extract_window_watch_data.py` - Extract windows for wearable sensors

**Paper Reference**: Section 5 (Data Collection)

**Usage**:
```bash
cd s1.create_session_data
python 1a_get_sessions_from_android.py --participant <name>
python 3c_extract_window_data.py --participant <name> --window_size 5.0 --sliding_window_length 0.5
```

---

### Stage 2: Cluster Creation (`s2.create_clusters/`)

**Purpose**: Identify key moments through spatial clustering and temporal change detection.

**Scripts**:
- `4a_preprocess_x_data.py` - Featurize sensor data (Doppler, LiDAR, thermal, depth, pose)
- `4b_preprocess_watch_data.py` - Featurize IMU data from wearables
- `5a_cluster_data.py` - Perform HDBSCAN clustering to identify recurring patterns
- `5b_change_detection.py` - Detect temporal changes using GMM-based anomaly detection
- `5c_compile_interesting_segments.py` - Consolidate key moments from both approaches

**Paper Reference**: Section 4.2 (Key Moments Identification)

**Key Concepts**:
- **Spatial Clustering**: HDBSCAN on sensor feature space to find recurring activity patterns
- **Change Detection**: GMM-based anomaly detection to identify activity transitions
- Only ~10% of total data is selected as key moments for VLM processing

**Usage**:
```bash
cd s2.create_clusters
python 4a_preprocess_x_data.py --participant <name>
python 5a_cluster_data.py --participant <name> --alpha 0.5
python 5b_change_detection.py --participant <name>
```

---

### Stage 3: Label Generation (`s3.generate_labels/`)

**Purpose**: Generate activity labels by processing key moments with VLMs and clustering semantically.

**Scripts**:
- `6a_generate_descriptions.py` - Generate scene descriptions using VLMs (GPT-4V)
- `6b_generate_location_labels.py` - Identify working zones (locations) in the environment
- `6c_generate_location_action_labels.py` - Extract location-action pairs from descriptions
- `7a_generate_label_sim_matrix.py` - Compute semantic similarity and cluster labels

**Paper Reference**: Section 4.3 (Label Discovery), Section 4.4 (Semantic Granularity Control)

**Key Concepts**:
- VLM processes only key moments (~10% of video data)
- LLM clusters similar descriptions into discrete activity labels
- Relaxation parameter λ controls granularity (λ=0.4: coarse, λ=0.2: fine-grained)

**Usage**:
```bash
cd s3.generate_labels
python 6a_generate_descriptions.py --participant <name> --vlm_model gpt-4-vision
python 7a_generate_label_sim_matrix.py --participant <name> --alpha 0.5 --merging_threshold 0.3
```

---

### Stage 4: Activity Model Training (`s4.train_activity_models/`)

**Purpose**: Train HAR models on discovered labels using privacy-preserving sensor features.

**Scripts**:
- `8a_train_activity_models.py` - Train per-sensor and ensemble models
- `8d_run_predictions.py` - Generate predictions on test data
- `9a_prepare_gt.py` / `9a_prepare_av_gt.py` - Prepare ground truth annotations

**Paper Reference**: Section 4.5 (Activity Recognition Model Training)

**Model Types**:
- Per-sensor models (Doppler, LiDAR, thermal, depth, pose, IMU)
- Late-fusion ensemble combining all sensors
- XGBoost classifiers with leave-one-session-out training

**Usage**:
```bash
cd s4.train_activity_models
python 8a_train_activity_models.py --participant <name> --alpha 0.5 --direction forward
python 8d_run_predictions.py --participant <name> --prob_threshold 0.0
```

---

### Stage 5: Evaluation (`s5.evaluation/`)

**Purpose**: Evaluate activity recognition performance against ground truth annotations.

**Scripts**:
- `9b_organize_annotations.py` - Organize ground truth annotations
- `9c_map_gt_labels_to_av.py` - Map ground truth to auto-discovered labels
- `9d_evaluate_predictions.py` - Compute accuracy, F1, precision, recall metrics

**Paper Reference**: Section 6 (Evaluation)

**Key Results** (from paper):
- **Basic Ambient**: 79% accuracy (4-5 activities)
- **Ambient + IMU**: 72% accuracy (6-7 activities)
- **Advanced Ambient + IMU**: 73% accuracy (8-9 activities)
- Average across configurations: 77% accuracy

**Usage**:
```bash
cd s5.evaluation
python 9b_organize_annotations.py --participant <name>
python 9c_map_gt_labels_to_av.py --participant <name> --alpha 0.5
python 9d_evaluate_predictions.py --participant <name> --alpha 0.5 --prob_threshold 0.0
```

---

## Replicating Paper Results

### Full Pipeline Execution

To replicate results for a single participant:

```bash
# Stage 1: Process raw data
cd s1.create_session_data
python 3c_extract_window_data.py --participant <name> --window_size 5.0 --sliding_window_length 0.5

# Stage 2: Identify key moments
cd ../s2.create_clusters
python 4a_preprocess_x_data.py --participant <name>
python 5a_cluster_data.py --participant <name> --alpha 0.5
python 5b_change_detection.py --participant <name>
python 5c_compile_interesting_segments.py --participant <name>

# Stage 3: Discover labels
cd ../s3.generate_labels
python 6a_generate_descriptions.py --participant <name> --vlm_model gpt-4-vision
python 7a_generate_label_sim_matrix.py --participant <name> --alpha 0.5 --merging_threshold 0.3

# Stage 4: Train models
cd ../s4.train_activity_models
python 8a_train_activity_models.py --participant <name> --alpha 0.5
python 8d_run_predictions.py --participant <name>

# Stage 5: Evaluate
cd ../s5.evaluation
python 9b_organize_annotations.py --participant <name>
python 9c_map_gt_labels_to_av.py --participant <name> --alpha 0.5
python 9d_evaluate_predictions.py --participant <name> --alpha 0.5
```

### Key Parameters

- `--alpha`: Clustering parameter (default: 0.5) - controls HDBSCAN sensitivity
- `--merging_threshold` (λ): Semantic similarity threshold for label merging
  - λ=0.4 → Conservative (4-5 coarse activities)
  - λ=0.3 → Balanced (6-7 medium activities)
  - λ=0.2 → Relaxed (8-9 fine-grained activities)
- `--window_size`: Feature window size in seconds (default: 5.0)
- `--sliding_window_length`: Window stride in seconds (default: 0.5)

---

## Code-to-Paper Mapping

| Paper Section | Code Location |
|---------------|---------------|
| §3 Hardware Design | Data collection infrastructure (not in repo) |
| §4.1 Featurization | `src/featurization/` |
| §4.2 Key Moments (Spatial Clustering) | `s2.create_clusters/5a_cluster_data.py` |
| §4.2 Key Moments (Change Detection) | `s2.create_clusters/5b_change_detection.py` |
| §4.3 Semantic Description Generation | `s3.generate_labels/6a_generate_descriptions.py` |
| §4.3 Location & Action Extraction | `s3.generate_labels/6b_generate_location_labels.py`, `6c_*` |
| §4.3 Activity Label Clustering | `s3.generate_labels/7a_generate_label_sim_matrix.py` |
| §4.4 Semantic Granularity Control | `src/label_discovery/SemanticActivityClustering.py` |
| §4.5 HAR Model Training | `s4.train_activity_models/8a_train_activity_models.py` |
| §6 Evaluation | `s5.evaluation/` |

---

## Notes

- Some scripts may require VLM API keys (OpenAI, Vertex AI) configured in your environment
- The pipeline assumes raw data follows the expected directory structure from data collection
- Intermediate results are cached to avoid reprocessing
- Leave-one-out analysis scripts are in `leaveoneoutanalysis/` subdirectories

---

## Citation

```bibtex
@INPROCEEDINGS{patidar25organichar,
    title = {OrganicHAR: Towards Activity Discovery in Organic Settings for Privacy Preserving Sensors Using Efficient Video Analysis},
    author = {Prasoon Patidar, Riku Arakawa, Ricardo Graça, Rúben Moutinho, Adriano Soares, Ana Vasconcelos, Filippo Talami, Joana
Couto da Silva, Inês Silva, Cristina Mendes Santos, Mayank Goel, and Yuvraj Agarwal},
    journal = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies}
    year = {2025},
    publisher = {ACM},
    address = {Shanghai, China},
    article = {203},
    volume = {9},
    number = {4},
    month = {12},
    doi = {https://doi.org/10.1145/3770674},
    numpages = {32},
    keywords = {ubiquitous sensing, privacy first design, human activity recognition},
}
```

---

## License

See [LICENSE](LICENSE) for details.
