# M2 Report: Process and Tooling

## 1. Experiment Tracking
### Purpose
The purpose of experiment tracking is to monitor and compare multiple machine learning runs. It logs parameters, metrics, and artifacts for improved model development and debugging.

### Implementation
1. Created a `train.py` script that logs parameters, metrics (accuracy), and artifacts (models) using MLflow.
2. Ran multiple experiments with different hyperparameters (`n_estimators`, `max_depth`).
3. Visualized experiment results using the MLflow UI.

### Results
- Logged parameters, metrics, and models in MLflow.
- Screenshots of MLflow UI showing experiment runs and detailed logs.

---

## 2. Data Versioning
### Purpose
Data versioning ensures reproducibility by tracking changes to datasets over time and enabling rollback to previous versions.

### Implementation
1. Added the dataset (`data/iris.csv`) to DVC for version control.
2. Modified the dataset and created new versions.
3. Used `dvc checkout` to revert to a previous dataset version.

### Results
- Successfully tracked dataset versions using DVC.
- Screenshots of `dvc add`, `dvc status`, and `dvc checkout` commands.

---

## Challenges and Resolutions
### Experiment Tracking
- **Challenge**: Managing MLflow tracking server for visualization.
- **Resolution**: Used the default local tracking server provided by MLflow.

### Data Versioning
- **Challenge**: Handling large dataset versions.
- **Resolution**: Optimized storage by using DVC’s cache system.

---

## Conclusion
Experiment tracking and data versioning were implemented using MLflow and DVC. These tools ensure reproducibility and provide better insights into model performance and dataset changes.
