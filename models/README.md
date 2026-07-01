# Model Checkpoints

No trained production PINN checkpoint is currently available.

The architecture and training loop are implemented in
`stellar_analyzer/ml/pinn_model.py`. A checkpoint must only be added here after
training on a sufficiently diverse stellar grid and recording train,
validation, and test metrics in `paper_report_data/07_machine_learning/`.
