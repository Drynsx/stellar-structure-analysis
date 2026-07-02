# Model Checkpoints

No trained production PINN checkpoint is committed.

Prepare data and train with:

```powershell
.venv\Scripts\python.exe -m stellar_analyzer prepare-pinn
.venv\Scripts\python.exe -m stellar_analyzer train-pinn --config configs\pinn_training.json
```

Checkpoints include the model state, architecture, training configuration,
loss history, split sizes, validation loss, and held-out test loss. The bundled
single-track dataset validates the pipeline only; production training requires
a broad grid of independent stellar tracks.
