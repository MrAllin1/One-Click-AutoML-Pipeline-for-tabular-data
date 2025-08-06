We trained and ran everything on the university servers (SLURM + CUDA 11.7).
To reproduce, do the following from the repo root:

1) Train the model
With one click this command takes the exam dataset, goes through the pipeline and saves the model under modelsFinal-exam-reproduced/
To do this run:
sbatch /src/pipeline/final_pipeline.sh

Model output (after job finishes - which took us a 7 hours and 2 minutes) can be found here:
/modelsFinal-exam-reproduced/final_model.pkl

2) Generate predictions
This command uses the saved model and writes a CSV of predictions for the exam dataset test:
sbatch /src/pipeline/prediction_pipeline.sh

Predictions outputs can be found here:
/modelsFinal-exam-reproduced/y_pred.csv

We export predictions as CSV for easier readability.
We used this command to preduce predictions.npy:

python - <<'PY'
import os, numpy as np, pandas as pd
md = "modelsFinal-exam-reproduced"
y = pd.read_csv(os.path.join(md, "y_pred.csv"), header=None).values.squeeze()
np.save(os.path.join(md, "predictions.npy"), y)
print("Saved:", os.path.join(md, "predictions.npy"))
PY

