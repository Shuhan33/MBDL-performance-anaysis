$ErrorActionPreference = "Stop"

Write-Host "Installing/updating Python dependencies..."
python -m pip install -r requirements.txt

Write-Host "Running the full comparison..."
python train_experiment.py --device auto --epochs 30

Write-Host "Finished. Open results/report.html in a browser."
