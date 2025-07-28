import argparse
import subprocess


#-----------------------------
#  Function to run TabNet training
#-----------------------------
def run_training(dataset_path: str, output_dir: str):
    n_splits = 10
    n_trials = 20
    print(f"\n[+] Starting TabNet {n_splits}-Fold CV Training...")
    train_cmd = [
        "python", "src/models/tabnet/tabnet_train.py",
        "--dataset_dir", dataset_path,
        "--output_dir", output_dir,
        "--n_splits", str(n_splits),
        "--n_trials", str(n_trials)
    ]
    subprocess.run(train_cmd, check=True)
    print(f"[+] Training completed. Fold models saved to {output_dir}")

#-----------------------------
#  Function to run TabNet ensembling and prediction
#-----------------------------  
def run_prediction(dataset_path: str, model_dir: str):
    output_file = "models/exam_output/y_pred.csv"  # fixed prediction output path
    print(f"\n[+] Starting TabNet Ensemble Prediction on Test Set...")
    predict_cmd = [
        "python", "src/models/tabnet/tabnet_ensembling.py",
        "--dataset_dir", dataset_path,
        "--model_dir", model_dir,
        "--output_file", output_file
    ]
    subprocess.run(predict_cmd, check=True)
    print(f"[+] Prediction completed. Predictions saved to {output_file}")

#-----------------------------
#  CLI
#-----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline: Train TabNet with KFold + Ensemble Predict")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--output_dir", type=str, default="models/exam_output", help="Directory to save fold models")

    args = parser.parse_args()

    run_training(args.dataset_path, args.output_dir)
    run_prediction(args.dataset_path, args.output_dir)


#usage: python src/models/tabnet/tabnet_pipeline.py --dataset_path examdata/exam_dataset/1 --output_dir models/exam_output