import argparse
import os
import pandas as pd
import xgboost as xgb
import joblib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--validation", type=str, default="/opt/ml/input/data/validation")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    args = parser.parse_args()

    # Load data
    train_df = pd.read_csv(f"{args.train}/train.csv")
    val_df = pd.read_csv(f"{args.validation}/validation.csv")

    X_train = train_df.drop("Flight_Status_Binary", axis=1)
    y_train = train_df["Flight_Status_Binary"]

    X_val = val_df.drop("Flight_Status_Binary", axis=1)
    y_val = val_df["Flight_Status_Binary"]

    # Train model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))