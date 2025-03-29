import argparse
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from imblearn.under_sampling import RandomUnderSampler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--validation", type=str, default="/opt/ml/input/data/validation")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    args = parser.parse_args()

    train_df = pd.read_csv(f"{args.train}/train.csv")
    val_df = pd.read_csv(f"{args.validation}/validation.csv")

    X_train = train_df.drop("Flight_Status_Binary", axis=1)
    y_train = train_df["Flight_Status_Binary"]

    X_val = val_df.drop("Flight_Status_Binary", axis=1)
    y_val = val_df["Flight_Status_Binary"]

    rus = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'class_weight': [None, 'balanced']
    }

    rf_tuned = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_grid,
        scoring='recall',
        cv=5,
        n_iter=5,
        random_state=42,
        verbose=1
    )

    rf_tuned.fit(X_train_resampled, y_train_resampled)
    best_model = rf_tuned.best_estimator_
    joblib.dump(best_model, f"{args.model_dir}/model.joblib")
