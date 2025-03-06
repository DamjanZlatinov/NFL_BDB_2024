import os
import shutil
import pickle
import json
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from datetime import datetime

id_cols = ['gameId',	'playId',	'nflId_tackler']
target_variables = ['tackle|max', 'assist|max', 'pff_missedTackle|max', 'tackle_success']
target_variable = 'tackle_success'

def train_model(df):
    print("Training model...")
   
    # Define target variable and features
    X = df.drop(columns=id_cols + target_variables)
    y = df[target_variable]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
)

    # Define parameter grid for Grid Search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'scale_pos_weight': [y_train.value_counts()[0] / y_train.value_counts()[1]],
        'random_state': [42]
    }

    # Initialize model
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')

    # Perform Grid Search
    print("Performing Grid Search...")
    grid_search = GridSearchCV(xgb_model, param_grid, scoring='f1', cv=3, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")

    # Train best model on full training set
    best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Evaluate model
    y_pred = best_model.predict(X_val)
    f1 = f1_score(y_val, y_pred)
    print(f"Validation F1-score: {f1:.3f}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['No Tackle', 'Tackle'], yticklabels=['No Tackle', 'Tackle'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_val, y_pred))

    # ============================
    # SAVE MODEL & METADATA
    # ============================

    # Define save path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"../models/saved_model_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Save model as .xgb
    model_path = os.path.join(output_dir, "best_xgb_model.xgb")
    best_model.save_model(model_path)


    # Save metadata
    metadata = {
        "best_params": grid_search.best_params_,
        "validation_f1_score": f1,
        "timestamp": timestamp
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    # Copy all Python scripts to output directory, so we know what code was used to train the model. 
    current_dir = os.getcwd()
    for file in os.listdir(os.path.join(current_dir, 'scripts')):
        if file.endswith(".py"):
            shutil.copy(os.path.join(os.path.join(current_dir, 'scripts'), file), os.path.join(output_dir, 'scripts'))

    print(f"Model and scripts saved in: {output_dir}")
    

if __name__ == "__main__":
    print('Vrmmm')