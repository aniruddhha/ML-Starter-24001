import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

import os
import subprocess

import mlflow
import optuna

# Load Data
df = pd.read_csv('data.csv')
X = df[['X']]
y = df['y']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set MLflow tracking url
mlflow.set_tracking_uri('file:./mlruns')

# Create MLflow Expriment
exp_nm = 'Simple Linear Regression - Optuna'
mlflow.set_experiment(exp_nm)

def objective(trial):
    with mlflow.start_run() as run:

        # Train Model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make Predictions
        predictions = model.predict(X_test)

        # Evaluate model
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # Log Parameters
        mlflow.log_param('mse', mse)
        mlflow.log_param('r2', r2)

        # Log Model
        mlflow.sklearn.log_model(model, 'model')

        # Log a plot
        plt.figure(figsize=(8, 6))
        plt.scatter(X_test, y_test, label="Actual")
        plt.plot(X_test, predictions, color='red', label="Predicted")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.legend()
        plot_path = "plot.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close() # Close the plot to prevent display issues

        # print(f"Run ID: {run.info.run_uuid}")
        # print(f"Metrics: MSE={mse}, R2={r2}")

        return mse
    
#Start MLflow UI
# os.system("mlflow ui --port 8989")
def start_mlflow_ui_bg(port = 5000):
    command = ["mlflow", "ui", "--port", str(port)]
    try:
        process = subprocess.Popen(command)
        print(f"MLflow UI started in background (PID: {process.pid}) on port {port}")
    except FileNotFoundError:
        print("Error: mlflow command not found. Make sure MLflow is installed and in your PATH.")
    except Exception as e:
        print(f"Error starting MLflow UI: {e}")

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)

    print('Best Trial')
    trial = study.best_trial

    print(f'Value {trial.value}')
    print(f'Params')
    for key, value in trial.params.items():
        print(f' {key} -> {value}')
    
    start_mlflow_ui_bg(8989)

