from crewai_tools import BaseTool
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import os

class ModelEvaluationTool(BaseTool):
    name: str = "Model Evaluator"
    description: str = "Evaluates the saved Random Forest model using the test data and generates an evaluation report"

    def _run(self, target_variable):
        # Load the saved model
        model_path = os.path.join('saved_model', 'random_forest_model.pkl')
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        # Load the saved scaler
        scaler_path = os.path.join('saved_model', 'scaler.pkl')
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)

        # Load the test data
        test_data_path = os.path.join('train_test_data', 'test_data.csv')
        test_data = pd.read_csv(test_data_path)

        X_test = test_data.drop(target_variable, axis=1)
        y_test = test_data[target_variable]

        # Scale the test features
        X_test_scaled = scaler.transform(X_test)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Generate the evaluation report
        report = "Model Evaluation Report:\n\n"
        report += f"Root Mean Squared Error (RMSE): ${rmse:.2f}\n"
        report += f"Mean Absolute Error (MAE): ${mae:.2f}\n"
        report += f"R-squared Score: {r2:.4f}\n\n"

        # Calculate and add residual statistics
        residuals = y_test - y_pred
        report += "Residual Statistics:\n"
        report += f"Mean Residual: ${residuals.mean():.2f}\n"
        report += f"Std Deviation of Residuals: ${residuals.std():.2f}\n"
        report += f"Min Residual: ${residuals.min():.2f}\n"
        report += f"Max Residual: ${residuals.max():.2f}\n\n"

        # Add feature importance
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        report += "Top 5 Important Features:\n"
        for _, row in feature_importance.head().iterrows():
            report += f"- {row['feature']}: {row['importance']:.4f}\n"

        return report



