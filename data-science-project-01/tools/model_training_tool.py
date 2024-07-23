from crewai import Agent, Task, Crew
from crewai_tools import BaseTool
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

class TrainingModelTool(BaseTool):
    name: str = "Random Forest Model Trainer"
    description: str = "Trains a Random Forest model for house price prediction and saves it as a pickle file"

    def _run(self, file_path, target_variable):
        # Load and prepare data
        data = pd.read_csv(file_path)
        X = data.drop(target_variable, axis=1)
        y = data[target_variable]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Random Forest model
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Create 'saved_model' folder if it doesn't exist
        os.makedirs('saved_model', exist_ok=True)

        # Create 'train_test_data' folder if it doesn't exist
        os.makedirs('train_test_data', exist_ok=True)

        # Save the model as a pickle file in the 'saved_model' folder
        model_filename = os.path.join('saved_model', 'random_forest_model.pkl')
        with open(model_filename, 'wb') as file:
            pickle.dump(model, file)

        # Save the scaler as a pickle file in the 'saved_model' folder
        scaler_filename = os.path.join('saved_model', 'scaler.pkl')
        with open(scaler_filename, 'wb') as file:
            pickle.dump(scaler, file)

        # Save train and test data in the 'train_test_data' folder
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        train_filename = os.path.join('train_test_data', 'train_data.csv')
        test_filename = os.path.join('train_test_data', 'test_data.csv')
        
        train_data.to_csv(train_filename, index=False)
        test_data.to_csv(test_filename, index=False)

        report = f"Random Forest Model Training Report:\n\n"
        report += f"Root Mean Squared Error: ${rmse:.2f}\n"
        report += f"R-squared Score: {r2:.4f}\n\n"
        report += "Top 5 Important Features:\n"
        for _, row in feature_importance.head().iterrows():
            report += f"- {row['feature']}: {row['importance']:.4f}\n"
        report += f"\nModel saved as: {os.path.abspath(model_filename)}\n"
        report += f"Scaler saved as: {os.path.abspath(scaler_filename)}\n"
        report += f"Train data saved as: {os.path.abspath(train_filename)}\n"
        report += f"Test data saved as: {os.path.abspath(test_filename)}\n"

        return report

