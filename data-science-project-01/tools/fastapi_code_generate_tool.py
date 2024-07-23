
from crewai_tools import BaseTool
import pickle
import os
import pandas as pd

class ModelAnalyzerTool(BaseTool):
    name: str = "Model Analyzer"
    description: str = "Analyzes the saved pickle model and generates FastAPI code"

    def _run(self):
        model_path = os.path.join('saved_model', 'random_forest_model.pkl')
        scaler_path = os.path.join('saved_model', 'scaler.pkl')
        test_data_path = os.path.join('train_test_data', 'test_data.csv')

        # Load the model and scaler
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        # Read the test data to get feature names
        test_data_path = os.path.join('train_test_data', 'test_data.csv')
        test_data = pd.read_csv(test_data_path)
        feature_names = test_data.iloc[:, :-1].columns.tolist() # remove the last column

        # Generate FastAPI code
        main_py_content = self._generate_fastapi_code(model_path, scaler_path, feature_names)

        # Save the main.py file
        with open('main.py', 'w') as f:
            f.write(main_py_content)

        return "main.py file has been generated successfully based on the model analysis."

    def _generate_fastapi_code(self, model_path, scaler_path, feature_names):
        # Generate the FastAPI code based on the analyzed model
        code = f"""
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        import pickle
        import numpy as np

        # Load the saved Random Forest model
        with open("{model_path}", "rb") as model_file:
            model = pickle.load(model_file)

        # Initialize FastAPI app
        app = FastAPI()

        # Define the request body for prediction
        class PredictionRequest(BaseModel):
            {self._generate_pydantic_fields(feature_names)}

        # Define the response model
        class PredictionResponse(BaseModel):
            prediction: float

        @app.post("/predict", response_model=PredictionResponse)
        def predict(request: PredictionRequest):
            try:
                # Extract features from the request
                features = np.array([[getattr(request, feature) for feature in {feature_names}]])
                
                # Load the model and scaler
                load_model_and_scaler()
                
                # Scale the input data
                scaled_data = scaler.transform(features)
                
                # Make a prediction
                prediction = model.predict(scaled_data)
                
                # Return the prediction
                return PredictionResponse(prediction=float(prediction[0]))
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/")
        def read_root():
            return {{"message": "Welcome to the Random Forest Model API"}}
        """
        return code

    def _generate_pydantic_fields(self, feature_names):
        return "\n    ".join([f"{feature}: float" for feature in feature_names])