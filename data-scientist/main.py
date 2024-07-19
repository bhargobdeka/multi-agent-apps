
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="House Price Prediction", 
    description="API to predict house prices", version="1.0")

# Load the trained model and scaler
def load_model_and_scaler():
    global model, scaler
    with open("saved_model/random_forest_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("saved_model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

# Create a class for the house features
class House(BaseModel):
    

@app.get("/")
async def read_root():
    return {"message": "Welcome to the House Price Prediction API"}

@app.post("/predict")
def predict(house: House):
    # Convert the data to a numpy array
    data = np.array([getattr(house, feature) for feature in []]).reshape(1, -1)
    
    # Load the model and scaler
    load_model_and_scaler()
    
    # Scale the input data
    scaled_data = scaler.transform(data)
    
    # Make a prediction
    prediction = model.predict(scaled_data).tolist()
    
    return {"predicted_price": round(prediction[0], 2)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
