import os
import pandas as pd
from crewai_tools import BaseTool
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor(BaseTool):
    name: str = "Data Preprocessor"
    description: str = "Preprocesses data by handling missing values, removing duplicates, and encoding categorical variables."

    def _run(self, file_path: str) -> str:
        # Load the data
        df = pd.read_csv(file_path)
        
        # Get initial info
        initial_shape = df.shape
        initial_missing = df.isnull().sum().sum()
        
        # Handle missing values
        df = df.dropna()  # or use df.fillna() with appropriate strategy
        
        # Remove duplicate entries
        df = df.drop_duplicates()
        
        # Identify categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        # Convert categorical variables to numerical
        label_encoder = LabelEncoder()
        for col in categorical_columns:
            df[col] = label_encoder.fit_transform(df[col])
        
        # Get final info
        final_shape = df.shape
        final_missing = df.isnull().sum().sum()
        
        # Save the processed data
        processed_file_path = os.path.join('processed_data', 'processed_data.csv')
        df.to_csv(processed_file_path, index=False)
        
        return f"""
        Data preprocessing completed:
        - Initial shape: {initial_shape}
        - Initial missing values: {initial_missing}
        - Final shape: {final_shape}
        - Final missing values: {final_missing}
        - Categorical variables encoded: {categorical_columns}
        - Duplicates removed
        - Processed data saved to: {processed_file_path}
        """
       
