from crewai import Agent
import os
from langchain_openai import ChatOpenAI
from tools.kaggle_dataset_tool import KaggleDatasetDownloader
from tools.data_processing_tool import DataPreprocessor
from tools.model_training_tool import TrainingModelTool
from tools.model_evaluation_tool import ModelEvaluationTool
from tools.fastapi_code_generate_tool import ModelAnalyzerTool

from crewai_tools import SerperDevTool, DirectoryReadTool, FileReadTool, CSVSearchTool
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv()) # important line if cannot load api key

# Getting the api keys from the .env file
os.environ["SERPER_API_KEY"] = os.getenv('SERPER_API_KEY')
os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


## LLM model

# llm = Ollama(model="llama3") # local LLM 

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# load crewai tools
serper_search_tool = SerperDevTool()
docs_tool_a = DirectoryReadTool(directory='raw_data')
docs_tool_b = DirectoryReadTool(directory='processed_data')
read_tool = FileReadTool(file_path='processed_data/processed_data.csv')
csv_search_tool = CSVSearchTool()

# load custom tools
kaggle_tool = KaggleDatasetDownloader()
data_processing_tool = DataPreprocessor()
model_training_tool = TrainingModelTool()
model_evaluation_tool = ModelEvaluationTool()
code_generator_tool = ModelAnalyzerTool()

class DataAgents():
    
    def data_collection_agent(self):
        return Agent(
            role='Data Acquisition Specialist',
            goal='Find and download appropriate datasets on a given topic',
            backstory='Expert in acquiring datasets from various sources, specializing in climate data',
            tools=[serper_search_tool, kaggle_tool],
            llm=llm,
            verbose=True
        )
        
    
    def data_processing_agent(self):
        return Agent(
            role="Data Preprocessing Specialist",
            goal="Load, clean, and perform initial transformations on datasets",
            backstory="Expert in data cleaning and preprocessing using pandas, numpy, and sklearn libraries",
            llm=llm,
            tools=[docs_tool_a, data_processing_tool],
            # allow_code_execution=True
        )
    
    
    def model_training_agent(self):
        return Agent(
            role="Random Forest Model Trainer",
            goal="Train an Random Forest model to predict house prices accurately",
            backstory="You are an expert in machine learning, specializing in Random Forest for regression tasks.",
            tools=[docs_tool_b, model_training_tool],
            llm=llm
        )
        
    
    def model_evaluation_agent(self):
        return Agent(
            role="Model Evaluator",
            goal="Evaluate the saved Random Forest model using the test data and generate a comprehensive evaluation report",
            backstory="You are an expert in machine learning model evaluation, specializing in regression tasks.",
            tools=[model_evaluation_tool],
            llm=llm
        )
        
    
    # def code_generation_agent(self):
    #     return Agent(
    #         role = "FastAPI Code Generator",
    #         goal = "Generate main.py file to run FastAPI server",
    #         backstory="You are an expert in Python code generation and evaluation, specializing in fastAPI",
    #         tools=[code_generator_tool],
    #         llm=llm,
    #         allow_code_execution=True
    #     )
        