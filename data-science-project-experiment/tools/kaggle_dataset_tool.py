
from crewai_tools import BaseTool
from kaggle.api.kaggle_api_extended import KaggleApi

class KaggleDatasetDownloader(BaseTool):
    name: str = "Kaggle Dataset Downloader"
    description: str = "Downloads datasets from Kaggle using a provided URL."

    def _run(self, url: str) -> str:
        try:
            # authenticate API
            api = KaggleApi()
            api.authenticate()
            # Extract dataset info from URL
            parts = url.split('/')
            owner = parts[-2]
            dataset_name = parts[-1]
            
            # Download the dataset
            api.dataset_download_files(f"{owner}/{dataset_name}", path='./raw_data', unzip=True)
            return f"Successfully downloaded dataset: {owner}/{dataset_name} to ./raw_data directory"
        except Exception as e:
            if '403' in str(e):
                return "Error 403: Forbidden. Please check your Kaggle API credentials and dataset permissions."
            else:
                return f"Error downloading dataset: {str(e)}"

