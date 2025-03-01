from google.cloud import storage
import pandas as pd
import os
from src.askdata import logger

def ingest_data(bucket_name: str, source_blob_name: str, local_path: str):
    """Ingest data from Google Cloud Storage to local CSV."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(local_path)
        logger.info(f"Data CSV downloaded from {bucket_name}/{source_blob_name} to {local_path}")

        data = pd.read_csv(local_path)
        logger.info(f"Loaded data with columns: {data.columns}")
        return data
    except Exception as e:
        logger.error(f"Error in data ingestion: {str(e)}")
        raise

if __name__ == "__main__":
    # Asumsi bucket dan file dari Oktavia
    bucket_name = "technical-test-datalabs"  
    source_blob_name = "ASK DATA - superstore_data.csv"       
    local_path = "data/processed_data.csv"

    # Buat folder data kalo belum ada
    os.makedirs("data", exist_ok=True)

    data = ingest_data(bucket_name, source_blob_name, local_path)
    print("DataFrame:\n", data)  # Test print DataFrame

    
