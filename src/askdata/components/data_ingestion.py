from google.cloud import storage, bigquery
from google.cloud import aiplatform
from google.cloud.aiplatform_v1.types import index as gca_index  # For IndexDatapoint
import pandas as pd
import io
from tqdm import tqdm
from typing import List
import yaml
from pathlib import Path
import vertexai

# Assuming these are custom modules in your project
from src.askdata.components.embedding import create_embeddings  # Function to create embeddings
from src.askdata import logger  # Logger setup

def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): Path to the config file relative to the project root.

    Returns:
        dict: Configuration dictionary.
    """
    config_file = Path(__file__).resolve().parents[3] / config_path
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config

def ingest_data() -> pd.DataFrame:
    """
    Ingests data into BigQuery, creates embeddings, and updates the Vector Search index.

    Returns:
        pd.DataFrame: The ingested data.

    Raises:
        Exception: If any step in the ingestion process fails.
    """
    config = load_config()

    # Initialize Vertex AI
    vertexai.init(project=config["gcp"]["project_id"], location=config["gcp"]["location"])

    try:
        # --- 1. Load Data from GCS ---
        storage_client = storage.Client()
        bigquery_client = bigquery.Client()
        bucket = storage_client.bucket(config["gcp"]["bucket_name"])
        blob = bucket.blob(config["gcp"]["source_blob_name"])
        data_bytes = blob.download_as_bytes()
        data = pd.read_csv(
            io.StringIO(data_bytes.decode("utf-8")),
            parse_dates=["order_date", "ship_date"]
        )
        # Format dates as strings for BigQuery
        data["order_date"] = data["order_date"].dt.strftime("%Y-%m-%d")
        data["ship_date"] = data["ship_date"].dt.strftime("%Y-%m-%d")
        logger.info(
            f"Data CSV downloaded from {config['gcp']['bucket_name']}/{config['gcp']['source_blob_name']}"
        )

        # --- 2. Define BigQuery Schema ---
        schema = [
            bigquery.SchemaField("order_id", "STRING"),
            bigquery.SchemaField("order_date", "DATE"),
            bigquery.SchemaField("ship_date", "DATE"),
            bigquery.SchemaField("ship_mode", "STRING"),
            bigquery.SchemaField("customer_name", "STRING"),
            bigquery.SchemaField("segment", "STRING"),
            bigquery.SchemaField("city", "STRING"),
            bigquery.SchemaField("country", "STRING"),
            bigquery.SchemaField("region", "STRING"),
            bigquery.SchemaField("category", "STRING"),
            bigquery.SchemaField("sub_category", "STRING"),
            bigquery.SchemaField("gmv", "FLOAT"),
            bigquery.SchemaField("profit", "FLOAT"),
            bigquery.SchemaField("quantity", "INTEGER"),
            bigquery.SchemaField("cost", "FLOAT"),
            bigquery.SchemaField("total_gmv", "FLOAT"),
            bigquery.SchemaField("total_cost", "FLOAT"),
            bigquery.SchemaField("total_profit", "FLOAT"),
            bigquery.SchemaField("lon", "FLOAT"),
            bigquery.SchemaField("lat", "FLOAT"),
        ]

        # --- 3. Load Data into BigQuery ---
        dataset_ref = bigquery_client.dataset(config["gcp"]["bq_dataset"])
        table_ref = dataset_ref.table(config["gcp"]["bq_table"])
        job_config = bigquery.LoadJobConfig(
            schema=schema,
            write_disposition="WRITE_TRUNCATE",
            create_disposition="CREATE_IF_NEEDED",
            source_format=bigquery.SourceFormat.CSV,
            skip_leading_rows=1,
            autodetect=False,
        )

        job = bigquery_client.load_table_from_dataframe(
            data, table_ref, job_config=job_config
        )
        job.result()

        if job.errors:
            logger.error(f"BigQuery load job errors: {job.errors}")
            raise Exception(f"BigQuery load job failed: {job.errors}")

        logger.info(
            f"Data loaded to BigQuery: {config['gcp']['bq_dataset']}.{config['gcp']['bq_table']}"
        )

        # --- 4. Create Embeddings and Update Vector Search Index ---
        embeddings_data = []
        batch_size = 100

        # Initialize the index outside the loop for efficiency
        my_index = aiplatform.MatchingEngineIndex(config["gcp"]["index_name"])

        for idx, row in tqdm(data.iterrows(), total=len(data), desc="Creating embeddings"):
            text = (
                f"Category: {row['category']}, Sub-Category: {row['sub_category']}, "
                f"Order ID: {row['order_id']}, Customer: {row['customer_name']}"
            )
            embeddings_data.append(text)

            if len(embeddings_data) >= batch_size:
                embeddings = create_embeddings(embeddings_data, config["embeddings"]["model_name"])
                ids = [str(i) for i in range(idx - len(embeddings_data) + 1, idx + 1)]
                to_upsert = [
                    gca_index.IndexDatapoint(
                        datapoint_id=id,
                        feature_vector=embedding
                    )
                    for id, embedding in zip(ids, embeddings)
                ]
                my_index.upsert_datapoints(to_upsert)
                embeddings_data = []

        # Process any remaining data
        if embeddings_data:
            embeddings = create_embeddings(embeddings_data, config["embeddings"]["model_name"])
            ids = [str(i) for i in range(len(data) - len(embeddings_data), len(data))]
            to_upsert = [
                gca_index.IndexDatapoint(
                    datapoint_id=id,
                    feature_vector=embedding
                )
                for id, embedding in zip(ids, embeddings)
            ]
            my_index.upsert_datapoints(to_upsert)

        logger.info("Embeddings created and Vector Search index updated.")
        return data

    except Exception as e:
        logger.error(f"Error in data ingestion: {str(e)}")
        raise

if __name__ == "__main__":
    ingest_data()