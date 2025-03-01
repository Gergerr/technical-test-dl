from google.cloud import storage, bigquery
import pandas as pd
from src.askdata import logger
import os
import chromadb  # Untuk vector database di cloud
from tqdm import tqdm  # Untuk progress bar
import io

def ingest_data(bucket_name: str, source_blob_name: str, bq_dataset: str, bq_table: str, vector_store_path: str):
    """Ingest data from Google Cloud Storage to BigQuery and create/update vector store in Cloud Storage for RAG using PersistentClient."""
    try:
        # Inisialisasi clients
        storage_client = storage.Client()
        bigquery_client = bigquery.Client()

        # Download CSV dari Cloud Storage ke memori (nggak simpen lokal)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        data_bytes = blob.download_as_bytes()
        data = pd.read_csv(io.StringIO(data_bytes.decode('utf-8')))
        logger.info(f"Data CSV downloaded from {bucket_name}/{source_blob_name}")

        # Simpan ke BigQuery (buat query cepat)
        dataset_ref = bigquery_client.dataset(bq_dataset)
        table_ref = dataset_ref.table(bq_table)
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_TRUNCATE",
            create_disposition="CREATE_IF_NEEDED"
        )
        job = bigquery_client.load_table_from_dataframe(data, table_ref, job_config=job_config)
        job.result()  # Tunggu job selesai
        logger.info(f"Data loaded to BigQuery: {bq_dataset}.{bq_table}")

        # Buat vector store pake PersistentClient di lokal sementara, lalu upload ke Cloud Storage
        vector_store_local = "vector_store_temp/"  # Path lokal sementara buat PersistentClient
        os.makedirs(vector_store_local, exist_ok=True)

        # Buat PersistentClient Chroma dengan path lokal
        chroma_client = chromadb.PersistentClient(path=vector_store_local)
        # Pake get_or_create_collection untuk menghindari UniqueConstraintError
        collection = chroma_client.get_or_create_collection("ask_data_collection")

        # Batch processing buat semua kolom
        batch_size = 100
        documents = []
        ids = []
        for idx, row in tqdm(data.iterrows(), total=len(data), desc="Creating/Updating vector store"):
            text = ", ".join([f"{col}: {str(row[col])}" for col in data.columns])
            documents.append(text)
            ids.append(f"doc_{idx}")
            if len(documents) >= batch_size:
                collection.add(documents=documents, ids=ids)
                documents = []
                ids = []

        if documents:
            collection.add(documents=documents, ids=ids)

        # Vector store otomatis disimpan ke vector_store_temp/ karena PersistentClient
        logger.info(f"Vector store created/updated and persisted in {vector_store_local}")

        # Upload semua file di vector_store_temp/ ke Cloud Storage (handle subfolder)
        for root, _, files in os.walk(vector_store_local):
            for file_name in files:
                local_path = os.path.join(root, file_name)
                # Buat path di Cloud Storage sesuai struktur lokal
                cloud_path = os.path.join(vector_store_path, os.path.relpath(local_path, vector_store_local))
                blob = storage_client.bucket(bucket_name).blob(cloud_path)
                blob.upload_from_filename(local_path)
        logger.info(f"Vector store uploaded to {bucket_name}/{vector_store_path}")

        # Hapus folder lokal sementara
        import shutil
        shutil.rmtree(vector_store_local)

        print("DataFrame (preview):\n", data.head())  # Preview di console buat test cepat
        return data
    except Exception as e:
        logger.error(f"Error in data ingestion: {str(e)}")
        raise

if __name__ == "__main__":
    bucket_name = "technical-test-datalabs"
    source_blob_name = "ASK DATA - superstore_data.csv"
    bq_dataset = "ask_data_dataset"
    bq_table = "superstore_data"
    vector_store_path = "vector_store/"  # Path di Cloud Storage

    data = ingest_data(bucket_name, source_blob_name, bq_dataset, bq_table, vector_store_path)