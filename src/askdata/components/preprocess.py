from google.cloud import bigquery, storage
from src.askdata import logger
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig
import vertexai
import chromadb  # Untuk vector database di cloud
import os

def preprocess_data(bq_dataset: str, bq_table: str, bucket_name: str, vector_store_path: str) -> dict:
    """Preprocess data from BigQuery, prepare summary, and load vector store from Cloud Storage for RAG."""
    try:
        # Inisialisasi BigQuery client
        bigquery_client = bigquery.Client()

        # Query data dari BigQuery
        query = f"SELECT * FROM `{bq_dataset}.{bq_table}` LIMIT 10000"  # Batasi buat test
        data = bigquery_client.query(query).to_dataframe()
        logger.info(f"Loaded data from BigQuery: {bq_dataset}.{bq_table}")

        # Ringkasan data
        data_summary = f"Dataset ini memiliki kolom: {', '.join(data.columns)}. Total baris: {len(data)}. Total order_id unik: {len(data['order_id'].unique())}."
        logger.info(f"Preprocessed data summary: {data_summary}")

        detailed_data = {
            "columns": data.columns.tolist(),
            "all_data": data.to_dict(orient='records'),  # Semua data
            "stats": data.describe().to_dict()
        }

        # Muat vector store dari Cloud Storage
        storage_client = storage.Client()
        vector_store_local = "vector_store_temp/"
        os.makedirs(vector_store_local, exist_ok=True)

        # Download vector store dari Cloud Storage
        for blob in storage_client.bucket(bucket_name).list_blobs(prefix=vector_store_path):
            blob.download_to_filename(os.path.join(vector_store_local, blob.name.split("/")[-1]))

        # Muat vector store pake Chroma pake PersistentClient
        chroma_client = chromadb.PersistentClient(path=vector_store_local)
        collection = chroma_client.get_collection("ask_data_collection")

        print("DataFrame (preview):\n", data.head())  # Preview di console buat test cepat
        return {"summary": data_summary, "details": detailed_data, "vector_store": collection}
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise

def integrate_llm(data_info: dict, query: str, bucket_name: str, vector_store_path: str) -> str:
    """Integrate data with Google Gemini via Vertex AI using RAG from Cloud Storage for web app."""
    try:
        # Inisialisasi Vertex AI
        vertexai.init(project="datalabs-test-452308", location="us-central1")  # Ganti sesuai project dan region

        # Load model Gemini
        model_dipake= "gemini-pro"
        model = GenerativeModel(model_dipake)
        generation_config = GenerationConfig(
                    temperature=0.2,  # Lebih deterministic
                    max_output_tokens=1024,  # Output panjang
                    top_p=0.9,  # Pilih token terbaik
                    top_k=40  # Pilih 40 token terbaik
                )
        # Muat vector store dari Cloud Storage (sementara lokal buat test, nanti cloud)
        storage_client = storage.Client()
        vector_store_local = "vector_store_temp/"
        os.makedirs(vector_store_local, exist_ok=True)

        # Download vector store dari Cloud Storage
        for blob in storage_client.bucket(bucket_name).list_blobs(prefix=vector_store_path):
            blob.download_to_filename(os.path.join(vector_store_local, blob.name.split("/")[-1]))

        # Muat vector store pake Chroma pake PersistentClient
        chroma_client = chromadb.PersistentClient(path=vector_store_local)
        collection = chroma_client.get_collection("ask_data_collection")

        # Retrieval: Cari data relevan dari vector store berdasarkan query, pake semua kolom
        results = collection.query(query_texts=[query], n_results=10)  # Ambil 10 dokumen paling relevan
        retrieved_context = "\n".join(results['documents'][0])

        # Augmentation: Tambah semua konteks data ke prompt, pake semua detail
        data_context = (f"Dengan dataset ini: {data_info['summary']}. "
                        f"Detail data: Kolom ada {data_info['details']['columns']}, "
                        f"semua data: {data_info['details']['all_data'][:5]} (dan lainnya), "
                        f"statistik: {data_info['details']['stats']}. "
                        f"Konteks relevan dari query: {retrieved_context}")
        prompt = f"{data_context}\nPertanyaan: {query}"
        response = model.generate_content(prompt, generation_config= generation_config)
        llm_response = response.text.strip() if response.text else "No response"
        
        # Pastikan nggak ada duplikat data_summary di jawaban
        if data_info['summary'] in llm_response:
            llm_response = llm_response.replace(data_info['summary'], "").strip()
        
        logger.info(f"LLM ({model_dipake}) response for query '{query}': {llm_response}")

        # Hapus folder lokal sementara
        import shutil
        shutil.rmtree(vector_store_local)

        return llm_response
    except Exception as e:
        logger.error(f"Error in LLM integration: {str(e)}")
        raise

if __name__ == "__main__":
    bq_dataset = "ask_data_dataset"
    bq_table = "superstore_data"
    bucket_name = "technical-test-datalabs"
    vector_store_path = "vector_store/"
    query = "Total profit dari semua order di Spanyol (Spain) berapa?"

    # Preprocess data dan bikin vector store di cloud
    data_info = preprocess_data(bq_dataset, bq_table, bucket_name, vector_store_path)
    print("Data summary for LLM:\n", data_info['summary'])

    # Integrasi ke LLM dengan RAG pake data cloud
    response = integrate_llm(data_info, query, bucket_name, vector_store_path)
    print("Jawaban LLM:\n", response)