import streamlit as st
from google.cloud import bigquery
from src.askdata.components.preprocess import integrate_llm
from src.askdata.components.visualization import generate_visualization
import os
from google.cloud import storage

# Konfigurasi Streamlit untuk local testing
st.set_page_config(layout="wide", page_title="ASK the Data Web App")

st.title("ASK the Data Web App (Cloud-based, Local Testing)")

# Inisialisasi BigQuery client
bigquery_client = bigquery.Client()

# Query data penuh dari BigQuery buat preprocess (bukan cuma LIMIT 5)
bq_dataset = "ask_data_dataset"
bq_table = "superstore_data"
data_query = f"SELECT * FROM `{bq_dataset}.{bq_table}` LIMIT 10000"  # Ambil lebih banyak data
data = bigquery_client.query(data_query).to_dataframe()

# Preview data (5 baris pertama) buat UI
st.write("Preview Data dari BigQuery:", data.head())

# Input query dari user (testing lokal)
query = st.text_input("Tanya tentang data (contoh: 'Berapa jumlah order_id unik?' atau 'Buat visualisasi distribusi GMV per kategori')")

if query:
    # Siapkan data_info pake data penuh dari BigQuery
    data_summary = f"Dataset ini punya {len(data)} baris dan kolom: {', '.join(data.columns)}. Order_id unik: {len(data['order_id'].unique())}."
    detailed_data = {
        "columns": data.columns.tolist(),
        "all_data": data.to_dict(orient='records'),  # Semua data
        "stats": data.describe().to_dict()
    }
    data_info = {"summary": data_summary, "details": detailed_data}

    # Preprocess dan integrasi ke LLM pake RAG dari cloud
    llm_response = integrate_llm(data_info, query, "technical-test-datalabs", "vector_store/")
    st.write("Jawaban LLM:\n", llm_response)

    # Visualisasi otomatis (kalo ada query visualisasi)
    if "distribusi" in query.lower() or "visualisasi" in query.lower():
        output_path = "visualization.png"
        generate_visualization(bq_dataset, bq_table, query, "technical-test-datalabs", output_path)
        # Tampilin visualisasi dari Cloud Storage
        storage_client = storage.Client()
        blob = storage_client.bucket("technical-test-datalabs").blob(f"visualizations/{os.path.basename(output_path)}")
        blob.download_to_filename(output_path)
        st.image(output_path, caption="Visualisasi Data", use_column_width=True)
        # Hapus file lokal sementara
        os.remove(output_path)