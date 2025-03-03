from google.cloud import bigquery, storage
import matplotlib.pyplot as plt
import os
from src.askdata import logger
import pandas as pd

def generate_visualization(bq_dataset: str, bq_table: str, query: str, bucket_name: str, output_path: str):
    """Generate a general visualization from BigQuery data based on user query and save to Cloud Storage for web app."""
    try:
        # Inisialisasi BigQuery client
        bigquery_client = bigquery.Client()

        # Query data dari BigQuery berdasarkan query visualisasi user
        if "distribusi" in query.lower() or "visualisasi" in query.lower():
            # Parse query untuk ekstrak kolom dan grup (misalnya, "distribusi profit per region")
            query_words = query.lower().split()
            columns = [word for word in query_words if word in ['gmv', 'profit', 'quantity', 'cost', 'total_gmv', 'total_cost', 'total_profit', 'order_id']]
            groups = [word for word in query_words if word in ['kategori', 'category', 'region', 'ship_mode', 'segment', 'country', 'city']]

            if not columns or not groups:
                logger.warning(f"Query '{query}' tidak spesifik, pake default (distribusi order_id per kategori)")
                data_query = f"SELECT category, COUNT(order_id) as order_count FROM `{bq_dataset}.{bq_table}` GROUP BY category"
                chart_title = "Distribusi Jumlah Order per Kategori"
                x_label = "Kategori (Category)"
                y_label = "Jumlah Order"
            else:
                # Buat query dinamis berdasarkan kolom dan grup
                column = columns[0]  # Ambil kolom pertama (misalnya, 'profit')
                group = groups[0]    # Ambil grup pertama (misalnya, 'region')
                data_query = f"SELECT {group}, SUM({column}) as total_{column} FROM `{bq_dataset}.{bq_table}` GROUP BY {group}"
                chart_title = f"Distribusi Total {column.upper()} per {group.capitalize()}"
                x_label = group.capitalize()
                y_label = f"Total {column.upper()}"

            # Query data dari BigQuery
            data = bigquery_client.query(data_query).to_dataframe()
            if data.empty:
                logger.warning(f"Tidak ada data untuk query: {query}")
                raise ValueError("Tidak ada data untuk visualisasi ini")

            # Generate visualisasi (default pake bar chart, fleksibel untuk semua kolom)
            plt.figure(figsize=(10, 6))
            plt.bar(data[group], data[f'total_{column}'] if column != 'order_id' else data['order_count'])
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(chart_title)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            logger.info(f"Visualization saved locally to {output_path}")

        # Upload ke Cloud Storage
        storage_client = storage.Client()
        blob = storage_client.bucket(bucket_name).blob(f"visualizations/{os.path.basename(output_path)}")
        blob.upload_from_filename(output_path)
        logger.info(f"Visualization uploaded to {bucket_name}/visualizations/{os.path.basename(output_path)}")

    except Exception as e:
        logger.error(f"Error in visualization: {str(e)}")
        raise

if __name__ == "__main__":
    bq_dataset = "ask_data_dataset"
    bq_table = "superstore_data"
    bucket_name = "technical-test-datalabs"
    query = "Buat visualisasi distribusi profit per region"  # Contoh query general
    output_path = "visualization.png"
    generate_visualization(bq_dataset, bq_table, query, bucket_name, output_path)