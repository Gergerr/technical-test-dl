import pandas as pd
from src.askdata import logger
import os
from vertexai.preview.generative_models import GenerativeModel
import vertexai

def preprocess_data(data_path: str, output_path: str) -> str:
    """Preprocess data and prepare summary for LLM."""
    try:
        data = pd.read_csv(data_path)
        data.fillna('', inplace=True)
        data.drop_duplicates(inplace=True)

        data_summary = f"Dataset ini memiliki kolom: {', '.join(data.columns)}. Total baris: {len(data)}. Total order_id unik: {len(data['order_id'].unique())}."
        logger.info(f"Preprocessed data summary: {data_summary}")

        data.to_csv(output_path, index=False)
        return data_summary
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise

def integrate_llm(data_summary: str, query: str) -> str:
    """Integrate data with Google Gemini via Vertex AI and get response."""
    try:
        # Inisialisasi Vertex AI
        vertexai.init(project="datalabs-test-452308", location="us-central1")  # Ganti sesuai project dan region

        # Load model Gemini
        model = GenerativeModel("gemini-pro")

        # Query LLM dengan prompt yang lebih spesifik, hindari duplikat data_summary
        prompt = f"Dengan dataset ini: {query}"
        response = model.generate_content(prompt)
        llm_response = response.text.strip() if response.text else "No response"
        
        # Pastikan nggak ada duplikat data_summary di jawaban
        if data_summary in llm_response:
            llm_response = llm_response.replace(data_summary, "").strip()
        
        #logger.info(f"LLM response for query '{query}': {llm_response}")
        return llm_response
    except Exception as e:
        logger.error(f"Error in LLM integration: {str(e)}")
        raise

if __name__ == "__main__":
    data_path = "data/processed_data.csv"
    output_path = "data/preprocessed_data.csv"
    query = "Berapa jumlah order_id unik dalam dataset ini?"

    # Pastikan folder data ada
    os.makedirs("data", exist_ok=True)

    # Preprocess data
    summary = preprocess_data(data_path, output_path)
    print("Data summary for LLM:\n", summary)

    # Integrasi ke LLM
    response = integrate_llm(summary, query)
    print("Jawaban LLM:\n", response)