from google.cloud import bigquery
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig
import vertexai
import yaml
from pathlib import Path
from src.askdata import logger
import re
import streamlit as st
import json
from google.oauth2 import service_account

def load_config(config_path: str = "config/config.yaml") -> dict:
    try:
        if "gcp" in st.secrets:
            config = {}
            for section in st.secrets:
                config[section] = dict(st.secrets[section])
            if "service_account_key" in config.get("gcp", {}):
                credentials = service_account.Credentials.from_service_account_info(
                    json.loads(config["gcp"]["service_account_key"])
                )
                config["gcp"]["credentials"] = credentials
            return config
        config_file = Path(__file__).resolve().parents[3] / config_path
        with open(config_file, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def get_table_schema() -> list:
    config = load_config()
    credentials = config["gcp"].get("credentials")
    bq_client = bigquery.Client(credentials=credentials) if credentials else bigquery.Client()
    table_ref = f"{config['gcp']['bq_dataset']}.{config['gcp']['bq_table']}"
    table = bq_client.get_table(table_ref)
    return [field.name for field in table.schema]

def preprocess_data() -> dict:
    config = load_config()
    try:
        columns = get_table_schema()
        data_summary = (
            f"Dataset table: {config['gcp']['bq_dataset']}.{config['gcp']['bq_table']}. "
            f"Columns: {', '.join(columns)}."
        )
        logger.info(f"Data summary: {data_summary}")
        return {"summary": data_summary}
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

def integrate_llm(data_info: dict, query: str, return_df: bool = False) -> tuple:
    config = load_config()
    credentials = config["gcp"].get("credentials")
    vertexai.init(project=config["gcp"]["project_id"], location=config["gcp"]["location"], credentials=credentials)
    try:
        model = GenerativeModel(config["llm"]["model_name"])
        generation_config = GenerationConfig(**config["llm"]["generation_config"])

        prompt = (
            f"{data_info['summary']}\n"
            f"User question: {query}\n"
            f"Generate a valid SQL query to answer the question using BigQuery table "
            f"`{config['gcp']['bq_dataset']}.{config['gcp']['bq_table']}`. "
            f"Use 'total_profit' for profit-related queries unless specified otherwise. "
            f"For date operations, use BigQuery functions like EXTRACT (e.g., EXTRACT(YEAR FROM order_date)), "
            f"FORMAT_DATE, or DATE_TRUNC instead of STRFTIME, which BigQuery does not support. "
            f"Return only the SQL query as plain text, no markdown (e.g., no ```sql or ```), "
            f"no explanations, and no extra formatting."
        )
        response = model.generate_content(prompt, generation_config=generation_config)
        sql_query_raw = response.text.strip()
        sql_query = re.sub(r'```sql|```', '', sql_query_raw).strip()
        sql_query = re.sub(r"STRFTIME\('%Y', (\w+)\)", r"EXTRACT(YEAR FROM \1)", sql_query, flags=re.IGNORECASE)
        sql_query = re.sub(r"STRFTIME\('%Y-%m', (\w+)\)", r"FORMAT_DATE('%Y-%m', \1)", sql_query, flags=re.IGNORECASE)
        logger.info(f"Generated SQL: {sql_query}")

        bq_client = bigquery.Client(credentials=credentials) if credentials else bigquery.Client()
        result_df = bq_client.query(sql_query).to_dataframe()

        if not result_df.empty:
            if len(result_df.columns) == 1 and len(result_df) == 1:
                answer = f"The answer is {result_df.iloc[0, 0]}."
            else:
                answer = result_df.to_string(index=False)
        else:
            answer = "No data found for this query."

        refine_prompt = (
            f"Dataset: {data_info['summary']}\n"
            f"SQL result: {answer}\n"
            f"User question: {query}\n"
            f"Provide a concise, natural language answer based on the SQL result."
        )
        refined_response = model.generate_content(refine_prompt, generation_config=generation_config)
        llm_response = refined_response.text.strip() if refined_response.text else answer
        logger.info(f"LLM response: {llm_response}")

        if return_df:
            return llm_response, result_df, sql_query  # Return SQL query as well
        return llm_response
    except Exception as e:
        logger.error(f"Error in LLM integration: {str(e)}")
        raise

if __name__ == "__main__":
    config = load_config()
    query = "What is the total profit for all orders in Spain?"
    data_info = preprocess_data()
    response = integrate_llm(data_info, query)
    print("LLM Answer:\n", response)