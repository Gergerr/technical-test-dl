import streamlit as st
import plotly.express as px
import pandas as pd
from google.cloud import bigquery
from src.askdata.components.preprocess import preprocess_data, integrate_llm, load_config

# Streamlit app configuration
st.set_page_config(page_title="Superstore Query App", page_icon="📊", layout="wide")

# Title and description
st.title("Superstore Query App")
st.markdown("Ask any question about the Superstore dataset, and get an answer with a visualization!")

# Input query
query = st.text_input("Enter your question:", placeholder="e.g., What is the total profit for all orders in Spain?")
st.markdown("""
    <style>
        div.stButton > button {
            background-color: red;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 8px 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([1, 1])

# Left column: Visualization and SQL Query
with col1:
    if st.button("Get Answer"):
        if query:
            with st.spinner("Processing your question..."):
                try:
                    data_info = preprocess_data()
                    response, result_df, sql_query = integrate_llm(data_info, query, return_df=True)
                    
                    # Escape dollar signs
                    response = response.replace("$", r"\$")
                    
                    # Display the text answer
                    st.subheader("Answer")
                    st.write(response)

                    # Visualization logic
                    if not result_df.empty:
                        if len(result_df.columns) == 1 and len(result_df) == 1:
                            st.write("Single value queries don’t have a visualization, here’s the result:")
                            st.write(result_df.iloc[0, 0])
                        elif len(result_df) > 1:
                            if len(result_df.columns) == 2:
                                x_col = result_df.columns[0]
                                y_col = result_df.columns[1]
                                if "profit" in y_col.lower() or "gmv" in y_col.lower() or "quantity" in y_col.lower():
                                    fig = px.bar(result_df, x=x_col, y=y_col, title=f"{y_col.capitalize()} by {x_col.capitalize()}")
                                else:
                                    fig = px.pie(result_df, names=x_col, values=y_col, title=f"{y_col.capitalize()} Distribution")
                                st.plotly_chart(fig, use_container_width=True)
                            elif len(result_df.columns) > 2:
                                fig = px.bar(result_df, x=result_df.columns[0], y=result_df.columns[1], 
                                            title=f"{result_df.columns[1].capitalize()} by {result_df.columns[0].capitalize()}")
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.write("No visualization available for this result.")
                    else:
                        st.write("No data to visualize.")

                    # Display SQL Query
                    st.subheader("SQL Query")
                    st.code(sql_query, language="sql")

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a question!")

# Right column: Table Preview (moved inside query logic or keep static with error handling)
with col2:
    try:
        st.subheader("Table Preview")
        config = load_config()
        bq_client = bigquery.Client(credentials=config["gcp"].get("credentials")) if config["gcp"].get("credentials") else bigquery.Client()
        preview_query = f"SELECT * FROM `{config['gcp']['bq_dataset']}.{config['gcp']['bq_table']}` LIMIT 5"
        preview_df = bq_client.query(preview_query).to_dataframe()
        st.dataframe(preview_df)
    except Exception as e:
        st.error(f"Could not load table preview: {str(e)}")

# Sample questions
with st.expander("Sample Questions"):
    st.write("""
    **Non Visualization Based Questions:**
    - What is the total profit for all orders in Spain?
    - How many orders were placed in France?
    - What’s the average quantity ordered in Germany?
    - Which category has the highest total profit?
    - How much GMV was generated in the South region?
    - List the top 5 customers by total profit in Spain.
    - What’s the most common ship mode in 2014?
    - What’s the total quantity sold for Office Supplies in Spain?
    - Which sub-category had the lowest profit in France?
    - How many unique customers ordered in the Consumer segment?
    
    **Visualization-Based Questions:**
    - What is the total profit trend over the years?
    - What is the distribution of order quantity?
    - What is the total GMV for each category?
    - Which cities contribute the most to total profit?
    - What is the proportion of each category in terms of profit?
    - What is the average profit per order per year?
    - What is the total profit for each sub-category?
    - What is the profit distribution across different shipping modes?
    - What is the correlation between GMV and Profit?
    - Which months have the highest number of orders?
    """)
