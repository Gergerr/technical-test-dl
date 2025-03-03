import streamlit as st
import plotly.express as px
import pandas as pd
from src.askdata.components.preprocess import preprocess_data, integrate_llm

# Streamlit app configuration
st.set_page_config(page_title="Ask Data - Superstore", page_icon="📊", layout="wide")

# Title and description
st.title("Ask Data - Superstore")
st.markdown("Ask any question about the Superstore dataset, and get an answer with a visualization (if exists)!")

# Input query
query = st.text_input("Enter your question:", placeholder="e.g., What is the total profit for all orders in Spain?")

# Process query and display result
if st.button("Get Answer"):
    if query:
        with st.spinner("Processing your question..."):
            try:
                # Get table info and answer
                data_info = preprocess_data()
                response, result_df = integrate_llm(data_info, query, return_df=True)
                
                # Escape dollar signs to prevent LaTeX rendering
                response = response.replace("$", r"\$")
                
                # Display the text answer
                st.success("Answer:")
                st.write(response)  # Markdown with escaped $, or use st.text(response) for plain text
                
                # Visualization logic
                if not result_df.empty:
                    st.subheader("Visualization")
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

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a question!")

# Sample questions
with st.expander("Sample Questions"):
    st.write("""
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
    """)