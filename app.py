import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Data Analysis App", layout="wide")
st.title("Data Analysis App")
st.write("This is a simple data analysis app built with Streamlit.")
st.write("Upload a CSV file to get started.")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Display the first few rows of the dataframe
    st.write("First few rows of the dataframe:")
    st.dataframe(df.head())
    
    # Display basic statistics
    st.write("Basic statistics:")
    st.write(df.describe())
    
    # Display a histogram of the first column
    st.write("Histogram of the first column:")
    plt.hist(df.iloc[:, 0], bins=20)
    plt.xlabel(df.columns[0])
    plt.ylabel("Frequency")
    st.pyplot(plt)

else:
    
    st.write("Please upload a CSV file to see the data analysis.")