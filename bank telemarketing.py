import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Load the dataset
df = pd.read_csv('/kaggle/input/bank-marketing-dataset/bank.csv')

# Streamlit application layout
st.title("Bank Marketing Analysis")
st.subheader("Pages")

# Page selection
def set_page_selection(page):
    st.session_state.page_selection = page

if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
    st.session_state.page_selection = 'dataset'

if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
    st.session_state.page_selection = "eda"

if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
    st.session_state.page_selection = "data_cleaning"

if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
    st.session_state.page_selection = "machine_learning"

if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
    st.session_state.page_selection = "prediction"

if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
    st.session_state.page_selection = "conclusion"

# Dataset Page
if st.session_state.page_selection == 'dataset':
    st.subheader("Dataset")
    if st.checkbox("Show Head"):
        st.write(df.head(4))
    if st.checkbox("Show Tail"):
        st.write(df.tail())
    if st.checkbox("Show Info"):
        st.write(df.info())
    if st.checkbox("Show Shape"):
        st.write(df.shape)

# EDA Page
if st.session_state.page_selection == 'eda':
    st.subheader("Exploratory Data Analysis")
    st.text("Visualizations")
    
    # Grouping and plotting
    gj = df.groupby(['y', 'job'])['job'].count().unstack()
    st.bar_chart(gj)

    gm = df.groupby(['y', 'marital'])['marital'].count().unstack()
    st.bar_chart(gm)

    ge = df.groupby(['y', 'education'])['education'].count().transform(lambda x: x/x.sum()).unstack()
    st.bar_chart(ge)

    # Additional plots can be added similarly...

# Data Cleaning Page
if st.session_state.page_selection == 'data_cleaning':
    st.subheader("Data Cleaning / Pre-processing")
    df.drop(columns=['duration', 'pdays'], inplace=True)
    st.write("Data cleaned. Current shape:", df.shape)

# Machine Learning Page
if st.session_state.page_selection == 'machine_learning':
    st.subheader("Machine Learning")
    # Placeholder for ML model training and evaluation
    st.write("Machine learning models will be implemented here.")

# Prediction Page
if st.session_state.page_selection == 'prediction':
    st.subheader("Prediction")
    # Placeholder for prediction functionality
    st.write("Prediction functionality will be implemented here.")

# Conclusion Page
if st.session_state.page_selection == 'conclusion':
    st.subheader("Conclusion")
    st.write("Conclusions drawn from the analysis will be presented here.")
