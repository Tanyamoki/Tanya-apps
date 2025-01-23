# Import the required packages
import streamlit as st
import pandas as pd
import altair as alt

# Page configuration
st.set_page_config(
    page_title="Iris Classification", 
    page_icon="assets/icon/icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

alt.themes.enable("dark")

# -------------------------
# Sidebar

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

with st.sidebar:
    st.title('Iris Classification')
    st.subheader("Pages")

    # List of pages for easier management
    pages = [
        ("About", 'about'),
        ("Dataset", 'dataset'),
        ("EDA", 'eda'),
        ("Data Cleaning / Pre-processing", 'data_cleaning'),
        ("Machine Learning", 'machine_learning'),
        ("Prediction", 'prediction'),
        ("Conclusion", 'conclusion')
    ]

    # Create buttons dynamically
    for title, page in pages:
        if st.button(title, use_container_width=True, key=page):
            set_page_selection(page)

    # Project Details
    st.subheader("Abstract")
    st.markdown("A Streamlit dashboard highlighting the results of a training two classification models using the Iris flower dataset from Kaggle.")
    st.markdown("📊 [Dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)")
    st.markdown("📗 [Google Colab Notebook](https://colab.research.google.com/drive/1KJDBrx3akSPUW42Kbeepj64ZisHFD-NV?usp=sharing)")
    st.markdown("🐙 [GitHub Repository](https://github.com/Zeraphim/Streamlit-Iris-Classification-Dashboard)")
    st.markdown("by: [`Zeraphim`](https://jcdiamante.com)")

# -------------------------

# Load data
df = pd.read_csv('iris.csv', delimiter=',')

# Set page title
st.title('ISJM BI - Exploration des données des Iris')

st.header('Pré-analyse visuelles données données des Iris TP1')

# Show Dataset with buttons for previewing DataFrame
if st.checkbox("Boutons de prévisualisation du DataFrame"):
    preview_buttons = {
        "Head": df.head(2),
        "Tail": df.tail(),
        "Infos": df.info(),
        "Shape": df.shape,
    }

    for title, output in preview_buttons.items():
        if st.button(title):
            st.write(output)

# Create chart
chart = alt.Chart(df).mark_point().encode(
    x='petal_length',
    y='petal_width',
    color="species"
)

# Display chart
st.write(chart)

# Interactive design representation 
chart2 = alt.Chart(df).mark_circle(size=60).encode(
    x='sepal_length',
    y='sepal_width',
    color='species',
    tooltip=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
).interactive()

st.write(chart2)

# About section
if st.button("About App"):
    st.subheader("App d'exploration des données des Iris")
    st.text("Construit avec Streamlit")
    st.text("Thanks to the Streamlit Team Amazing Work")

if st.checkbox("By"):
    st.text("Stéphane C. K. Tékouabou")
    st.text("ctekouaboukoumetio@gmail.com")
