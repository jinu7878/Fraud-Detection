# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import base64
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go


# Set page configuration
st.set_page_config(
    page_title="Online Payment Fraud Detection",
    page_icon="🔍",
    layout="wide"
)

# Load the saved model and scaler
@st.cache_resource
def load_model():
    try:
        with open('random_forest_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

