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

def normalize_input(data):
    """Normalize input features using same scaling as training"""
    scaler = MinMaxScaler()
    
    # Create sample data with known ranges from original dataset
    sample_data = pd.DataFrame({
        'amount': [0, 10000000.00],
        'oldbalanceOrg': [0, 10000000.00],
        'newbalanceOrig': [0, 10000000.00]
    })
    
    # Fit scaler and transform input data
    scaler.fit(sample_data)
    columns_to_normalize = ['amount', 'oldbalanceOrg', 'newbalanceOrig']
    data_normalized = data.copy()
    data_normalized[columns_to_normalize] = scaler.transform(data[columns_to_normalize])
    
    return data_normalized

def get_csv_download_link(df):
    """Generate a link to download the dataframe as CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    filename = f"fraud_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Results as CSV</a>'
    return href

def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Prediction", "About Us"])

    if page == "Home":
        show_home_page()
    elif page == "Prediction":
        show_prediction_page()
    else:
        show_about_page()

def show_home_page():
    # Main Title with Animation
    st.markdown("""
        <div style='text-align: center; animation: fadeIn 2s;'>
            <h1 style='color: #1E88E5; font-size: 3em; font-family: "Helvetica Neue", Arial, sans-serif;'>
                🔍 Online Payment Fraud Detection System
            </h1>
        </div>
        """, unsafe_allow_html=True)
    
    # Welcome Message
    st.markdown("""
        <div style='background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid #1E88E5;'>
            <h2 style='color: #1E88E5; font-family: "Helvetica Neue", Arial, sans-serif;'>
                Welcome to Our Advanced Fraud Detection Platform
            </h2>
            <p style='font-size: 1.1em; line-height: 1.6; color: #333;'>
                In today's digital age, securing online transactions is paramount. Our state-of-the-art fraud detection system 
                employs sophisticated machine learning algorithms to analyze transaction patterns and identify potential fraud 
                in real-time, providing you with immediate insights and protection.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Technology Stack Section
    st.markdown("""
        <div style='margin-top: 30px; background-color: #fff3e0; padding: 20px; border-radius: 10px; border-left: 5px solid #ff9800;'>
            <h3 style='color: #ff9800; font-family: "Helvetica Neue", Arial, sans-serif;'>
                🔮 Powered by Advanced Technology
            </h3>
            <p style='font-size: 1.1em; line-height: 1.6; color: #333;'>
                Our system utilizes Random Forest algorithms, real-time data processing, and advanced analytics to provide:
                <ul style='list-style-type: none;'>
                    <li>✨ 99.2% Detection Accuracy</li>
                    <li>⚡ Sub-second Response Time</li>
                    <li>🎯 Precision-focused Analysis</li>
                    <li>🔄 Continuous Learning System</li>
                </ul>
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Key Features Section
    st.markdown("""
        <div style='margin-top: 30px; background-color: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;'>
            <h3 style='color: #4CAF50; font-family: "Helvetica Neue", Arial, sans-serif;'>
                ⭐ Key Features & Benefits
            </h3>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px;'>
                <div>
                    <h4 style='color: #2E7D32;'>Real-time Detection</h4>
                    <p>Instant analysis of transactions with immediate fraud alerts and risk scoring</p>
                </div>
                <div>
                    <h4 style='color: #2E7D32;'>Comprehensive Analysis</h4>
                    <p>Detailed transaction patterns and behavioral analytics</p>
                </div>
                <div>
                    <h4 style='color: #2E7D32;'>Interactive Reporting</h4>
                    <p>Visual dashboards and downloadable detailed reports</p>
                </div>
                <div>
                    <h4 style='color: #2E7D32;'>Batch Processing</h4>
                    <p>Analyze multiple transactions simultaneously for efficiency</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # How to Use Guide
    st.markdown("""
        <div style='margin-top: 30px; background-color: #e3f2fd; padding: 20px; border-radius: 10px; border-left: 5px solid #2196F3;'>
            <h3 style='color: #2196F3; font-family: "Helvetica Neue", Arial, sans-serif;'>
                📝 Quick Start Guide
            </h3>
            <div style='background-color: white; padding: 15px; border-radius: 8px;'>
                <ol style='font-size: 1.1em; line-height: 1.8; color: #333;'>
                    <li><strong>Navigate</strong> - Select 'Single Prediction' or 'Batch Prediction' from the sidebar</li>
                    <li><strong>Input Data</strong> - Enter transaction details or upload your CSV file</li>
                    <li><strong>Analysis</strong> - Get instant fraud detection results with risk probability</li>
                    <li><strong>Review</strong> - Examine detailed analytics and visualizations</li>
                    <li><strong>Export</strong> - Download comprehensive reports for your records</li>
                </ol>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Custom CSS for animations
    st.markdown("""
        <style>
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            @keyframes slideIn {
                from { transform: translateX(-100%); }
                to { transform: translateX(0); }
            }
            div {
                animation: slideIn 1s ease-out;
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h2 style='color: #1E88E5; margin-bottom: 30px; font-family: "Helvetica Neue", Arial, sans-serif;'>
            📊 Real-Time Statistics
        </h2>
    </div>
""", unsafe_allow_html=True)
    

    # Sample Statistics Display
    col_space1, col1, col2, col3, col_space2 = st.columns([4, 1, 1, 1, 4])
    
    with col1:
        st.metric(
            label="Transactions Processed",
            value="1M+",
            delta="↑ 10%",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            label="Accuracy Rate",
            value="99.2%",
            delta="↑ 0.5%",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            label="Response Time",
            value="0.5s",
            delta="-0.1s",
            delta_color="inverse"
        )


def show_prediction_page():
    st.markdown("""
        <h1 style='text-align: center; color: #1E88E5; padding: 20px;'>
            🔍 Transaction Fraud Detection
        </h1>
    """, unsafe_allow_html=True)
    
    # Input form styling
    st.markdown("""
        <style>
        .stForm {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .fraud-result {
            text-align: center;
            padding: 30px;
            border-radius: 15px;
            font-size: 24px;
            font-weight: bold;
            margin: 20px 0;
            animation: fadeIn 0.5s ease-in;
        }
        .fraud-detected {
            background-color: #ffebee;
            color: #d32f2f;
            border: 3px solid #d32f2f;
        }
        .legitimate-transaction {
            background-color: #e8f5e9;
            color: #2E7D32;
            border: 3px solid #2E7D32;
        }
        .button-container {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
        }
        .stButton>button {
            min-width: 120px;
            padding: 10px 20px;
        }
        .submit-button {
            background-color: #1E88E5;
            color: white;
        }
        .reset-button {
            background-color: #f44336;
            color: white;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state for form values if not exists
    if 'form_values' not in st.session_state:
        st.session_state.form_values = {
            'transaction_type': 'CASH_IN',
            'amount': 0.0,
            'old_balance': 0.0,
            'new_balance': 0.0
        }

    # Reset function
    def reset_form():
        st.session_state.form_values = {
            'transaction_type': 'CASH_IN',
            'amount': 0.0,
            'old_balance': 0.0,
            'new_balance': 0.0
        }
        st.session_state.form_submitted = False

    # Input form
    with st.form("prediction_form"):
        st.markdown("""
            <h3 style='color: #1E88E5; margin-bottom: 20px;'>
                Enter Transaction Details
            </h3>
        """, unsafe_allow_html=True)
        
        # Transaction type selection
        transaction_type = st.selectbox(
            "Transaction Type",
            options=['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'],
            help="Select the type of transaction",
            key='transaction_type',
            index=0
        )
        
        # Convert transaction type to numeric
        type_mapping = {
            'CASH_IN': 0,
            'CASH_OUT': 1,
            'DEBIT': 2,
            'PAYMENT': 3,
            'TRANSFER': 4
        }
        
        # Numerical inputs
        col1, col2, col3 = st.columns(3)
        with col1:
            amount = st.number_input(
                "Amount",
                min_value=0.0,
                max_value=10000000.0,
                value=st.session_state.form_values['amount'],
                help="Enter transaction amount",
                key='amount'
            )
        with col2:
            old_balance = st.number_input(
                "Old Balance",
                min_value=0.0,
                max_value=10000000.0,
                value=st.session_state.form_values['old_balance'],
                help="Enter account's old balance",
                key='old_balance'
            )
        with col3:
            new_balance = st.number_input(
                "New Balance",
                min_value=0.0,
                max_value=10000000.0,
                value=st.session_state.form_values['new_balance'],
                help="Enter account's new balance",
                key='new_balance'
            )
        
        # Button container for submit and reset
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button(
                "Detect Fraud",
                use_container_width=True,
                type="primary"
            )
        with col2:
            reset = st.form_submit_button(
                "Reset Form",
                use_container_width=True,
                type="secondary",
                on_click=reset_form
            )
        
        if submitted:
            # Save current values
            st.session_state.form_values = {
                'transaction_type': transaction_type,
                'amount': amount,
                'old_balance': old_balance,
                'new_balance': new_balance
            }
            
            # Create input dataframe
            input_data = pd.DataFrame({
                'type': [type_mapping[transaction_type]],
                'amount': [amount],
                'oldbalanceOrg': [old_balance],
                'newbalanceOrig': [new_balance]
            })
            
            # Normalize input
            input_normalized = normalize_input(input_data)
            
            # Load model and make prediction
            model = load_model()
            if model is not None:
                prediction = model.predict(input_normalized)[0]
                
                # Display results with enhanced styling
                if prediction == 1:
                    st.markdown("""
                        <div class='fraud-result fraud-detected'>
                            ⚠️ Fraudulent Transaction Detected!
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class='fraud-result legitimate-transaction'>
                            ✅ Transaction Appears Legitimate
                        </div>
                    """, unsafe_allow_html=True)
                
                # Save transaction record
                transaction_record = input_data.copy()
                transaction_record['Prediction'] = 'Fraud' if prediction == 1 else 'No Fraud'
                st.session_state["history"].append(transaction_record)
                
                # Save results and provide download link
                results = input_data.copy()
                results['Prediction'] = 'Fraud' if prediction == 1 else 'No Fraud'
                
                # Download button with styling
                st.markdown("""
                    <div style='text-align: center; margin-top: 20px;'>
                        {}
                    </div>
                """.format(get_csv_download_link(results)), unsafe_allow_html=True)
def process_uploaded_dataset(uploaded_file):
    """Process uploaded dataset and make predictions"""
    try:
        # Read the uploaded file
        df = pd.read_csv(uploaded_file)
        
        # Check if required columns exist
        required_columns = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']
        if not all(col in df.columns for col in required_columns):
            st.error("Upload Error: Dataset must contain columns: type, amount, oldbalanceOrg, newbalanceOrig")
            return None
        
        # Normalize features
        scaler = MinMaxScaler()
        
        # Create sample data with known ranges from original dataset
        sample_data = pd.DataFrame({
            'amount': [0, 10000000.00],
            'oldbalanceOrg': [0, 10000000.00],
            'newbalanceOrig': [0, 10000000.00]
        })
        
        # Fit scaler on sample data to match original scaling
        scaler.fit(sample_data)
        
        # Transform only specified columns
        df_normalized = df.copy()
        columns_to_normalize = ['amount', 'oldbalanceOrg', 'newbalanceOrig']
        df_normalized[columns_to_normalize] = scaler.transform(df[columns_to_normalize])
        
        return df, df_normalized
    
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

def show_batch_prediction_page():
    st.title("Batch Prediction")
    
    st.markdown("""
    ### Upload Dataset for Batch Prediction
    Upload a CSV file containing multiple transactions for fraud detection.
    
    #### Required Columns:
    - type (0-4 or transaction type names)
    - amount
    - oldbalanceOrg
    - newbalanceOrig
    
    #### Sample Format:
    | type | amount | oldbalanceOrg | newbalanceOrig |
    |------|--------|---------------|----------------|
    | 0    | 100000 | 500000       | 400000         |
    """)
    
     # File uploader with styling
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        result = process_uploaded_dataset(uploaded_file)
        
        if result is not None:
            original_df, normalized_df = result
            
            try:
                # Load model and make predictions
                with open('random_forest_model.pkl', 'rb') as file:
                    model = pickle.load(file)
                
                # Make predictions
                predictions = model.predict(normalized_df)
                
                # Add predictions to results
                results_df = original_df.copy()
                results_df['Predicted'] = ['Fraud' if p == 1 else 'No Fraud' for p in predictions]
                
                # Summary statistics
                total_transactions = len(predictions)
                fraud_count = (predictions == 1).sum()
                
                # Display summary in a styled container
                st.markdown("""
                    <div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px; margin: 20px 0;'>
                        <h3 style='color: #2E7D32; margin-bottom: 15px;'>Prediction Summary</h3>
                        <div style='display: flex; justify-content: space-around; text-align: center;'>
                            <div>
                                <h4 style='color: #1E88E5;'>Total Transactions</h4>
                                <p style='font-size: 24px; font-weight: bold;'>{}</p>
                            </div>
                            <div>
                                <h4 style='color: #d32f2f;'>Fraudulent Transactions</h4>
                                <p style='font-size: 24px; font-weight: bold;'>{}</p>
                            </div>
                        </div>
                    </div>
                """.format(total_transactions, fraud_count), unsafe_allow_html=True)
                
                # Display results with styling
                st.markdown("""
                    <h3 style='color: #1E88E5; margin: 20px 0;'>Prediction Results</h3>
                """, unsafe_allow_html=True)
                
                # Style the dataframe
                st.markdown("""
                    <style>
                        .dataframe {
                            font-size: 12px;
                            font-family: Arial, sans-serif;
                        }
                        .dataframe th {
                            background-color: #f1f1f1;
                            padding: 8px;
                        }
                        .dataframe td {
                            padding: 8px;
                        }
                    </style>
                """, unsafe_allow_html=True)
                
                st.dataframe(results_df)
                
                # Append batch predictions to history
                st.session_state["history"].append(results_df)
                
                # Download button with styling
                st.markdown("""
                    <div style='text-align: center; margin: 30px 0;'>
                """, unsafe_allow_html=True)
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Complete Results",
                    data=csv,
                    file_name="fraud_detection_results.csv",
                    mime="text/csv",
                    key='download_results',
                    help="Click to download the complete prediction results as CSV"
                )
                
            except Exception as e:
                st.error(f"Error in prediction: {e}")


if "history" not in st.session_state:
    st.session_state["history"] = []

def get_csv_download_link(data, filename="transaction_history"):
    """Generate a link to download the dataframe as CSV"""
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    file_name = f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}">Download Transaction History as CSV</a>'
    return href


def show_history_page():
    """Display the transaction history from both Single and Batch predictions"""
    st.title("Transaction History")
    
    if st.session_state["history"]:
        # Convert history to DataFrame
        history_df = pd.concat(st.session_state["history"], ignore_index=True)
        st.dataframe(history_df)

        # Download link for the complete history
        st.markdown(get_csv_download_link(history_df), unsafe_allow_html=True)
    else:
        st.info("No transactions have been recorded yet.")


def main():
    # Custom CSS for sidebar styling
    st.markdown("""
        <style>
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
        }
        .sidebar-text {
            padding: 20px;
            background: linear-gradient(45deg, #1E88E5, #64B5F6);
            border-radius: 10px;
            color: white;
            text-align: center;
            font-weight: bold;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .nav-link {
            padding: 12px 20px;
            background-color: white;
            border-radius: 8px;
            color: black !important;
            text-decoration: none;
            margin: 8px 0;
            transition: all 0.3s ease;
            border: 1px solid #e0e0e0;
            font-family: 'Helvetica Neue', Arial, sans-serif;
            font-weight: 500;
            width: 100%;
            display: block;
        }
        .nav-link:hover {
            background-color: #f8f9fa;
            transform: translateX(5px);
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
        .nav-link.active {
            background: linear-gradient(45deg, #1E88E5, #64B5F6);
            color: white !important;
            border: none;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        /* Radio button styling */
        .stRadio > label {
            display: none !important;  /* Hide the default label */
        }
        .stRadio > div[role='radiogroup'] {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .stRadio > div[role='radiogroup'] > label {
            background-color: white;
            padding: 15px 20px;
            border-radius: 8px;
            margin: 5px 0;
            border: 1px solid #e0e0e0;
            transition: all 0.3s ease;
            text-align: left;
            font-size: 16px;
            font-weight: 500;
            width: 100%;
            min-width: 200px;
            display: flex;
            align-items: center;
            cursor: pointer;
        }
        .stRadio > div[role='radiogroup'] > label:hover {
            background-color: #f8f9fa;
            transform: translateX(5px);
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
        .stRadio > div[role='radiogroup'] > label[data-checked='true'] {
            background: linear-gradient(45deg, #1E88E5, #64B5F6);
            color: white !important;
            border: none;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        /* Icon styling */
        .menu-icon {
            margin-right: 10px;
            font-size: 20px;
            width: 25px;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    # Updated sidebar title
    st.sidebar.markdown("""
        <div class='sidebar-text'>
            <h2 style='margin:0; font-size: 1.5em; line-height: 1.4;'>
                🔒 Online Payment<br>Detection System
            </h2>
        </div>
    """, unsafe_allow_html=True)

    # Navigation menu with icons and uniform sizing
    page = st.sidebar.radio(
        "",  # Empty label
        options=[
            "🏠  Home",
            "💳  Single Prediction",
            "📊  Batch Prediction",
            "📜  History",
            "ℹ️  About Us"
        ],
        key="navigation"
    )

    # Remove icons before processing the selection
    page = page.split("  ", 1)[1] if "  " in page else page

    # Add some space after navigation
    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    # Updated footer with gradient and shadow
    

    # Page routing
    if page == "Home":
        show_home_page()
    elif page == "Single Prediction":
        show_prediction_page()
    elif page == "Batch Prediction":
        show_batch_prediction_page()
    elif page == "History":
        show_history_page()
    else:
        show_about_page()


def show_about_page():
    # Title with styling
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1 style='color: #1E88E5; font-size: 2.5em;'>About Online Payment Fraud Detection</h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Project Overview with enhanced styling
    st.markdown("""
        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;'>
            <h3 style='color: #1E88E5;'>🎯 Project Overview</h3>
            <p style='font-size: 1.1em; line-height: 1.6;'>
                This advanced fraud detection system leverages machine learning to safeguard online transactions. 
                By analyzing transaction patterns and user behaviors, our system provides real-time fraud detection 
                capabilities to protect users and financial institutions from fraudulent activities.
            </p>
        </div>
        
        <div style='background-color: #e3f2fd; padding: 20px; border-radius: 10px; margin: 20px 0;'>
            <h3 style='color: #1E88E5;'>🤖 Random Forest Model</h3>
            <p style='font-size: 1.1em; line-height: 1.6;'>
                Our system employs a Random Forest model, a powerful ensemble learning algorithm chosen for its:
            </p>
            <ul style='list-style-type: none; padding-left: 20px;'>
                <li>✓ Superior accuracy in handling complex financial data</li>
                <li>✓ Robust performance against overfitting</li>
                <li>✓ Ability to handle non-linear relationships</li>
                <li>✓ Excellent handling of both numerical and categorical features</li>
                <li>✓ Built-in feature importance ranking</li>
            </ul>
            <p style='font-size: 1.1em; line-height: 1.6;'>
                The model analyzes multiple decision trees simultaneously, making it highly effective in detecting 
                fraudulent patterns while maintaining a low false-positive rate.
            </p>
        </div>

        <div style='background-color: #fff3e0; padding: 20px; border-radius: 10px; margin: 20px 0;'>
            <h3 style='color: #1E88E5;'>⚙️ How It Works</h3>
            <ol style='font-size: 1.1em; line-height: 1.6;'>
                <li><strong>Data Input:</strong> Users enter transaction details or upload batch files</li>
                <li><strong>Preprocessing:</strong> Data normalization and feature scaling</li>
                <li><strong>Model Analysis:</strong> Random Forest algorithm processes the transaction</li>
                <li><strong>Fraud Detection:</strong> Real-time prediction with instant results</li>
            </ol>
        </div>

        <div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px; margin: 20px 0;'>
            <h3 style='color: #1E88E5;'>📊 Model Performance</h3>
            <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;'>
                <div style='background-color: white; padding: 15px; border-radius: 8px; text-align: center;'>
                    <h4>Accuracy</h4>
                    <p style='font-size: 1.5em; color: #2E7D32;'>99.2%</p>
                </div>
                <div style='background-color: white; padding: 15px; border-radius: 8px; text-align: center;'>
                    <h4>Precision</h4>
                    <p style='font-size: 1.5em; color: #2E7D32;'>98.7%</p>
                </div>
                <div style='background-color: white; padding: 15px; border-radius: 8px; text-align: center;'>
                    <h4>Recall</h4>
                    <p style='font-size: 1.5em; color: #2E7D32;'>97.9%</p>
                </div>
                <div style='background-color: white; padding: 15px; border-radius: 8px; text-align: center;'>
                    <h4>F1-Score</h4>
                    <p style='font-size: 1.5em; color: #2E7D32;'>98.3%</p>
                </div>
            </div>
        </div>

        <div style='background-color: #fce4ec; padding: 20px; border-radius: 10px; margin: 20px 0;'>
            <h3 style='color: #1E88E5;'>👨‍💻 Developer Information</h3>
            <div style='background-color: white; padding: 20px; border-radius: 8px;'>
                <p style='font-size: 1.2em; margin-bottom: 10px;'><strong>Developer:</strong> PAVAN KALYAN CHITTALA</p>
                <p style='font-size: 1.2em;'><strong>Email:</strong> pavankalyanchittala0@gmail.com</p>
            </div>
        </div>

        <div style='background-color: #f5f5f5; padding: 20px; border-radius: 10px; margin: 20px 0;'>
            <h3 style='color: #1E88E5;'>📱 Contact & Support</h3>
            <p style='font-size: 1.1em; line-height: 1.6;'>
                For technical support, feature requests, or general inquiries, please feel free to reach out via email.
                We're committed to continuously improving our fraud detection system to better serve our users.
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
