import joblib
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import os
import matplotlib.pyplot as plt

# Load the saved model
log_reg_loaded = joblib.load(r'C:\Users\prasa\OneDrive\Desktop\infosys_online\website\random_forest.pkl')

# Transaction types for dropdown
transaction_types = {
    "Select Transaction Type": -1,
    "Cash-in": 0,
    "Cash-out": 1,
    "Debit": 2,
    "Payment": 3,
    "Transfer": 4,
    "Deposit": 5,
}

# Create or load transaction history
history_path = "transactions_history.csv"
if not os.path.exists(history_path):
    pd.DataFrame(columns=["Transaction Type", "Amount", "Old Balance", "New Balance", "Prediction"]).to_csv(history_path, index=False)

# Set Streamlit page configuration
st.set_page_config(page_title="Fraud Detection System", layout="wide")

# App-wide Background Style
st.markdown("""
    <style>
        body {
            background-color: #051856;
            color: black;
            # font-size:1.5em;
        }
        .stApp {
            # font-size:3em;
            background-color: #0093E9;
background-image: linear-gradient(160deg, #0093E9 0%, #80D0C7 100%);

            # background-color: #FFFFFF;
# background-image: linear-gradient(180deg, #FFFFFF 0%, #6284FF 50%, #FF0000 100%);
            color:black;

            # background-image: url('https://www.oktopayments.com/wp-content/uploads/2023/11/online-payment-methods-meaning.png'); /* Replace with your image URL */
            # background-size: cover; /* Ensures the image covers the entire screen */
            # background-repeat: no-repeat; /* Prevents tiling */
            # background-position: center; /* Centers the image */
            # background: rgb(63,94,251);
            # background: radial-gradient(circle, rgba(63,94,251,1) 0%, rgba(252,70,107,1) 100%);
            # color: white; /* Adjust text color for readability */
            
        }
        .card {
            background-color: white;
            border-radius: 15px;
            padding: 25px;
            margin: 10px 
0;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.15);
            background: rgba( 255, 255, 255, 0.25 );
            box-shadow: 0 8px 32px 0 rgba( 31, 38, 135, 0.37 );
            backdrop-filter: blur( 4px );
            -webkit-backdrop-filter: blur( 4px );
            border-radius: 10px;
            border: 1px solid rgba( 255, 255, 255, 0.18 );
        }
        div.stButton > button {
        appearance: none;
        background-color: transparent;
        border: 2px solid #1A1A1A;
        border-radius: 15px;
        box-sizing: border-box;
        color: #3B3B3B;
        cursor: pointer;
        display: inline-block;
        font-family: Roobert, -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, 
                     "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
        font-size: 16px;
        font-weight: 600;
        line-height: normal;
        margin: 0;
        min-height: 60px;
        min-width: 0;
        outline: none;
        padding: 16px 24px;
        text-align: center;
        text-decoration: none;
        transition: all 300ms cubic-bezier(.23, 1, 0.32, 1);
        user-select: none;
        -webkit-user-select: none;
        touch-action: manipulation;
        width: 100%;
        will-change: transform;
    }

    div.stButton > button:disabled {
        pointer-events: none;
    }

    div.stButton > button:hover {
        color: #fff;
        background-color: #1A1A1A;
        box-shadow: rgba(0, 0, 0, 0.25) 0 8px 15px;
        transform: translateY(-2px);
    }

    div.stButton > button:active {
        box-shadow: none;
        transform: translateY(0);
    }
#         div.stButton > button {
#         background-color: #F4D03F;
# background-image: linear-gradient(132deg, #F4D03F 0%, #16A085 100%);

#         color: white; /* Text color */
#         border: none; /* Remove border */
#         border-radius: 8px; /* Rounded corners */
#         padding: 10px 20px; /* Button padding */
#         font-size: 30px; /* Font size */
#         font-weight: bold; /* Font weight */
#         cursor: pointer; /* Pointer cursor */
#         # cursor: progress;
#     }
    

#     /* Optional: Hover effect for better interaction */
#     div.stButton > button:hover {
#         background-image: linear-gradient(160deg, #80D0C7 0%, #0093E9 100%);
#         box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
#     }
        # .stButton>button {
        #     background-color: #007bff;
        #     color: white;
        #     border: none;
        #     padding: 10px 20px;
        #     border-radius: 5px;
        #     font-size: 16px;
        # }
        # .stButton>button:hover {
        #     background-color: #0056b3;
        # }
        .fraud {
            display: flex;
    justify-content: center;  /* Horizontally centers the content */
    # align-items: center;
            align-items:center;
            background-color: lightcoral;
            color: white;
            height:40px;
            width:200%;
            border-radius:5px;
        }
        .not-fraud {
            display: flex;
    justify-content: center;  /* Horizontally centers the content */
    # align-items: center;
            align-items:center;
            background-color: lightgreen;
            color: black;
            width:200%;
            height:40px;
            border-radius:5px;
        }
        div.stAlert {
            background-color: #f1004b;
background-image: linear-gradient(180deg, #f1004b 0%, #FF0000 33%, #ff0000 66%);

                        # background: rgb(131,58,180);
                        # background: linear-gradient(90deg, rgba(131,58,180,1) 0%, rgba(253,29,29,1) 50%, rgba(252,176,69,1) 100%) !important;
                        font-size:2em;
                        width:200%;
                        color: white;
                        blur:10%;
                        border-radius: 10px;
                        padding: 10px;
                    }
        label {
        color: black !important;
        # font-size:1.2em;
        font-size: 60px !important; /* Adjust font size as needed */
    }
    /* Optional: Change the font color of specific sections or titles */
    .section-title {
        color: black;        
    }
    .fixed-image {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            z-index: 9999;
        }
        .content {
            margin-top: 120px;  /* Adjust this value based on the height of your image */
        }
    </style>
""", unsafe_allow_html=True)


st.markdown('<img src="{./top1.png}" class="fixed-image">', unsafe_allow_html=True)
# Horizontal Menu with updated styles
selected = option_menu(
    menu_title=None,
    options=["Home", "Single Transaction", "Bulk Upload", "Transaction History", "About"],
    icons=["house", "currency-dollar", "upload", "list-task", "info-circle"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#f8f9fa","border-radius":"60px",},
        "icon": {"color": "black", "font-size": "22px"},
        "nav-link": {
            "font-size": "20px",
            "color": "black",
            "text-align": "center",
            "margin": "0px",
            "padding": "10px",
            "--hover-color": "#e9ecef",
        },
        # "nav-link-selected": {"background: rgb(63,94,251);background: radial-gradient(circle, rgba(63,94,251,1) 0%, rgba(252,70,107,1) 100%);background-color": "#007bff", "color": "black"},
        "nav-link-selected": {"background: radial-gradient(circle at 1.2% 5.3%, rgb(255, 85, 166) 0%, rgb(255, 154, 98) 100.2%);background-color": "pink", "color": "black","border-radius":"25px","scale":"0.85"},
    },
)

if "transaction_type" not in st.session_state:
    st.session_state.transaction_type = "Select Transaction Type"
if "transaction_amount" not in st.session_state:
    st.session_state.transaction_amount = 0.0
if "old_balance" not in st.session_state:
    st.session_state.old_balance = 0.0
if "new_balance" not in st.session_state:
    st.session_state.new_balance = 0.0


# Pages Logic
if selected == "Home":
    st.markdown('<h1>Online Payment Fraud Detection</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 18px; line-height: 1.8;">
        Welcome to the <b>Online Payment Fraud Detection Tool</b>! This application is designed to safeguard your financial transactions by identifying and preventing fraudulent activities using cutting-edge machine learning technologies. 
    </div>
    
    <h2>What Can This Tool Do?</h2>
    <ul style="font-size: 16px; line-height: 1.8;">
        <li><b>Detect Fraud in Real-Time:</b> Analyze individual transactions instantly to determine their legitimacy.</li>
        <li><b>Bulk Transaction Analysis:</b> Upload transaction datasets to detect anomalies across multiple records simultaneously.</li>
        <li><b>Track Transaction History:</b> Maintain a detailed log of past transactions for review and compliance purposes.</li>
        <li><b>Educate and Inform:</b> Access comprehensive resources to understand fraud trends and how to protect yourself.</li>
    </ul>

    <h2>How It Works:</h2>
    <ol style="font-size: 16px; line-height: 1.8;">
        <li><b>Data Input:</b> Provide transaction data manually or upload a file containing multiple transactions.</li>
        <li><b>Fraud Detection Analysis:</b> The system uses machine learning algorithms trained on historical fraud data to flag suspicious transactions.</li>
        <li><b>Output Report:</b> Receive a detailed report highlighting potential fraudulent activities with actionable insights.</li>
    </ol>
    
    <h2>Why Choose This Tool?</h2>
    <p style="font-size: 16px; line-height: 1.8;">
        Fraudulent transactions not only cause financial losses but also erode trust in digital platforms. Our tool offers:
    </p>
    <ul style="font-size: 16px; line-height: 1.8;">
        <li><b>High Accuracy:</b> Machine learning algorithms continuously learn and adapt to new fraud patterns.</li>
        <li><b>User-Friendly Interface:</b> Designed for simplicity and ease of use, enabling anyone to detect fraud effortlessly.</li>
        <li><b>Data Security:</b> Built with robust encryption and compliance with industry standards to protect your sensitive data.</li>
    </ul>

    <h2>Steps to Get Started:</h2>
    <p style="font-size: 16px; line-height: 1.8;">
        Follow these easy steps to begin using the Online Payment Fraud Detection Tool:
    </p>
    <ol style="font-size: 16px; line-height: 1.8;">
        <li>Navigate to the <b>Fraud Detection</b> tab.</li>
        <li>Input transaction details or upload your dataset.</li>
        <li>Click <b>Analyze</b> to view results.</li>
        <li>Explore the <b>Tips</b> section to learn about secure online payment practices.</li>
    </ol>

    <h2>Stay Ahead of Fraud:</h2>
    <p style="font-size: 16px; line-height: 1.8;">
        Fraud prevention starts with awareness and proactive measures. By leveraging advanced technologies, 
        this tool empowers you to stay one step ahead of cybercriminals. Whether you're an individual or a business, 
        safeguarding your transactions has never been more important.
    </p>
    
    <p style="font-size: 16px; line-height: 1.8; text-align: center; color: #007BFF;">
        Start your journey toward secure digital transactions today!
    </p>
    """, unsafe_allow_html=True)


elif selected == "Single Transaction":
    # Layout with columns for left-aligned inputs and right-aligned button/results
    col0,col1 ,col2 = st.columns([1,2, 1])  # First column is wider, second column is for right alignment

    # Left column: Input fields
    with col1:
        st.markdown('<h2 class="section-title">Single Transaction Fraud Detection</h2>', unsafe_allow_html=True)
        st.session_state.transaction_type = st.selectbox("Transaction Type", options=list(transaction_types.keys()), index=list(transaction_types.keys()).index(st.session_state.transaction_type))
        st.session_state.transaction_amount = st.number_input("Transaction Amount", min_value=0.0, value=st.session_state.transaction_amount, help="Amount should be greater than 0.")
        st.session_state.old_balance = st.number_input("Old Balance of Origin Account", min_value=0.0, value=st.session_state.old_balance)
        st.session_state.new_balance = st.number_input("New Balance of Origin Account", min_value=0.0, value=st.session_state.new_balance)

    # Right column: Button and Results
    # with col2:
        # Predict button on the right
        col3, col4 = st.columns([1, 1])  # First column is wider, second column is for right alignment
        with col3:
            
            if st.button("Predict", key="predict_single"):
                if st.session_state.transaction_type == "Select Transaction Type":
                    st.error("Transaction type not selected! Please choose a valid transaction type.")
                elif st.session_state.transaction_amount <= 0:
                    st.error("Transaction amount must be greater than 0.")
                else:
                    # Create input data for the prediction
                    input_data = pd.DataFrame({
                        'type': [transaction_types[st.session_state.transaction_type]],
                        'amount': [st.session_state.transaction_amount],
                        'oldbalanceOrg': [st.session_state.old_balance],
                        'newbalanceOrig': [st.session_state.new_balance]
                    })
                    
                    # Predict the result
                    prediction = log_reg_loaded.predict(input_data)
                    prediction_text = "Fraud" if prediction[0] == "Fraud" else "Not Fraud"
                    color_class = "fraud" if prediction_text == "Fraud" else "not-fraud"
                    
                    # Display prediction result
                    st.markdown(f"""
                    <div class="highlight-box {color_class}">
                        <strong>Prediction: {prediction_text}</strong>
                    </div>
                    """, unsafe_allow_html=True)

                    # Log the transaction
                    transaction = pd.DataFrame([[st.session_state.transaction_type, st.session_state.transaction_amount, st.session_state.old_balance, st.session_state.new_balance, prediction_text]],
                                            columns=["Transaction Type", "Amount", "Old Balance", "New Balance", "Prediction"])
                    transaction.to_csv(history_path, mode='a', header=False, index=False)
        with col4:
                # Reset button
            if st.button("Reset", key="reset_button"):
                # Reset the form fields by clearing session state values
                st.session_state.transaction_type = "Select Transaction Type"
                st.session_state.transaction_amount = 0.0
                st.session_state.old_balance = 0.0
                st.session_state.new_balance = 0.0
                # st.experimental_rerun()  # Rerun the app to reflect changes

elif selected == "Bulk Upload":
    st.markdown('<h2 class="section-title">Bulk Transaction Fraud Detection</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    
    if uploaded_file:
        transactions = pd.read_csv(uploaded_file)
        required_columns = ["Transaction Type", "Amount", "Old Balance", "New Balance"]
        
        # Check if required columns are present
        if all(column in transactions.columns for column in required_columns):
            # Map 'Transaction Type' to integer values for prediction
            transactions["Transaction Type"] = transactions["Transaction Type"].map(transaction_types)
            
            # Handle rows with invalid 'Transaction Type' (e.g., empty or unrecognized types)
            transactions = transactions[transactions["Transaction Type"].notna()]
            
            # Prepare the data for prediction, include 'Transaction Type'
            input_data = transactions[required_columns].values  # Include 'Transaction Type'
            
            # Predict using the model
            predictions = log_reg_loaded.predict(input_data)
            
            # Map numeric predictions back to 'Fraud' or 'Not Fraud'
            transactions["Prediction"] = ["Fraud" if pred == "Fraud" else "Not Fraud" for pred in predictions]
            
            # Map 'Transaction Type' back to original values for display (reverse map)
            reverse_transaction_types = {v: k for k, v in transaction_types.items()}
            transactions["Transaction Type"] = transactions["Transaction Type"].map(reverse_transaction_types)
            
            # Display the results with appropriate styling
            st.write("### Prediction Results")
            st.dataframe(transactions.style.applymap(
                lambda x: "background-color: lightcoral; color: white;" if x == "Fraud" else "background-color: lightgreen; color: black;",
                subset=["Prediction"]
            ))
            
            # Save the results to the history file
            transactions.to_csv(history_path, mode='a', header=False, index=False)
        else:
            st.error(f"CSV must include columns: {required_columns}")

elif selected == "Transaction History":
    st.markdown('<h2 class="section-title">Transaction History</h2>', unsafe_allow_html=True)

    history_path = "transactions_history.csv"  # Update the path accordingly

    if os.path.exists(history_path):
        history = pd.read_csv(history_path)
        
        # Create two columns for organizing layout
        col1, col2 = st.columns([2, 1])  # Adjust column widths as needed
        
        with col1:
            # Displaying the transaction history with colored predictions
            st.dataframe(history.style.map(
                lambda x: "background-color: lightcoral; color: white;" if x == "Fraud" else "background-color: lightgreen; color: black;",
                subset=["Prediction"]
            ))
        
        with col2:
            history_path = "transactions_history.csv"

            # Check if the file exists
            if os.path.exists(history_path):
                # Read the CSV file
                history = pd.read_csv(history_path)

                # Group the data by Transaction Type and Prediction (Fraud or Not Fraud)
                transaction_fraud_counts = history.groupby(['Transaction Type', 'Prediction']).size().unstack(fill_value=0)

                # Prepare data for the pie chart
                pie_data = transaction_fraud_counts.stack()

                # Create a figure and axis for a larger pie chart
                fig, ax = plt.subplots(figsize=(32, 32))  # Increased size further to make it larger

                # Define the colors for the pie chart (using a distinct color palette)
                colors = plt.cm.Paired.colors

                # Plot the pie chart with percentages displayed on the chart
                wedges, texts, autotexts = ax.pie(
                    pie_data, 
                    labels=pie_data.index, 
                    autopct='%1.1f%%', 
                    startangle=90, 
                    colors=colors,
                    wedgeprops={'edgecolor': 'black', 'linewidth': 1.5},  # Adds borders to the wedges for better visibility
                    pctdistance=0.75  # Moved percentage text a bit further from the center to avoid congestion
                )

                # Equal aspect ratio ensures the pie is drawn as a circle.
                ax.axis('equal')

                # Set title for the chart
                # ax.set_title('Distribution of Fraud and Not Fraud for Each Transaction Type', fontsize=20)

                # Customize the labels' appearance
                for text in texts + autotexts:
                    text.set_fontsize(14)
                    text.set_fontweight('bold')

                # Adding a legend to differentiate between Fraud and Not Fraud
                ax.legend(
                    labels=pie_data.index, 
                    loc="upper left", 
                    fontsize=14, 
                    title="Transaction Type and Fraud Status",
                    bbox_to_anchor=(1.2, 1)  # Move legend outside of the pie chart for better spacing
                )

                # Display the pie chart using Streamlit
                st.pyplot(fig)
            # history_path = "transactions_history.csv"

            # # Check if the file exists
            # if os.path.exists(history_path):
            #     # Read the CSV file
            #     history = pd.read_csv(history_path)

            #     # Group the data by Transaction Type and Prediction (Fraud or Not Fraud)
            #     transaction_fraud_counts = history.groupby(['Transaction Type', 'Prediction']).size().unstack(fill_value=0)

            #     # Prepare data for the pie chart
            #     pie_data = transaction_fraud_counts.stack()

            #     # Create a figure and axis for a larger pie chart
            #     fig, ax = plt.subplots(figsize=(14, 14))  # Increased size further to make it larger

            #     # Define the colors for the pie chart (using a distinct color palette)
            #     colors = plt.cm.Paired.colors

            #     # Plot the pie chart with percentages displayed on the chart
            #     wedges, texts, autotexts = ax.pie(
            #         pie_data, 
            #         labels=pie_data.index, 
            #         autopct='%1.1f%%', 
            #         startangle=90, 
            #         colors=colors,
            #         wedgeprops={'edgecolor': 'black', 'linewidth': 1.5},  # Adds borders to the wedges for better visibility
            #         pctdistance=0.85  # Moves percentage text closer to the edge
            #     )

            #     # Equal aspect ratio ensures the pie is drawn as a circle.
            #     ax.axis('equal')

            #     # Set title for the chart
            #     ax.set_title('Distribution of Fraud and Not Fraud for Each Transaction Type', fontsize=16)

            #     # Customize the labels' appearance
            #     for text in texts + autotexts:
            #         text.set_fontsize(12)
            #         text.set_fontweight('bold')

            #     # Adding a legend to differentiate between Fraud and Not Fraud
            #     ax.legend(
            #         labels=pie_data.index, 
            #         loc="upper left", 
            #         fontsize=16, 
            #         title="Transaction Type and Fraud Status",
            #         bbox_to_anchor=(1.2, 1)  # Move legend outside of the pie chart
            #     )
                
            #     # Display the pie chart using Streamlit
            #     st.pyplot(fig)
            # history_path = "transactions_history.csv"

            # # Check if the file exists
            # if os.path.exists(history_path):
            #     # Read the CSV file
            #     history = pd.read_csv(history_path)

            #     # Count the occurrences of each transaction type
            #     transaction_counts = history['Transaction Type'].value_counts()

            #     # Create a pie chart
            #     fig, ax = plt.subplots()
            #     ax.pie(transaction_counts, labels=transaction_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
            #     ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

            #     # Set title
            #     ax.set_title('Transaction Types Distribution')

            #     # Display the pie chart using Streamlit
            #     st.pyplot(fig)

    else:
        st.write("No transaction history available.")


elif selected == "About":
    # st.markdown('<h2 class="section-title">About Online Payment Fraud Detection</h2>', unsafe_allow_html=True)
    # st.markdown("""
    # Fraudulent activities in online payments are on the rise due to the rapid growth of digital transactions. 
    # Cybercriminals use various techniques, such as phishing, identity theft, and malware, to exploit vulnerabilities 
    # in online payment systems. Protecting users and businesses from such fraud requires advanced tools and awareness.
    # """)
    
    # # Add an image related to fraud detection
    # st.markdown('<img class="img" style="margin-left:100px;" src="https://5logistics.com/wp-content/uploads/Fraud-1.jpg" alt="Fraud Detection">', unsafe_allow_html=True)
        
    # st.markdown("""
    # ### Features of Fraud Detection:
    
    # - *Real-Time Analysis:* 
    #   Fraudulent transactions are identified as they happen, minimizing potential losses. Real-time monitoring uses algorithms to detect anomalies in payment patterns.
    # - *Machine Learning:* 
    #   Leveraging historical data, machine learning models can identify patterns and predict fraudulent behavior. These systems improve over time, adapting to new threats.
    # - *Multi-Layer Security:* 
    #   Modern fraud detection systems combine multiple security protocols, including encryption, tokenization, and biometric authentication, to ensure robust protection.
    # - *Risk Scoring:* 
    #   Transactions are assigned risk scores based on factors like geolocation, transaction amount, and device used, helping to identify suspicious activities.
    # - *Behavioral Analytics:* 
    #   Monitoring user behavior, such as login patterns and spending habits, to flag deviations that may indicate fraud.
    # """)
    
    # # Add an image for machine learning or analytics
    # st.markdown('<img style="margin-left:100px;margin-right:auto;" src="https://www.digipay.guru/static/24fb1b1f75d3f9ddb1373c2e1cebbd75/16546/online-payment-security-Image_04.png" alt="Fraud Detection">', unsafe_allow_html=True)
    
    # st.markdown("""
    # ### Tips for Staying Protected During Online Transactions:
    # - *Verify Website Security:* 
    #   - Only transact on websites with HTTPS protocols. Look for a padlock icon in the browser's address bar.
    #   - Avoid using public Wi-Fi for online transactions unless you're connected to a trusted VPN.
    # - *Enable Multi-Factor Authentication (MFA):*
    #   - Add an extra layer of security by requiring a one-time password (OTP) or biometric authentication alongside your login credentials.
    # - *Keep Devices Updated:*
    #   - Regularly update your operating system, browser, and payment apps to patch known vulnerabilities.
    # - *Monitor Bank Statements:*
    #   - Review your bank statements and transaction history regularly to detect unauthorized activities early.
    # - *Avoid Phishing Scams:*
    #   - Be cautious of emails, messages, or calls requesting sensitive information. Cybercriminals often pose as legitimate entities.
    # - *Use Virtual Cards or Wallets:* 
    #   - Where possible, use virtual debit/credit cards or digital wallets for online payments. These options provide an extra layer of protection by masking your actual card details.
    # - *Educate Yourself and Others:* 
    #   - Stay informed about common fraud techniques, such as skimming, spoofing, and account takeover fraud, to recognize red flags.
    # """)

    # # Add an image related to online security or phishing
    # st.markdown('<img style="height:400px;width:600px;" src="https://pbsorg.siuat.visa.com/content/dam/financial-literacy/practical-business-skills/images/non-card/types-of-fraud-graphic.jpg" alt="Fraud Detection">', unsafe_allow_html=True)

    
    # st.markdown("""
    # ### Common Types of Online Payment Fraud:
    # - *Phishing Attacks:*
    #   Fraudsters trick users into providing login credentials or credit card information through fake websites or emails.
    # - *Card-Not-Present (CNP) Fraud:*
    #   Unauthorized transactions occur using stolen card details during online payments.
    # - *Man-in-the-Middle (MitM) Attacks:*
    #   Hackers intercept communication between the user and the payment system to steal sensitive information.
    # - *Account Takeover:*
    #   Cybercriminals gain access to user accounts and initiate fraudulent transactions.
    # - *Chargeback Fraud:*
    #   Customers falsely claim a legitimate transaction was unauthorized to receive a refund.
    # """)

    # st.markdown('<img src="https://blogimage.vantagefit.io/vfitimages/2021/06/MENTEL-HEALTH-AWARNESS-CELEBRATION--1.png" alt="Fraud Detection">', unsafe_allow_html=True)


    # st.markdown("""
    # ### Raising Awareness:
    # - *Use Reputable Services:*
    #   Opt for well-known and trusted payment gateways.
    # - *Educate Your Network:* 
    #   Share tips and resources with friends and family to promote safer online transaction habits.
    # - *Report Suspicious Activity:* 
    #   Inform your bank or payment service provider immediately if you notice any unusual activity.
    # - *Leverage Fraud Detection Tools:* 
    #   Use services or tools that proactively monitor transactions and send alerts for suspicious activities.

    # By adopting these practices and leveraging advanced fraud detection tools, you can significantly minimize the risk of falling victim to online payment fraud.
    # """)
    st.markdown('<h2 class="section-title">About Online Payment Fraud Detection</h2>', unsafe_allow_html=True)
    st.markdown("""
    Fraudulent activities in online payments are on the rise due to the rapid growth of digital transactions. 
    Cybercriminals use various techniques, such as phishing, identity theft, and malware, to exploit vulnerabilities 
    in online payment systems. Protecting users and businesses from such fraud requires advanced tools and awareness.
    """)

    # Create two columns
    col1, col2 = st.columns([1, 1])

    with col2:
        # Add an image related to fraud detection
        st.markdown('<img class="img" style="margin-left:100px;height:300px;" src="https://5logistics.com/wp-content/uploads/Fraud-1.jpg" alt="Fraud Detection">', unsafe_allow_html=True)

    with col1:
        # Add text in the second column
        st.markdown("""
        ### Features of Fraud Detection:
        
        - *Real-Time Analysis:* 
        Fraudulent transactions are identified as they happen, minimizing potential losses. Real-time monitoring uses algorithms to detect anomalies in payment patterns.
        - *Machine Learning:* 
        Leveraging historical data, machine learning models can identify patterns and predict fraudulent behavior. These systems improve over time, adapting to new threats.
        - *Multi-Layer Security:* 
        Modern fraud detection systems combine multiple security protocols, including encryption, tokenization, and biometric authentication, to ensure robust protection.
        - *Risk Scoring:* 
        Transactions are assigned risk scores based on factors like geolocation, transaction amount, and device used, helping to identify suspicious activities.
        - *Behavioral Analytics:* 
        Monitoring user behavior, such as login patterns and spending habits, to flag deviations that may indicate fraud.
        """)

    

    # Create a new section for tips for staying protected during online transactions
    st.markdown("""### Tips for Staying Protected During Online Transactions:""")

    # Create two more columns for this section
    col3, col4 = st.columns([1, 1])

    with col3:
        st.markdown("""
        - *Verify Website Security:* 
        - Only transact on websites with HTTPS protocols. Look for a padlock icon in the browser's address bar.
        - Avoid using public Wi-Fi for online transactions unless you're connected to a trusted VPN.
        - *Enable Multi-Factor Authentication (MFA):*
        - Add an extra layer of security by requiring a one-time password (OTP) or biometric authentication alongside your login credentials.
        - *Keep Devices Updated:*
        - Regularly update your operating system, browser, and payment apps to patch known vulnerabilities.
        """)
        st.markdown("""
        - *Monitor Bank Statements:*
        - Review your bank statements and transaction history regularly to detect unauthorized activities early.
        - *Avoid Phishing Scams:*
        - Be cautious of emails, messages, or calls requesting sensitive information. Cybercriminals often pose as legitimate entities.
        """)

    with col4:
        # Add an image for machine learning or analytics
        st.markdown('<img style="margin-left:10px;margin-right:auto" src="https://www.digipay.guru/static/24fb1b1f75d3f9ddb1373c2e1cebbd75/16546/online-payment-security-Image_04.png" alt="Fraud Detection">', unsafe_allow_html=True)
        

    col7, col8 = st.columns([1, 1])

    with col7:
        # Raising awareness section
        st.markdown("""### Raising Awareness:""")
        st.markdown("""
        - *Use Reputable Services:*
        Opt for well-known and trusted payment gateways.
        - *Educate Your Network:* 
        Share tips and resources with friends and family to promote safer online transaction habits.
        - *Report Suspicious Activity:* 
        Inform your bank or payment service provider immediately if you notice any unusual activity.
        - *Leverage Fraud Detection Tools:* 
        Use services or tools that proactively monitor transactions and send alerts for suspicious activities.

        By adopting these practices and leveraging advanced fraud detection tools, you can significantly minimize the risk of falling victim to online payment fraud.
        """)

    with col8:
        st.markdown('<img src="https://blogimage.vantagefit.io/vfitimages/2021/06/MENTEL-HEALTH-AWARNESS-CELEBRATION--1.png" alt="Fraud Detection">', unsafe_allow_html=True)
    st.title("Feedback Form")

    # Add a short description
    st.write("We would love to hear your feedback! Please fill out the form below:")

    # Feedback Form
    with st.form(key='feedback_form'):
        # Collect user information (Optional)
        name = st.text_input("Your Name")
        email = st.text_input("Your Email (Optional)")

        # Collect feedback rating
        rating = st.radio("How would you rate your experience?", [1, 2, 3, 4, 5])

        # Collect detailed feedback
        feedback = st.text_area("Please provide your detailed feedback:")

        # Add a submit button
        submit_button = st.form_submit_button("Submit Feedback")

    # After the form is submitted
    if submit_button:
        if name and feedback:
            # Display the user's input (for demo purposes)
            st.write(f"Thank you for your feedback, {name}!")
            st.write(f"Rating: {rating}/5")
            st.write(f"Feedback: {feedback}")
            if email:
                st.write(f"Email: {email}")
        else:
            st.warning("Please make sure you provide both your name and feedback.")
