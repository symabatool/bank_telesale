import streamlit as st
import pandas as pd
import pickle 
import numpy as np
from sklearn.pipeline import Pipeline
# Load the trained model and preprocessing steps

pipeline = pickle.load(open('pipeline.pkl', 'rb'))
st.image('bank3.jpg')
st.title('Bank TeleSales deposit Prediction App')
st.write("This app is based on 16 inputs that predict wheather a customer will deposit or not? Using this app, a bank can identify specific customer segments; that will make deposits.")
st.write("Please use the following form to get started!")

# Main content - Two Columns
col1, col2 = st.columns(2)

    # Content in Column 1
with col1:
    st.header("Customer Characteristics ")
 
# Arrays for categorical features
job_values = ['admin.', 'technician', 'services', 'management', 'retired', 'blue-collar', 'unemployed', 'entrepreneur', 'housemaid', 'unknown', 'self-employed', 'student']
marital_values = ['married', 'single', 'divorced']
education_values = ['secondary', 'tertiary', 'primary', 'unknown']
default_values = ['no', 'yes']
housing_values = ['yes', 'no']
loan_values = ['no', 'yes']
contact_values = ['unknown', 'cellular', 'telephone']
month_values = ['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'jan', 'feb', 'mar', 'apr', 'sep']
poutcome_values = ['unknown', 'other', 'failure', 'success']

      # Create a user-input dictionary
user_input = {
    'age':  st.slider('Select Age Range', min_value=18, max_value=75, value=35),
    'job': st.selectbox('Select Job', job_values),
    'marital': st.selectbox('Select Marital Status', marital_values),
    'education': st.selectbox('Select Education', education_values),
    'default': st.selectbox('Select Default', default_values),
    'balance':  st.slider('Select Balance', min_value=0, max_value=25000, value=5000),
    'housing': st.selectbox('Select Housing', housing_values),
    'loan': st.selectbox('Select Loan', loan_values),
    'contact': st.selectbox('Select Contact', contact_values),
    'day': st.slider('Select Day', min_value=1, max_value=31, value=15),
    'month': st.selectbox('Select Month', month_values),
    'duration': st.slider('Select Duration', min_value=0, max_value=2000, value=500),
    'campaign': st.slider('Select Campaign', min_value=1, max_value=50, value=10),
    'pdays': st.slider('Select Pdays', min_value=-1, max_value=50, value=-1),
    'previous': st.slider('Select Previous', min_value=0, max_value=50, value=0),
    'poutcome': st.selectbox('Select Poutcome', poutcome_values)
}
def predict_deposit(user_input, pipeline):

    user_input_pd=pd.DataFrame.from_dict(user_input,orient = 'index').T
    
    # Make predictions using the pipeline
    y_pred = pipeline.predict(user_input_pd.head(1))

    deposit = 'Will not Subscribe'
    if y_pred[0] == 'yes':
        deposit = 'Will Subscribe'

    #st.success(deposit)
    st.success(f'Predicted User behaviour: {deposit}')

if st.button("Predict deposit"):
    predict_deposit(user_input, pipeline)

with st.sidebar:
    st.image('bank3.jpg', width = 300)
    st.title("Bank Telesales")
    st.header("Financial Package")
    st.subheader("Bank Deposit prediction(weather user will deposit money or not)")
    st.markdown(""" The ability to accurately predict customer tendencies in subscribing to a term deposit holds immense significance for banks. 
    \nBy leveraging machine learning techniques, banks can gain valuable insights into customer behavior and preferences, enabling them to make informed decisions regarding their marketing strategies and product offerings. 
    \nThis predictive capability allows banks to optimize their resources, tailor their communication efforts, and effectively allocate their marketing budgets, ultimately leading to improved customer acquisition, retention, and profitability. 
    \n Furthermore, by identifying potential customers who are more likely to subscribe to a term deposit, banks can enhance their overall risk management and optimize their loan portfolios, thereby strengthening their financial stability and resilience.""")
    st.markdown("""What is a Term Deposit? A Term deposit is a deposit that a bank or a financial institution offers with a fixed rate (often better than just opening a deposit account) in which your money will be returned at a specific maturity time. For more information with regards to Term Deposits please click on this link from Investopedia """)
st.markdown("""Retired client has high interest on deposit client who has housing loan seems to be not interested much on deposit
if pre campaign outcome that is poutcome=success then, there is a high chance of client to show interest on deposit
in March, September, October, and December, the client show high interest to deposit
in the month of may, records are high but client interest ratio is very less""")

      # Content in Column 2
# with col2:
#     # Load CSV data
#     df = pd.read_csv('C:/Users/Dell/Downloads/internship_bank/Nbank_Tele.csv')

# # Select a specific row for prediction
# selected_row = df.iloc[14:15, :]

# # Display the selected row
# st.dataframe(selected_row)

