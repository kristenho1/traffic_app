# Tool: ChaGPT (GPT-5)
# Purpose: Debugging errors related to prediction inputs, such as mismatched columns and data types; customizing app using markdown; 
#          sessions state management for prediction intervals
# Usage: Adopted session state management ideas; modified code to ensure columns and data types matched; modified markdown code based on suggestions for customizing
# Location: Documented here and further comments in traffic.ipynb

# Import necessary libraries
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mapie.regression import MapieRegressor


# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# title 
# Used chatGPT to help with rainbow text 
st.markdown("""<h1 style = "text-align: center;
            font-size: 60px;
            font-weight: bold;
            background: linear-gradient(to right, #ff512f, #f09819, #4caf50);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;">Traffic Volume Predictor</h1>""", unsafe_allow_html=True)

st.markdown("""<p style="text-align: center; font-size: 25px;">Utilize our advanced Machine Learning application to predict traffic volume</p>""", unsafe_allow_html=True)
st.image("traffic_image.gif", width=700)

# sidebar 
st.sidebar.image('traffic_sidebar.jpg', width = 400, caption ="Traffic Volume Predictor")
st.sidebar.subheader("**Input Features**")
st.sidebar.write("You can either upload your data file or manually enter input features.")

# loading original data to use for form 
orig_df = pd.read_csv('Traffic_Volume.csv')
orig_df['date_time'] = pd.to_datetime(orig_df['date_time'], format='%m/%d/%y %H:%M')
orig_df['hour'] = orig_df['date_time'].dt.hour
orig_df['weekday'] = orig_df['date_time'].dt.day_name()
orig_df['month'] = orig_df['date_time'].dt.month_name()

# options for uploading data or using form
with st.sidebar.expander("**Option 1: Upload CSV File**"):
    st.write("Upload a CSV file containing traffic details.")
    user_data = st.file_uploader('Choose a CSV file', type=['csv'])
    st.subheader("Sample Data Format for Upload")
    st.dataframe(pd.read_csv('traffic_data_user.csv').head(5), use_container_width=True)
    st.warning("⚠️**Ensure your uploaded file has the same column names and data types as shown above.**")

with st.sidebar.expander("**Option 2: Fill Out Form**"):
    st.write("enter the traffic detailes manually using the form below")
    holiday = st.selectbox("Choose whether today is a designated holiday", options = orig_df['holiday'].fillna('None').unique())
    temp = st.number_input("Average temperature in Kelvin", min_value = orig_df['temp'].min(), max_value = orig_df['temp'].max(), value = orig_df['temp'].mean())
    rain = st.number_input("Amount in mm of rain that occurred in the hour", min_value = orig_df['rain_1h'].min(), max_value = orig_df['rain_1h'].max(), value = orig_df['rain_1h'].mean())
    snow = st.number_input("Amount in mm of snow that occurred in the hour", min_value = orig_df['snow_1h'].min(), max_value = orig_df['snow_1h'].max(), value = orig_df['snow_1h'].mean())
    clouds = st.number_input("Percentage of cloud cover", min_value = orig_df['clouds_all'].min(), max_value = orig_df['clouds_all'].max())
    weather = st.selectbox("General weather condition", options = orig_df['weather_main'].unique())
    month = st.selectbox("Choose Month", options = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
    day= st.selectbox("Choose Day of the Week", options = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    hour = st.selectbox("Choose Hour of the Day", options = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
    predict_button = st.button("Submit Form Data", key="predict_sidebar")

# Load the pre-trained model from the pickle file
traffic_pickle = open('traffic.pickle', 'rb') 
reg_xg = pickle.load(traffic_pickle) 

if user_data is not None:
    st.success('✅ **CSV file uploaded successfully.**')
elif predict_button:
    st.success('✅ **Form data submitted successfully.**')
else: 
    st.info('ℹ️ **Please choose a data input method to proceed.**')

alpha = st.slider('Select alpha value estimating prediction intervals', min_value=0.01, max_value=0.50, value=0.1, step=0.01)

#Used ChatGPT to help with session state management
if "mapie" not in st.session_state:
    st.session_state.mapie = None
if "user_df_encoded" not in st.session_state:
    st.session_state.user_df_encoded = None
if "predicted" not in st.session_state:
    st.session_state.predicted = False

# Using predict() with new data provided by the user
# need to be in the same order as the training data 
if user_data is not None:
    input_df = pd.read_csv(user_data)
    orig_df = pd.read_csv('Traffic_Volume.csv')
    orig_df['date_time'] = pd.to_datetime(orig_df['date_time'], format='%m/%d/%y %H:%M')
    orig_df['hour'] = orig_df['date_time'].dt.hour
    orig_df['weekday'] = orig_df['date_time'].dt.day_name()
    orig_df['month'] = orig_df['date_time'].dt.month_name()

    # Loading data
    user_df = input_df # User provided data
    
    
    # Remove output (species) and year columns from original data
    orig_df = orig_df.drop(columns = ['traffic_volume', 'date_time'])
    
    # Ensure the order of columns in user data is in the same order as that of original data
    user_df = user_df[orig_df.columns]

    # Concatenate two dataframes together along rows (axis = 0)
    combined_df = pd.concat([orig_df, user_df], axis = 0)

    # Number of rows in original dataframe
    original_rows = orig_df.shape[0]

    # Create dummies for the combined dataframe
    cat = ['holiday', 'weather_main', 'weekday', 'month', 'hour']
    combined_df_encoded = pd.get_dummies(combined_df, columns=cat, drop_first=True)

    # Split data into original and user dataframes using row index
    original_df_encoded = combined_df_encoded[:original_rows]
    user_df_encoded = combined_df_encoded[original_rows:]

    # Predictions for user data
    user_pred, user_pis = reg_xg.predict(user_df_encoded, alpha = alpha)

    if "prev_alpha" not in st.session_state:
        st.session_state.prev_alpha = alpha

    # Detect change
    alpha_changed = alpha != st.session_state.prev_alpha
    st.session_state.prev_alpha = alpha  # update for next rerun

    # Now you can conditionally run logic
    if alpha_changed and st.session_state.predicted:
        user_pred, user_pis = st.session_state.mapie.predict(
            st.session_state.user_df_encoded, alpha=alpha
        )
    st.session_state.mapie = reg_xg
    st.session_state.user_df_encoded = user_df_encoded
    st.session_state.predicted = True
    
    # Add predictions and intervals to dataframe
    user_df['Predicted Volume'] = user_pred.round().astype(float)
    user_df['Lower Limit'] = user_pis[:, 0, 0].round().astype(float)
    user_df['Upper Limit'] = user_pis[:, 1, 0].round().astype(float)
    
    # Show the predicted species on the app
    st.subheader(f"Prediction Results with {(1-alpha)*100}% Confidence Level")
    st.dataframe(user_df)




elif predict_button:
    # Prepare data in same format as training
    user_df = pd.DataFrame([{
        'holiday': holiday, 'temp': temp, 'rain_1h': rain, 'snow_1h': snow,
        'clouds_all': clouds, 'weather_main': weather, 'month': month, 'weekday': day, 'hour': hour
    }])
    orig_df = pd.read_csv('Traffic_Volume.csv')
    orig_df['date_time'] = pd.to_datetime(orig_df['date_time'], format='%m/%d/%y %H:%M')
    orig_df['hour'] = orig_df['date_time'].dt.hour
    orig_df['weekday'] = orig_df['date_time'].dt.day_name()
    orig_df['month'] = orig_df['date_time'].dt.month_name()
    
    # Remove output (species) and year columns from original data
    original_df = orig_df.drop(columns = ['traffic_volume', 'date_time'])
    
    # Ensure the order of columns in user data is in the same order as that of original data
    user_df = user_df[original_df.columns]
    user_df['holiday'] = user_df['holiday'].replace('None', np.nan)

    # Concatenate two dataframes together along rows (axis = 0)
    combined_df = pd.concat([original_df, user_df], axis = 0)

    # Number of rows in original dataframe
    original_rows = original_df.shape[0]

    # Create dummies for the combined dataframe
    cat = ['holiday', 'weather_main', 'weekday', 'month', 'hour']
    combined_df_encoded = pd.get_dummies(combined_df, columns=cat, drop_first=True)

    # Split data into original and user dataframes using row index
    original_df_encoded = combined_df_encoded[:original_rows]
    user_df_encoded = combined_df_encoded[original_rows:]


    user_pred, user_pis = reg_xg.predict(user_df_encoded, alpha = alpha)
    st.subheader("Predicting Traffic Volume...")
    st.metric("**Predicted Traffic Volume:** ", value = f"{int(user_pred[0].round())}")
    st.write(f"**Prediction Interval** ({(1-alpha)*100:.0f}%): [{int(user_pis[:,0,0].round())}, {int(user_pis[:,1,0].round())}]")

    st.session_state.mapie = reg_xg
    st.session_state.user_df_encoded = user_df_encoded
    st.session_state.predicted = True

# Updating prediciton intervals based on alpha 
if "prev_alpha" not in st.session_state:
    st.session_state.prev_alpha = alpha

# Detect change
alpha_changed = alpha != st.session_state.prev_alpha
st.session_state.prev_alpha = alpha  # update for next rerun

# Now you can conditionally run logic
if alpha_changed and st.session_state.predicted:
    user_pred, user_pis = st.session_state.mapie.predict(
        st.session_state.user_df_encoded, alpha=alpha
    )
    st.subheader("Predicting Traffic Volume...")
    st.metric("**Predicted Traffic Volume:** ", value = f"{int(user_pred[0].round())}")
    st.write(f"**Prediction Interval** ({(1-alpha)*100:.0f}%): [{int(user_pis[:,0,0].round())}, {int(user_pis[:,1,0].round())}]")


elif (user_data is None) and (not predict_button): 
    st.subheader("Predicting Traffic Volume...")
    st.metric("**Predicted Traffic Volume:** ", value = 0)
    st.write(f"**Prediction Interval** ({(1-alpha)*100:.0f}%): [0, 664]")


# Showing additional items in tabs
st.subheader("Model Insights")
tab1, tab2, tab3, tab4 = st.tabs([ "Feature Importance", "Histogram of Residuals", "Predicted vs. Actual", "Coverage Plot"])

    # Tab 1: Feature Importance Visualization
with tab1:
    st.write("### Feature Importance")
    st.image('xg_feature_imp.svg')
    st.caption("Relative importance of features in prediction")

# Tab 2: Confusion Matrix
with tab2:
    st.write("### Histogram of Residuals")
    st.image('hist_residuals.svg')
    st.caption("Distribution of residuals to evaluate prediction quality")

# Tab 3: Predicted vs. Actual
with tab3:
    st.write("### Predicted vs. Actual")
    st.image('pred_actual.svg')
    st.caption("Visual comparision of precicted and actual values.")
# Tab 4: Coverage Plot
with tab4:
    st.write("### Coverage Plot")
    st.image('coverage.svg')
    st.caption("Range of predictions with confidence intervals")