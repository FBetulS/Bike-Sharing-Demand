import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from datetime import datetime, timedelta
import pickle
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Bike Sharing Demand Prediction",
    page_icon="🚲",
    layout="wide"
)

# Title and introduction
st.title("🚲 Bike Sharing Demand Prediction")
st.markdown("""
This application predicts the number of bike rentals based on weather conditions and time information.
Upload your trained model or use the provided example to make predictions.
""")

# Sidebar
st.sidebar.header("Options")
page = st.sidebar.selectbox("Choose a page", ["Home", "Prediction", "About"])

# Load example data
@st.cache_data
def load_example_data():
    # This is just a sample - in real app, you would use your actual data
    try:
        train = pd.read_csv('train.csv')
        if 'datetime' in train.columns:
            train['datetime'] = pd.to_datetime(train['datetime'])
        return train
    except:
        # Create sample data if file doesn't exist
        dates = pd.date_range(start='2011-01-01', end='2012-12-31', freq='H')
        np.random.seed(42)
        df = pd.DataFrame({
            'datetime': dates,
            'season': np.random.choice([1, 2, 3, 4], size=len(dates)),
            'holiday': np.random.choice([0, 1], size=len(dates), p=[0.97, 0.03]),
            'workingday': np.random.choice([0, 1], size=len(dates)),
            'weather': np.random.choice([1, 2, 3, 4], size=len(dates), p=[0.7, 0.2, 0.09, 0.01]),
            'temp': np.random.uniform(0, 1, size=len(dates)),
            'atemp': np.random.uniform(0, 1, size=len(dates)),
            'humidity': np.random.uniform(0, 1, size=len(dates)),
            'windspeed': np.random.uniform(0, 1, size=len(dates)),
            'count': np.random.randint(1, 1000, size=len(dates))
        })
        return df

# Preprocess data (similar to your code)
def preprocess_data(df):
    df = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
        df['datetime'] = pd.to_datetime(df['datetime'])
        
    # Extract datetime features
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['weekday'] = df['datetime'].dt.weekday
    
    # Additional features
    df['is_business_day'] = ((df['weekday'] < 5) & (df['holiday'] == 0)).astype(int)
    
    # Season features
    df['season_spring'] = (df['season'] == 1).astype(int)
    df['season_summer'] = (df['season'] == 2).astype(int)
    df['season_fall'] = (df['season'] == 3).astype(int)
    df['season_winter'] = (df['season'] == 4).astype(int)
    
    # Weather features
    df['weather_clear'] = (df['weather'] == 1).astype(int)
    df['weather_mist'] = (df['weather'] == 2).astype(int)
    df['weather_light_rain_snow'] = (df['weather'] == 3).astype(int)
    df['weather_heavy_rain_snow'] = (df['weather'] == 4).astype(int)
    
    # Create numeric time of day instead of categorical
    df['night_hours'] = ((df['hour'] >= 0) & (df['hour'] < 6)).astype(int)
    df['morning_hours'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
    df['afternoon_hours'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
    df['evening_hours'] = ((df['hour'] >= 18) & (df['hour'] < 24)).astype(int)
    
    # Temperature ratio
    df['temp_ratio'] = df['temp'] / df['atemp']
    
    # Season-hour interaction
    df['season_hour'] = df['season'].astype(str) + "_" + df['hour'].astype(str)
    
    # Temperature-humidity interaction
    df['temp_humidity'] = df['temp'] * df['humidity']
    
    return df

# Function to create download link for plots
def get_image_download_link(fig, filename, text):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'

# Home page
if page == "Home":
    st.header("Welcome to the Bike Sharing Demand Predictor")
    
    st.markdown("""
    ### Project Overview
    This project aims to predict the number of bike rentals based on weather conditions and time information.
    
    ### Features
    - **Exploratory Data Analysis (EDA)**: Visualize patterns and trends in the data
    - **Prediction**: Make real-time predictions using your trained model
    - **Model Performance**: Evaluate and compare different models
    
    ### How to Use
    1. Navigate to the **EDA** page to explore data patterns
    2. Visit the **Prediction** page to make predictions with your model
    3. Check the **About** page for more information about this project
    """)
    
    # Fixed deprecated parameter
    st.image("https://images.unsplash.com/photo-1556316384-12c35d30afa4?ixlib=rb-1.2.1&auto=format&fit=crop&w=1200&q=80", 
             caption="Bike Sharing System", use_container_width=True)


# Prediction page
elif page == "Prediction":
    st.header("Make Predictions")
    
    # Upload model or use example model
    st.subheader("Step 1: Upload or Use Example Model")
    
    model_option = st.radio("Select model option", ["Use example model", "Upload your model"])
    
    model = None
    if model_option == "Upload your model":
        model_file = st.file_uploader("Upload your trained model (.pkl or .txt)", type=["pkl", "txt"])
        if model_file:
            try:
                model = pickle.load(model_file)
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {e}")
    else:
        # Create a simple example model
        st.info("Using example LightGBM model (this is a placeholder and won't give accurate predictions)")
        try:
            # Create a very basic model
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
            # Just a dummy model that predicts based on hour and weather
            X = np.array([[h, w] for h in range(24) for w in range(1, 5)])
            y = np.array([
                max(10, 100 * np.sin(h * np.pi/12) + 50 * (5-w) + np.random.normal(0, 10)) 
                for h in range(24) for w in range(1, 5)
            ])
            model.fit(X, y)
            st.success("Example model loaded!")
        except Exception as e:
            st.error(f"Error creating example model: {e}")
    
    # Input parameters for prediction
    st.subheader("Step 2: Enter Parameters for Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        prediction_date = st.date_input("Select date", datetime.now())
        prediction_time = st.time_input("Select time", datetime.now().time())
        season = st.selectbox("Season", [
            (1, "Spring"), 
            (2, "Summer"), 
            (3, "Fall"), 
            (4, "Winter")
        ], format_func=lambda x: x[1])
        holiday = st.checkbox("Is holiday?")
        workingday = st.checkbox("Is working day?")
        
    with col2:
        weather = st.selectbox("Weather", [
            (1, "Clear/Few clouds"), 
            (2, "Mist/Cloudy"), 
            (3, "Light Snow/Rain"), 
            (4, "Heavy Rain/Snow/Fog")
        ], format_func=lambda x: x[1])
        
        temp = st.slider("Temperature (normalized 0-1)", 0.0, 1.0, 0.5, 0.01)
        atemp = st.slider("Feeling Temperature (normalized 0-1)", 0.0, 1.0, 0.5, 0.01)
        humidity = st.slider("Humidity (normalized 0-1)", 0.0, 1.0, 0.5, 0.01)
        windspeed = st.slider("Wind Speed (normalized 0-1)", 0.0, 1.0, 0.2, 0.01)
    
    # Create a DataFrame for the input
    combined_dt = datetime.combine(prediction_date, prediction_time)
    input_data = pd.DataFrame({
        'datetime': [combined_dt],
        'season': [season[0]],
        'holiday': [int(holiday)],
        'workingday': [int(workingday)],
        'weather': [weather[0]],
        'temp': [temp],
        'atemp': [atemp],
        'humidity': [humidity],
        'windspeed': [windspeed]
    })
    
    # Preprocess the input data
    input_processed = preprocess_data(input_data)
    
    # Prepare for prediction
    if model is not None and st.button("Predict"):
        try:
            # Display input summary
            st.subheader("Input Summary")
            st.write(f"Date and Time: {combined_dt}")
            st.write(f"Season: {season[1]}")
            st.write(f"Weather: {weather[1]}")
            st.write(f"Temperature: {temp:.2f} (normalized)")
            
            # Remove target columns and non-numeric columns
            for col in ['datetime', 'casual', 'registered', 'count', 'count_log', 'season_hour', 'time_of_day']:
                if col in input_processed.columns:
                    input_processed = input_processed.drop(col, axis=1)
            
            # Create dummy variables for categorical features
            categorical_cols = ['season', 'holiday', 'workingday', 'weather', 'year', 'month', 'day', 'hour', 'weekday']
            prediction_df = pd.get_dummies(input_processed, columns=[col for col in categorical_cols if col in input_processed.columns])
            
            # Check if model is LightGBM or sklearn based
            if hasattr(model, 'predict'):
                # For sklearn models, we need to handle the feature set carefully
                try:
                    prediction = model.predict(prediction_df)
                except Exception as e:
                    st.warning(f"Feature mismatch error: {e}")
                    st.info("Trying with simplified feature set...")
                    
                    # Simplify to just the core features that the example model expects
                    hour = input_processed['hour'].values[0]
                    weather_val = input_processed['weather'].values[0]
                    prediction = model.predict([[hour, weather_val]])
                
                if len(prediction) > 0:
                    # Check if log transform was used
                    if np.max(prediction) < 10:  # Assuming logged values will be small
                        st.write("Detected log-transformed prediction, converting back...")
                        prediction = np.expm1(prediction)
                    
                    # Round and ensure non-negative
                    prediction = max(0, round(prediction[0]))
                    
                    # Display prediction
                    st.subheader("Prediction Result")
                    st.markdown(f"<div style='text-align: center;'><h1>{prediction}</h1> bikes expected to be rented</div>", unsafe_allow_html=True)
                    
                    # Show confidence interval (dummy for example)
                    lower_bound = max(0, int(prediction * 0.8))
                    upper_bound = int(prediction * 1.2)
                    st.write(f"Approximate range: {lower_bound} - {upper_bound} bikes")
                    
                else:
                    st.error("Prediction returned empty result")
            elif hasattr(model, 'predict_proba'):
                st.error("Model appears to be a classifier, not a regressor. Please use a regression model.")
            else:
                st.error("Model doesn't have a standard prediction method")
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("This could be due to a mismatch between the model's expected features and the features provided. Try uploading your trained model file for better results.")

# About page
elif page == "About":
    st.header("About This Project")
    
    st.markdown("""
    ### Bike Sharing Demand Prediction
    
    This project aims to predict the number of bike rentals based on historical data, weather conditions, and calendar information. The prediction model was trained on the Kaggle Bike Sharing Demand competition dataset.
    
    ### Features Used in Prediction
    - **Temporal features**: Hour, day, month, year, weekday
    - **Weather conditions**: Temperature, humidity, wind speed
    - **Calendar features**: Holiday, working day
    
    ### Models Implemented
    - Random Forest Regressor
    - Gradient Boosting Regressor
    - LightGBM
    - XGBoost
    
    ### Model Performance
    The LightGBM model achieved the best performance with an RMSLE score of 0.38706 on the Kaggle leaderboard.
    
    ### Preprocessing Steps
    - Feature engineering from datetime
    - Log transformation of target variable
    - Outlier removal (values over 1000)
    - Feature interactions (temp × humidity)
    - One-hot encoding of categorical variables
    
    ### How to Deploy This App
    To deploy this app on Hugging Face Spaces:
    
    1. Create a new Space on Hugging Face
    2. Select Streamlit as the SDK
    3. Upload this app.py file along with your trained model
    4. Add requirements.txt with the necessary dependencies
    5. Push changes to your repository
    
    ### Dependencies
    - streamlit==1.26.0
    - pandas==2.0.3
    - numpy==1.24.4
    - matplotlib==3.7.2
    - seaborn==0.12.2
    - scikit-learn==1.3.0
    - lightgbm==3.3.5
    - xgboost==1.7.6
    
    ### Credits
    Built with ❤️ using Streamlit
    """)
    
    st.markdown("### Repository Structure")
    st.code("""
    ├── app.py                    # Streamlit application
    ├── model.pkl                 # Trained model (optional)
    ├── requirements.txt          # Dependencies
    ├── .gitignore                # Git ignore file
    └── README.md                 # Project documentation
    """)

# Footer
st.markdown("---")
st.markdown("Bike Sharing Demand Prediction | Built with Streamlit")