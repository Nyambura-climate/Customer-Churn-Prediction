import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .churn-high { background-color: #ffebee; color: #c62828; }
    .churn-low { background-color: #e8f5e8; color: #2e7d32; }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🎯 Customer Churn Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Prediction", "Analytics", "Data Upload"])

# Load model function (with error handling)
@st.cache_resource
def load_model():
    try:
        model = joblib.load('customer_churn_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'customer_churn_model.pkl' is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Prediction function
def predict_churn(model, features):
    try:
        prediction = model.predict([features])
        probability = model.predict_proba([features])
        return prediction[0], probability[0]
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

# Main prediction page
def prediction_page():
    st.header("🔮 Make a Churn Prediction")
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Demographics")
        age = st.slider("Age", 18, 100, 35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        
    with col2:
        st.subheader("Service Information")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])

# More service options
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Additional Services")
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        
    with col4:
        st.subheader("Contract & Billing")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", 
                                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0)
    
    # Convert inputs to model format
    if st.button("🎯 Predict Churn", type="primary"):
        # Create feature array (this would need to match your model's expected input)
        features = [
            age, 1 if gender == "Male" else 0, 1 if senior_citizen == "Yes" else 0,
            1 if partner == "Yes" else 0, 1 if dependents == "Yes" else 0,
            tenure, 1 if phone_service == "Yes" else 0,
            1 if multiple_lines == "Yes" else 0, 1 if internet_service == "Fiber optic" else 0,
            1 if online_security == "Yes" else 0, 1 if online_backup == "Yes" else 0,
            1 if device_protection == "Yes" else 0, 1 if tech_support == "Yes" else 0,
            1 if streaming_tv == "Yes" else 0, 1 if streaming_movies == "Yes" else 0,
            1 if contract == "Month-to-month" else 0, 1 if paperless_billing == "Yes" else 0,
            1 if payment_method == "Electronic check" else 0, monthly_charges, total_charges
        ]
        
        prediction, probability = predict_churn(model, features)
        
        if prediction is not None:
            churn_prob = probability[1] * 100
            
            if prediction == 1:
                st.markdown(f'<div class="prediction-result churn-high">⚠️ HIGH CHURN RISK<br>Probability: {churn_prob:.1f}%</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="prediction-result churn-low">✅ LOW CHURN RISK<br>Probability: {churn_prob:.1f}%</div>', 
                           unsafe_allow_html=True)
            
            # Show probability breakdown
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Stay', 'Churn'],
                y=[probability[0]*100, probability[1]*100],
                marker_color=['green', 'red']
            ))
            fig.update_layout(title="Churn Probability Breakdown", yaxis_title="Probability (%)")
            st.plotly_chart(fig, use_container_width=True)

# Analytics page
def analytics_page():
    st.header("📊 Churn Analytics Dashboard")
    
    # Sample data for demonstration (replace with your actual data)
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'Age': np.random.randint(18, 80, 1000),
        'Tenure': np.random.randint(0, 72, 1000),
        'MonthlyCharges': np.random.uniform(20, 120, 1000),
        'TotalCharges': np.random.uniform(20, 8000, 1000),
        'Churn': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
    })
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        churn_rate = (sample_data['Churn'].sum() / len(sample_data)) * 100
        st.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
    
    with col2:
        avg_tenure = sample_data['Tenure'].mean()
        st.metric("Average Tenure", f"{avg_tenure:.1f} months")
    
    with col3:
        avg_monthly = sample_data['MonthlyCharges'].mean()
        st.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")
    
    with col4:
        total_customers = len(sample_data)
        st.metric("Total Customers", total_customers)
    
    # Charts
    st.subheader("📈 Churn Analysis Charts")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Age distribution by churn
        fig1 = px.histogram(sample_data, x='Age', color='Churn', 
                           title='Age Distribution by Churn Status',
                           labels={'Churn': 'Churned'})
        st.plotly_chart(fig1, use_container_width=True)
    
    with chart_col2:
        # Monthly charges vs churn
        fig2 = px.box(sample_data, x='Churn', y='MonthlyCharges', 
                      title='Monthly Charges by Churn Status',
                      labels={'Churn': 'Churned', 'MonthlyCharges': 'Monthly Charges ($)'})
        st.plotly_chart(fig2, use_container_width=True)
    
    # Tenure analysis
    fig3 = px.scatter(sample_data, x='Tenure', y='MonthlyCharges', 
                      color='Churn', title='Tenure vs Monthly Charges',
                      labels={'Churn': 'Churned'})
    st.plotly_chart(fig3, use_container_width=True)

# Data upload page
def data_upload_page():
    st.header("📤 Upload Customer Data")
    
    st.markdown("""
    Upload a CSV file with customer data to make batch predictions or analyze patterns.
    
    **Expected columns:**
    - Age, Gender, SeniorCitizen, Partner, Dependents
    - Tenure, PhoneService, MultipleLines, InternetService
    - OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport
    - StreamingTV, StreamingMovies, Contract, PaperlessBilling
    - PaymentMethod, MonthlyCharges, TotalCharges
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully uploaded! Dataset has {len(df)} rows and {len(df.columns)} columns.")
            
            # Show preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Basic statistics
            st.subheader("Basic Statistics")
            st.write(df.describe())
            
            # Option to download sample predictions
            if st.button("Generate Sample Predictions"):
                # Add sample predictions (replace with actual model predictions)
                df['Predicted_Churn'] = np.random.choice([0, 1], len(df), p=[0.7, 0.3])
                df['Churn_Probability'] = np.random.uniform(0, 1, len(df))
                
                st.subheader("Predictions Added")
                st.dataframe(df[['Predicted_Churn', 'Churn_Probability']].head())
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

# Main app logic
def main():
    # Navigation based on sidebar selection
    if page == "Prediction":
        prediction_page()
    elif page == "Analytics":
        analytics_page()
    elif page == "Data Upload":
        data_upload_page()

# Footer
def add_footer():
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p>🎯 Customer Churn Prediction Dashboard</p>
            <p>Built with Streamlit • Powered by Machine Learning</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Run the app
if __name__ == "__main__":
    main()
    add_footer()

# Footer with image and styling
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.image("Headshot.png", width=100) 
    st.markdown(
        """
        <div style='text-align: center'>
            <h4>Created by Nyambura Gachahi</h4>
            <p>Data & Climate Scientist | Machine Learning Engineer</p>
            <p>📧 gacchahi@gmail.com | 💼 LinkedIn | 🐙 GitHub</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
