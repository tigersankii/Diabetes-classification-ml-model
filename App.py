import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🏥",
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
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .diabetic {
        background-color: #ffcccc;
        border: 2px solid #ff4444;
    }
    .non-diabetic {
        background-color: #ccffcc;
        border: 2px solid #44ff44;
    }
    .feature-importance {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        with open('best_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file 'best_model.pkl' not found. Please train a model first.")
        return None

def train_new_model():
    # Load and prepare data
    data = pd.read_csv('diabetes.csv')
    
    # Split features and target
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    with open('best_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    
    return model, X_test, y_test

def main():
    st.markdown('<h1 class="main-header">🏥 Diabetes Prediction App</h1>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose App Mode", 
                                  ["Single Prediction", "Batch Prediction", "Model Training & Analysis"])
    
    # Load or train model
    model = load_model()
    
    if app_mode == "Single Prediction":
        single_prediction_mode(model)
    elif app_mode == "Batch Prediction":
        batch_prediction_mode(model)
    else:
        model_training_mode()

def single_prediction_mode(model):
    st.header("🔍 Single Patient Prediction")
    st.write("Enter the patient's medical details below to predict diabetes risk.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pregnancies = st.slider("Pregnancies", 0, 17, 1)
        glucose = st.slider("Glucose Level", 0, 200, 120)
        blood_pressure = st.slider("Blood Pressure", 0, 122, 70)
        skin_thickness = st.slider("Skin Thickness", 0, 99, 20)
    
    with col2:
        insulin = st.slider("Insulin Level", 0, 846, 79)
        bmi = st.slider("BMI", 0.0, 67.1, 25.0)
        diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
        age = st.slider("Age", 21, 81, 33)
    
    with col3:
        st.subheader("Feature Ranges (from dataset)")
        st.write(f"**Glucose**: 0-199 mg/dL")
        st.write(f"**Blood Pressure**: 0-122 mmHg")
        st.write(f"**BMI**: 0-67.1 kg/m²")
        st.write(f"**Age**: 21-81 years")
    
    # Create feature dictionary
    features = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree,
        'Age': age
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([features])
    
    # Prediction section
    st.markdown("---")
    col_pred1, col_pred2 = st.columns([2, 1])
    
    with col_pred1:
        if st.button("🎯 Predict Diabetes Risk", use_container_width=True):
            if model is not None:
                try:
                    # Make prediction
                    prediction = model.predict(input_df)[0]
                    probability = model.predict_proba(input_df)[0]
                    
                    # Display results
                    if prediction == 1:
                        st.markdown(
                            f'<div class="prediction-box diabetic">'
                            f'<h2>⚠️ HIGH RISK OF DIABETES</h2>'
                            f'<p>Probability: {probability[1]:.2%}</p>'
                            f'</div>', 
                            unsafe_allow_html=True
                        )
                        st.warning("This patient shows high risk factors for diabetes. Consider further medical evaluation.")
                    else:
                        st.markdown(
                            f'<div class="prediction-box non-diabetic">'
                            f'<h2>✅ LOW RISK OF DIABETES</h2>'
                            f'<p>Probability: {probability[0]:.2%}</p>'
                            f'</div>', 
                            unsafe_allow_html=True
                        )
                        st.success("This patient shows low risk factors for diabetes.")
                    
                    # Show probability breakdown - FIXED SYNTAX ERROR
                    st.subheader("Risk Probability Breakdown")
                    prob_data = {
                        'Risk Level': ['Low Risk (No Diabetes)', 'High Risk (Diabetes)'],
                        'Probability': [probability[0], probability[1]]
                    }
                    prob_df = pd.DataFrame(prob_data)
                    st.dataframe(prob_df.style.format({'Probability': '{:.2%}'}))
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
            else:
                st.error("No trained model available. Please train a model first.")
    
    with col_pred2:
        st.subheader("Input Summary")
        st.dataframe(input_df.T.rename(columns={0: 'Value'}))

def batch_prediction_mode(model):
    st.header("📊 Batch Prediction")
    st.write("Upload a CSV file with patient data for multiple predictions.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            batch_data = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(batch_data.head())
            
            # Check if required columns are present
            required_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                              'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            
            if all(col in batch_data.columns for col in required_columns):
                if st.button("🔍 Predict for All Patients", use_container_width=True):
                    if model is not None:
                        # Make predictions
                        predictions = model.predict(batch_data[required_columns])
                        probabilities = model.predict_proba(batch_data[required_columns])
                        
                        # Add predictions to dataframe
                        batch_data['Prediction'] = predictions
                        batch_data['Diabetes_Probability'] = probabilities[:, 1]
                        batch_data['Risk_Level'] = batch_data['Prediction'].map({0: 'Low Risk', 1: 'High Risk'})
                        
                        # Display results
                        st.subheader("Prediction Results")
                        
                        # Summary statistics
                        high_risk_count = (predictions == 1).sum()
                        low_risk_count = (predictions == 0).sum()
                        total_count = len(predictions)
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Patients", total_count)
                        col2.metric("High Risk Patients", high_risk_count)
                        col3.metric("Low Risk Patients", low_risk_count)
                        
                        # Display results with styling
                        def color_risk(val):
                            color = 'red' if val == 'High Risk' else 'green'
                            return f'color: {color}; font-weight: bold'
                        
                        display_df = batch_data[required_columns + ['Risk_Level', 'Diabetes_Probability']].copy()
                        st.dataframe(
                            display_df.style.format({'Diabetes_Probability': '{:.2%}'})\
                            .applymap(color_risk, subset=['Risk_Level'])
                        )
                        
                        # Download results
                        csv = batch_data.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions",
                            data=csv,
                            file_name="diabetes_predictions.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("No trained model available. Please train a model first.")
            else:
                st.error(f"CSV file must contain these columns: {', '.join(required_columns)}")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def model_training_mode():
    st.header("🤖 Model Training & Analysis")
    
    # Load data
    data = pd.read_csv('diabetes.csv')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Dataset Overview")
        st.write(f"**Total Records:** {len(data)}")
        st.write(f"**Diabetic Patients:** {data['Outcome'].sum()} ({data['Outcome'].mean():.1%})")
        st.write(f"**Non-Diabetic Patients:** {len(data) - data['Outcome'].sum()} ({1 - data['Outcome'].mean():.1%})")
        
        # Show data preview
        with st.expander("View Dataset Sample"):
            st.dataframe(data.head(10))
        
        # Show basic statistics
        with st.expander("View Dataset Statistics"):
            st.dataframe(data.describe())
    
    with col2:
        st.subheader("Data Quality")
        # Check for missing values
        missing_values = data.isnull().sum()
        st.write("Missing Values:")
        for col, missing in missing_values.items():
            st.write(f"- {col}: {missing} ({missing/len(data):.1%})")
    
    # Train new model
    st.markdown("---")
    st.subheader("Model Training")
    
    if st.button("🔄 Train New Model", use_container_width=True):
        with st.spinner("Training model... This may take a few seconds."):
            model, X_test, y_test = train_new_model()
            st.success("Model trained successfully!")
            
            # Model evaluation
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            st.metric("Model Accuracy", f"{accuracy:.2%}")
            
            # Feature importance
            st.subheader("Feature Importance")
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(feature_importance['feature'], feature_importance['importance'])
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance in Diabetes Prediction')
            st.pyplot(fig)
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Non-Diabetic', 'Diabetic'],
                       yticklabels=['Non-Diabetic', 'Diabetic'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
    
    # Show dataset distribution
    st.markdown("---")
    st.subheader("Data Distribution")
    
    col_dist1, col_dist2 = st.columns(2)
    
    with col_dist1:
        # Outcome distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        data['Outcome'].value_counts().plot(kind='bar', ax=ax, color=['lightblue', 'salmon'])
        ax.set_title('Diabetes Outcome Distribution')
        ax.set_xlabel('Outcome (0: Non-Diabetic, 1: Diabetic)')
        ax.set_ylabel('Count')
        st.pyplot(fig)
    
    with col_dist2:
        # Glucose distribution by outcome
        fig, ax = plt.subplots(figsize=(8, 6))
        for outcome in [0, 1]:
            subset = data[data['Outcome'] == outcome]
            ax.hist(subset['Glucose'], alpha=0.7, label=f'Outcome {outcome}', bins=20)
        ax.set_title('Glucose Distribution by Outcome')
        ax.set_xlabel('Glucose Level')
        ax.set_ylabel('Frequency')
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
