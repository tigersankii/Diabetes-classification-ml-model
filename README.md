# Diabetes Prediction App


📖 Overview
A comprehensive Streamlit web application for predicting diabetes risk based on patient medical data. This machine learning application uses the Pima Indians Diabetes Dataset to classify patients as high or low risk for diabetes.

🚀 Features
🔍 Single Prediction Mode
Interactive input form with sliders for all medical parameters

Real-time diabetes risk prediction

Color-coded results (Green for Low Risk, Red for High Risk)

Probability breakdown with percentage scores

Input validation with dataset value ranges

📊 Batch Prediction Mode
Upload CSV files for multiple patient predictions

Automatic column validation and error handling

Results summary with key metrics

Color-coded risk level display

Download predictions as CSV file

🤖 Model Training & Analysis
Train new Random Forest classifier

Model performance evaluation with accuracy scores

Feature importance visualization

Confusion matrix display

Dataset statistics and distribution analysis

🛠️ Installation
Prerequisites
Python 3.7 or higher

pip (Python package manager)

Step-by-Step Setup
Clone or Download the Project

bash
# Create project directory
mkdir diabetes-prediction-app
cd diabetes-prediction-app
Save Required Files

Save the Streamlit app code as diabetes_app.py

Save the dataset as diabetes.csv in the same directory

Save dependencies as requirements.txt

Install Dependencies

bash
pip install -r requirements.txt
Run the Application

bash
streamlit run diabetes_app.py
📋 Requirements
Create a requirements.txt file with the following content:

txt
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.1
seaborn==0.12.2
🎯 Usage Guide
Single Patient Prediction
Navigate to Single Prediction Mode

Use the sidebar to select "Single Prediction"

Adjust all 8 medical parameters using sliders

Input Parameters:

Pregnancies: Number of times pregnant (0-17)

Glucose: Plasma glucose concentration (0-199 mg/dL)

BloodPressure: Diastolic blood pressure (0-122 mmHg)

SkinThickness: Triceps skin fold thickness (0-99 mm)

Insulin: 2-Hour serum insulin (0-846 mu U/ml)

BMI: Body mass index (0.0-67.1 kg/m²)

DiabetesPedigreeFunction: Diabetes pedigree function (0.0-2.5)

Age: Age in years (21-81)

Get Results:

Click "Predict Diabetes Risk" button

View color-coded risk assessment

Check probability percentages

Batch Predictions
Prepare CSV File:

Ensure CSV has columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age

No missing values required

Upload and Process:

Select "Batch Prediction" mode

Upload CSV file

Click "Predict for All Patients"

Download results as CSV

Model Training
Access Training Mode:

Select "Model Training & Analysis"

Click "Train New Model"

View Results:

Model accuracy score

Feature importance chart

Confusion matrix

Dataset statistics

📊 Dataset Information
The application uses the Pima Indians Diabetes Dataset containing:

768 patient records

8 medical features

Binary outcome (0: Non-diabetic, 1: Diabetic)

Class distribution: ~35% diabetic, ~65% non-diabetic

Feature Descriptions:
Pregnancies: Number of times pregnant

Glucose: Plasma glucose concentration 2 hours in oral glucose tolerance test

BloodPressure: Diastolic blood pressure (mm Hg)

SkinThickness: Triceps skin fold thickness (mm)

Insulin: 2-Hour serum insulin (mu U/ml)

BMI: Body mass index (weight in kg/(height in m)^2)

DiabetesPedigreeFunction: Diabetes pedigree function (genetic risk)

Age: Age in years

🏗️ Model Architecture
Algorithm: Random Forest Classifier
Ensemble method with multiple decision trees

100 estimators (trees)

Default scikit-learn parameters

Train-test split: 80-20 ratio

Model Performance:
Typical accuracy: 75-85%

Handles non-linear relationships well

Robust to outliers and missing values

Provides feature importance scores

🎨 Interface Features
Visual Elements:
Responsive layout with multiple columns

Color-coded risk indicators

Interactive charts and graphs

Professional medical styling

Real-time validation

User Experience:
Intuitive navigation with sidebar

Clear error messages

Loading indicators for long operations

Export functionality for results

🔧 Technical Details
File Structure:
text
diabetes-prediction-app/
│
├── diabetes_app.py          # Main Streamlit application
├── diabetes.csv            # Dataset file
├── best_model.pkl         # Trained model (auto-generated)
└── requirements.txt       # Python dependencies
Key Functions:
load_model(): Loads pre-trained model from pickle file

train_new_model(): Trains new Random Forest model

single_prediction_mode(): Handles individual predictions

batch_prediction_mode(): Processes CSV file uploads

model_training_mode(): Manages model training and analysis

⚠️ Important Notes
Medical Disclaimer:
⚠️ WARNING: This application is for educational and demonstration purposes only. It should not be used for actual medical diagnosis or treatment decisions. Always consult healthcare professionals for medical advice.

Data Limitations:
Dataset represents specific population (Pima Indians)

Model performance may vary with different populations

Consider retraining with local data for better accuracy

Performance Tips:
The app automatically caches the model for faster loading

Batch processing works best with CSV files under 10MB

Model training may take 10-30 seconds depending on hardware

🐛 Troubleshooting
Common Issues:
Model file not found:

text
Error: Model file 'best_model.pkl' not found
Solution: Use "Model Training" mode to train a new model

Missing CSV columns:

text
Error: CSV file must contain specific columns
Solution: Ensure CSV has all 8 required columns with exact names

Installation errors:

text
Package not found during installation
Solution: Use pip install --upgrade pip and retry

Port already in use:

text
Error: Port 8501 is already in use
Solution: Use streamlit run diabetes_app.py --server.port 8502

📈 Future Enhancements
Potential improvements for the application:

Add more machine learning algorithms for comparison

Implement cross-validation for better model evaluation

Add data preprocessing options

Include model interpretability (SHAP values)

Add user authentication

Deploy as web service

Add multi-language support

Include API endpoints

👥 Contributing
To contribute to this project:

Fork the repository

Create a feature branch

Make your changes

Test thoroughly

Submit a pull request

📄 License
This project is intended for educational purposes. Please ensure proper attribution when using or modifying the code.

🆘 Support
For issues and questions:

Check the troubleshooting section above

Verify all dependencies are installed correctly

Ensure dataset file is in correct location

Check Streamlit documentation for interface issues
