# EmployeeSalaryPrediction-
Employee Salary Prediction Project
This project is an end-to-end machine learning solution that predicts whether an individual's annual income is greater than $50,000. It uses the "Adult" dataset from the UCI Machine Learning Repository, takes the data through a full analysis and modeling pipeline, and deploys the final model as an interactive web application using Streamlit.


Features
In-Depth Data Analysis: The project starts with a detailed Exploratory Data Analysis (EDA) to understand the data's characteristics and patterns.
    
    Model Comparison: Trains and evaluates three different classification models (Logistic Regression, Decision Tree, and Random Forest) to find the best performer.
    Machine Learning Pipeline: Uses scikit-learn Pipelines to create a robust and reproducible workflow for data preprocessing and modeling.
    Interactive Web Application: A user-friendly web UI built with Streamlit allows for easy interaction with the predictive model.
    Live Deployment: The application can be deployed online using ngrok for easy sharing and demonstration.

Project Workflow

    Data Analysis & Cleaning: The adult 3.csv dataset is loaded, cleaned (handling missing values and duplicates), and visualized.
    Feature Engineering: Irrelevant features are dropped, and categorical data is prepared for modeling using one-hot encoding.
    Model Training: The data is split, and three models are trained and evaluated. The best model (based on accuracy) is saved.
    Web App Development: A Streamlit script (app.py) is created to load the saved model and build the user interface.
    Deployment: The Streamlit app is launched locally and then exposed to the internet using ngrok.

Technologies and Libraries Used

    Python 3.10
    Jupyter Notebook: For data analysis and model development.
    Pandas & NumPy: For data manipulation and numerical operations.
    Matplotlib & Seaborn: For data visualization.
    Scikit-learn: For building the machine learning pipeline and models.
    Joblib: For saving the trained model.
    Streamlit: For building the interactive web application.
    Pyngrok: For deploying the application online.

Setup and Installation
To run this project on your local machine, follow these steps:

Install the required libraries:
Create a file named requirements.txt with the following content:

    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn
    joblib
    streamlit
    pyngrok

Then, install the libraries using pip:

    pip install -r requirements.txt

Usage
Run the Jupyter Notebook:
Open and run the salary_prediction_notebook.ipynb file to see the data analysis, model training process, and to generate the best_model_refined.pkl file.

Run the Streamlit Web App:
Once the .pkl file is generated, run the following command in your terminal:

    streamlit run app.py

Your web browser will open with the running application.

Model Performance
The project compared three different models. The final model was selected based on the highest accuracy score on the test set.

Model Name:

Accuracy Score

    Logistic Regression  84.88%
    Random Forest  84.47%
    Decision Tree  81.63%
The Logistic Regression model was chosen as the best performer for this task.
