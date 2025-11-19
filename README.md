
Telco Customer Churn Prediction

This project builds a machine learning pipeline to predict customer churn for a telecom company using the **Telco Customer Churn** dataset (from Kaggle or IBM Watson Studio).

The goal is to identify customers that are likely to leave so that the business can proactively take retention actions (discounts, targeted outreach, service improvements, etc.).


1. Problem Description

Customer churn — when a customer stops using a service — directly affects revenue and growth. Telecom companies collect rich customer data: demographics, contract details, services used, and billing information.  
Using this historical data, we can train a classification model that estimates the probability that a given customer will churn in the next period.

**Business value:**

- Prioritize at-risk customers for retention campaigns  
- Optimize marketing spend by targeting customers with high churn probability  
- Understand which factors drive churn (e.g., contract type, tenure, monthly charges)


 2. Dataset

I use the **Telco Customer Churn** dataset available on:

- Kaggle: *Telco Customer Churn*  
- IBM Watson Studio: *Telco Customer Churn* sample

Expected file name (place it in the "data" folder):

"WA_Fn-UseC_-Telco-Customer-Churn.csv"

3.Project Structure


telco_churn_project:
├── Dockerfile
├── README.md
├── requirements.txt
├── train.py
├── predict.py
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   
│   └── README.md
├── models/
│   └── churn_model.pkl                       # created after running train.py
├── notebooks/
│   └── notebook.ipynb                        # EDA, feature analysis, model selection
└── images/
    └── demo_screenshot.png                   

4. Environment Setup
4.1. Using pip and virtualenv (recommended)

bash
git clone <url>
cd telco_churn_project

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

pip install -r requirements.txt

4.2. Data

Download the dataset and place it in the "data" folder:

5. Notebook: EDA, Feature Importance, Model Selection

Open the notebook:

bash
jupyter notebook notebooks/notebook.ipynb

The notebook contains:

1. Data loading, cleaning, and preparation  
2. Exploratory Data Analysis (EDA):  
   - Target distribution (Churn)
   - Relationship between features and churn
3. Feature importance analysis using tree-based models  
4. Training and comparison of multiple models:  
   - Logistic Regression (classification)
   - Decision Tree / Random Forest (tree-based models)
5. Final model selection based on AUC and export

6. Training the Final Model
Run the training script:

bash
python train.py

What it does:

- Loads and cleans the Telco dataset  
- Encodes categorical features using "OneHotEncoder"  
- Splits data into train/validation sets  
- Trains multiple models (Logistic Regression, Random Forest)  
- Evaluates them using ROC AUC  
- Selects the best model and saves it to "models/churn_model.pkl"

7. Serving Predictions via a Web Service

Run the prediction service (Flask app):

bash
python predict.py

8. Running with Docker (Local Deployment)

Build the Docker image:

bash
docker build -t telco-churn-service .

Run the container:

bash
docker run -it --rm -p 9696:9696 telco-churn-service

Now we can send the same "curl" request to `http://localhost:9696/predict`.

This simulates a production-like deployment scenario, where the model is packaged and served as a standalone service.


9. Files Overview

- notebooks/notebook.ipynb`  
  Full exploration, EDA, feature importance, and model selection.

- train.py`  
  Script to train and evaluate models, and save the best model to disk.

- predict.py  
  Flask web service that loads the saved model and exposes a "predict" endpoint.

- requirements.txt  
  Python dependencies for the project.

- Dockerfile  
  Docker configuration for containerized deployment.

- images/demo_screenshot.png  
    image illustrating interaction with the deployed service(DOCKER HUB, GCP-RUNNER).
    testing API locally
    command prompt to deploy to docker hub
    command prompt to test public URL in windows power shell
-Prompts : prompts used on anaconda prompt and windows powershell
    

10. Cloud Deployment (GCP Cloud Run) 
    The model is deployed as a serverless container on Google Cloud Run.
1. Build + Push Docker Image to Docker Hub
   "docker tag telco-churn-service:latest rinimondal7/telco-churn-service:latest
docker push rinimondal7/telco-churn-service:latest"

Public image:docker.io/rinimondal7/telco-churn-service:latest

✔ 2. Deploy Using Google Cloud Run

Inside the GCP Console:

Go to Cloud Run

Click Create Service

Select Deploy container image

Enter:docker.io/rinimondal7/telco-churn-service:latest

Set container port 9696 (or use $PORT in predict.py)

Allow Unauthenticated requests

Deploy

public URL :

https://telco-churn-service-1075172886844.us-central1.run.app

11. Prediction Request Example (PowerShell)
Windows powershel command: 
$body = '{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 5,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 80.0,
  "TotalCharges": 400.0
}'
the response i got : 
$response = Invoke-WebRequest `
  -Uri "https://telco-churn-service-1075172886844.us-central1.run.app/predict" `
  -Method POST `
  -Headers @{ "Content-Type" = "application/json" } `
  -Body $body

$response.Content
{"churn_prediction":[1],"churn_probability":[0.7839187494857017]}

🎉 Conclusion

This project demonstrates a complete end-to-end ML pipeline:

Data processing and EDA

Model training and selection

Model serialization

Serving predictions via Flask

Docker containerization

Cloud deployment using GCP Cloud Run

Real-time inference through a public API

It showcases how machine learning models can be deployed as scalable microservices suitable for production environments.