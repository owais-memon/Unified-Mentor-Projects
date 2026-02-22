# Unified-Mentor-Projects
UNIFIED MENTOR – DATA SCIENCE PROJECT PORTFOLIO  This repository contains four end-to-end Machine Learning and Data Analysis projects developed as part of the Unified Mentor Program.  Projects Included:  Climate Change Modeling, Personalized Healthcare Recommendation System, Life Expectancy Analysis, Inventory Data Analysis &amp; Optimization

Each project follows a structured ML pipeline:
1) Problem Definition
2) Data Cleaning & Feature Engineering
3) Model Development
4) Evaluation & Interpretation
5) Insights & Business Impact

Project 1 : NASA-Sentiment-Analysis/
│
├── nasa_comments.csv
├── advanced_nasa_sentiment_analysis.csv
├── sentiment_analysis.ipynb
├── README.md

Project 2 : advanced-healthcare-recommendation/
│
├── 📁 data/
│   ├── raw/
│   │   └── blood_data.csv
│   ├── processed/
│   │   └── cleaned_data.csv
│   └── external/
│
├── 📁 notebooks/
│   └── healthcare_ai_notebook.ipynb
│
├── 📁 src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── rule_engine.py
│   ├── drift_detection.py
│   ├── logging_utils.py
│   └── prediction_pipeline.py
│
├── 📁 artifacts/
│   ├── trained_model.pkl
│   ├── preprocessor.pkl
│   ├── meta.json
│   └── prediction_logs.json
│
├── 📁 reports/
│   ├── model_evaluation_metrics.png
│   ├── shap_explanations.png
│   └── final_report.pdf
│
├── 📁 api/
│   └── fastapi_app.py
│
├── requirements.txt
├── README.md
└── LICENSE

Project 3: life_expectancy_project/
│
├── Life_Expectancy_Data.csv
├── dashboard.py
├── outputs/
│   ├── cleaned_data.csv
│   ├── feature_importance.csv
│
├── README.md
└── main_analysis.py

Project 4:
inventory_optimization_project/
│
├── product_data.csv
├── final_inventory_analysis.csv
├── purchase_orders.csv
├── inventory_system.db
├── dashboard.html
├── main_script.py
└── README.md

**NOTE: PROJECT 4 HAS 2 VERSIONS ONE CREATING A DETAILED REPORT WITH DIFFERENT FILES AND THE SECOND VERSION CREATED A DASHBOARD**

Project 1: NASA Social Media Sentiment Analysis
Advanced NLP & Machine Learning Project

 Project Overview:This project performs advanced Natural Language Processing (NLP) and Machine Learning analysis on NASA social media comments. The goal is to analyze public sentiment, extract discussion topics, and correlate engagement metrics (likes and comments) with sentiment trends.

The project combines:
Rule-based sentiment analysis (VADER)
Machine Learning classification (TF-IDF + Logistic Regression)
Topic modeling (LDA)
Time-series sentiment trends
Engagement analysis using likes and comment counts

This project demonstrates real-world data science workflow from preprocessing to model evaluation and insight extraction.

Dataset Description

The dataset contains the following columns:
date – Date of the comment
likesCount – Number of likes received
profileName – User name
commentsCount – Number of replies
text – Actual comment text

Project Objectives:
Perform advanced text preprocessing and cleaning
Conduct rule-based sentiment analysis using VADER
Train a supervised ML sentiment classifier
Extract hidden discussion themes using LDA topic modeling
Analyze sentiment trends over time
Study engagement metrics (likes/comments) vs sentiment

⚙️ Technologies Used
Python
Pandas
NumPy
NLTK
Scikit-learn
Matplotlib
Seaborn
WordCloud

🔬 Project Workflow
1️⃣ Data Cleaning
Lowercasing
Removing URLs, mentions, hashtags
Stopword removal
Lemmatization

2️⃣ Sentiment Analysis
VADER sentiment scoring
Classification into:
Positive - Neutral - Negative

3️⃣ Machine Learning Model
TF-IDF vectorization
Logistic Regression classifier
Train-test split
Model evaluation (Accuracy, Precision, Recall, F1-score)

4️⃣ Topic Modeling
CountVectorizer
Latent Dirichlet Allocation (LDA)
Extraction of top keywords per topic

5️⃣ Engagement Analysis
Sentiment vs Likes
Sentiment vs Comment volume

6️⃣ Time Series Analysis
Monthly sentiment trend visualization

📊 Model Performance
The Logistic Regression model is trained on TF-IDF features and evaluated using:
Accuracy Score, Confusion Matrix, Classification Report

📈 Key Insights
Public sentiment towards NASA is predominantly positive.
High-engagement comments often correlate with emotionally strong sentiment.
Topic modeling reveals recurring themes around missions, space exploration, and scientific achievements.
Sentiment trends vary across months depending on major events.

📷 Visualizations Included

Sentiment Distribution Plot
WordCloud of frequent terms:
Confusion Matrix
Topic Modeling output
Sentiment trend over time
Engagement comparison charts



Project 2 :Advanced Personalized Healthcare Recommendation System
Overview
This project implements an Explainable AI–based Personalized Healthcare Recommendation Engine that analyzes patient-level clinical data (demographics, vitals, lab parameters) to generate actionable health recommendations.
The system combines:
🔹 Calibrated XGBoost classification
🔹 Clinical rule-based reasoning
🔹 Confidence-based safety gating
🔹 Drift detection monitoring
🔹 Prediction logging for auditability
It is designed to simulate a production-grade, ethically aware healthcare AI pipeline.

 Key Features
Advanced Modeling
XGBoost classifier with probability calibration (Isotonic Regression)
Structured preprocessing pipeline (imputation, scaling, encoding)
Multi-class health risk stratification

Clinical Recommendation Engine
Model-based risk prediction
Integrated rule-based clinical safety checks
Actionable recommendations (monitoring, lifestyle modification, specialist referral)

Explainability & Safety
Confidence threshold gating for uncertain predictions
Drift detection against training distribution
Transparent decision logic via structured outputs

Engineering & Monitoring
Modular pipeline architecture
Prediction logging for traceability
Designed for extension to SHAP explainability and fairness auditing

Technologies Used
Python, Pandas, NumPy, Scikit-learn, XGBoost, Probability Calibration, Modular ML Pipelines

Project Objective
To build an internship-level, production-inspired healthcare AI system that demonstrates:
Applied machine learning in clinical settings
Responsible AI design principles
Safety-aware recommendation generation
Scalable pipeline architecture


Project 3:Life Expectancy Analysis (WHO + UN Data, 2000–2015)

Project Overview
This project analyzes global life expectancy trends using WHO health indicators and UN economic data for ~193 countries from 2000 to 2015.

The objective is to:
Identify key drivers of life expectancy
Evaluate economic, healthcare, and social determinants
Build interpretable statistical models
Develop high-performance machine learning predictors
Generate policy-oriented insights
Deploy an interactive dashboard for scenario analysis
This project combines econometrics, machine learning, and policy analytics — designed at an internship-ready level.

Research Questions
How strongly does GDP influence life expectancy?
Do immunization programs significantly improve outcomes?
Is schooling a stronger predictor than healthcare expenditure?
Are determinants different for developed vs developing countries?
What policy levers matter most for low-life-expectancy nations?

Dataset Source:
WHO Global Health Observatory
United Nations Economic Data
Time Period: 2000–2015
Countries: ~193
Observations: Panel data (country-year)

Key Variables:
Life Expectancy (target)
Adult Mortality
GDP
Total Health Expenditure
Schooling
Immunization (HepB, Polio, Diphtheria)
HIV/AIDS
BMI
Income Composition
Population

Technical Implementation
1️⃣ Data Preprocessing
Missing value analysis
Iterative Imputation (MICE)
Removal of highly incomplete countries
Log transformations (GDP, Population)
Feature scaling where required

2️⃣ Advanced Feature Engineering
Immunization Composite Index
GDP Growth Rate
Lag Features (GDP_lag1, HealthExp_lag1, Schooling_lag1)
Interaction Terms
Segmentation variables (Developed vs Developing)

3️⃣ Statistical Modeling
Multiple Linear Regression (OLS)
Coefficient interpretation
p-values and statistical inference
Model Diagnostics
Variance Inflation Factor (VIF)
Breusch-Pagan test for heteroskedasticity
Residual analysis

4️⃣ Machine Learning Models
Random Forest Regressor (with hyperparameter tuning)
XGBoost Regressor
TimeSeriesSplit cross-validation (to respect panel time structure)

Model Evaluation Metrics: R² Score, RMSE, Cross-validated performance

5️⃣ Segmentation Analysis
Separate models for:
Developed countries
Developing countries
Findings show structural differences in determinants.

6️⃣ Model Interpretability
Feature Importance (Tree-based models)
Policy driver ranking
Comparative impact analysis

7️⃣ Deployment
An interactive Streamlit dashboard allows:
Country selection
Trend visualization
Policy variable simulation
Live life expectancy prediction
Run with:
streamlit run dashboard.py

📈 Key Findings
Immunization coverage strongly predicts improvements in low-income countries.
Schooling has consistent long-term impact across all regions.
GDP shows diminishing returns beyond middle-income levels.
Healthcare expenditure impacts life expectancy with lag effects.
Mortality and infectious disease indicators remain critical in developing nations.


Project 4: AI-Driven Inventory Optimization & Risk Intelligence System

An advanced data science project that transforms raw e-commerce product data into a full inventory optimization and decision-support system using machine learning, optimization, simulation, and analytics.

Project Overview
This project builds an end-to-end AI-powered inventory management system that:
Forecasts product demand using machine learning
Compares models using RMSE, MAE, and MAPE
Calculates EOQ and Reorder Points
Optimizes order quantities using numerical optimization
Performs ABC analysis and SKU clustering
Generates risk scores for products
Estimates financial impact and cost savings
Automates replenishment triggers
Stores data in SQL
Exports an interactive dashboard
The system simulates real-world inventory decision-making for a manufacturing or e-commerce company.

Dataset
The dataset includes product-level information:
product_id
product_name
category
discounted_price
actual_price
discount_percentage
rating
rating_count
about_product
user_id
user_name
review_id
review_title
review_content
img_link
product_link
Since transactional inventory data was not available, rating_count is used as a demand proxy and enhanced through simulation.

Key Features
1️⃣ Advanced Data Cleaning
Removes currency symbols and formatting
Converts string-based numeric columns
Handles missing values safely

2️⃣ Feature Engineering
Margin calculation
Demand proxy modeling
Lead time simulation
Demand variance modeling
Annual demand estimation

3️⃣ Machine Learning Demand Forecasting
Model:
Random Forest Regressor

Model evaluation metrics: RMSE, MAE, MAPE

Uses TimeSeriesSplit cross-validation to ensure realistic forecasting performance.

4️⃣ Inventory Policy Optimization
Implements:
Economic Order Quantity (EOQ)
Safety Stock (Service-level based)
Reorder Point (ROP)
Includes:
Monte Carlo-based demand variance handling
Optimization using scipy.optimize

5️⃣ ABC Classification
Products categorized into:
A (High value, tight control)
B (Moderate control)
C (Bulk/low value items)

6️⃣ Clustering (SKU Segmentation)
Uses KMeans clustering based on:
Annual demand
Unit cost
Lead time
Creates intelligent inventory segmentation.

7️⃣ Risk Scoring Engine
Composite risk score based on:
Demand variability
Lead time
Annual value
Ranks high-risk products for management focus.

8️⃣ Financial Impact Simulation
Calculates:
Current inventory cost, Optimized cost, Total projected savings

Provides measurable business impact.

9️⃣ Automated Replenishment Engine
Triggers purchase orders when:
Current Inventory ≤ Reorder Point

Exports: purchase_orders.csv

🔟 SQL Integration
Stores processed data into: inventory_system.db

Demonstrates production-level database handling.

1️⃣1️⃣ Interactive Dashboard
Exports:
dashboard.html

Displays:
Top risk products
Risk score visualization

Tech Stack: Python, Pandas, NumPy, Scikit-learn, SciPy, Matplotlib / Seaborn, Plotly, SQLite
