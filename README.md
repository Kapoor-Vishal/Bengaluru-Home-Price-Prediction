## 🏠 Bengaluru House Price Prediction

What’s the price of a 3BHK in Indiranagar? Is that Koramangala flat worth the hype? Real estate in Bengaluru is as dynamic as the city itself — and buyers, sellers, and investors are often left guessing.

This project takes a data science approach to bring some clarity. Using a publicly available dataset and machine learning, I built a model that predicts house prices in Bengaluru, wrapped it all up in an interactive Streamlit web app, and deployed it online so anyone can test it out in real time.

## 🎯 Motivation

Property prices depend on so many variables — size, bedrooms, bathrooms, and especially location (a 2BHK in Whitefield is a very different story from a 2BHK in Jayanagar). Buyers rarely have transparent tools that combine these factors into a fair estimate.

So I decided to simulate a full data science lifecycle on this problem:

Starting from a messy Kaggle dataset.

Cleaning, transforming, and engineering features to make sense of the chaos.

Training and tuning machine learning models.

Deploying a clean, professional web app that lets anyone play with the model.

## 🛠️ Approach

Like any Bengaluru road trip, the data journey started with potholes: missing values, weird sqft ranges (2100-2850), inconsistent location names, and even bizarre entries (10 bathrooms for a 2BHK? Come on).

Steps I took:

Data Cleaning → standardizing sqft, extracting BHKs, sanity-checking bathrooms, removing impossible outliers.

Feature Engineering → price per sqft, grouping rare locations into “Other,” one-hot encoding locations.

Outlier Handling → filtering unrealistic price per sqft variations and enforcing logical rules.

Modeling → tested Linear Regression, Lasso, Decision Trees; selected Linear Regression (surprisingly robust once features were well-prepared).

Deployment → built an end-to-end Streamlit app that turns the model into an interactive experience.

## 💻 The Web Application

The fun part is the Streamlit app, which makes the project usable for anyone, not just data nerds.

## ✨ Features:

🔮 Price Prediction → Enter sqft, BHK, bathrooms, and location → get an instant estimate in Lakhs.

📊 Interactive EDA Dashboard → Explore BHK distributions, bathroom counts, price histograms, and location trends.

🏙️ Location Insights → See average prices by area, from HSR Layout to Hebbal.

🧪 Sensitivity Analysis → Tweak sqft or BHK to see how the price shifts.

📂 Batch Predictions → Upload a CSV to test multiple properties at once.

📑 Auto PDF Reports → Generate a professional report with KPIs and charts for stakeholders.

All styled with a clean dark theme for readability and deployed via Streamlit Cloud.

## 👉 Launch the App

https://bengaluru-home-price-prediction-fryrrrr5lau9eyxp25556i.streamlit.app/

## ⚙️ Tech Stack

Languages & Frameworks → Python, Streamlit

Libraries → Pandas, NumPy, Scikit-learn, Plotly, ReportLab, Kaleido

Model → Linear Regression (trained with engineered features + one-hot encoded locations)

Artifacts →

banglore_home_prices_model.pickle → trained ML model

columns.json → list of model features

bangalore_home_prices_cleaned.csv → cleaned dataset for analysis

## 🚀 Running Locally

Clone the repo & navigate inside.

Install Python 3.10+ and dependencies:

pip install -r requirements.txt


Ensure these files exist:

app.py

banglore_home_prices_model.pickle

columns.json

bangalore_home_prices_cleaned.csv (can be recreated via notebook)

Run the app:

streamlit run app.py


## 📦 Deployment

Deployment is handled via Streamlit Cloud. Just connect the repo, include requirements.txt and .streamlit/config.toml, and Streamlit spins up the app with a shareable URL.

## 📚 Key Learnings

Real-world data is messy — but with careful cleaning and feature engineering, even a simple model like Linear Regression can shine.

Building an interactive app makes ML outputs accessible and useful, moving beyond static notebooks.

Deployment, caching, theming, and PDF reporting are not just add-ons — they’re what make the project business-ready.

## 🙏 Credits

Dataset → Bengaluru House Price Data (Kaggle by Amitabh Ajoy)

Tools → Streamlit, Pandas, NumPy, Scikit-learn, Plotly, ReportLab

## 📬 Contact  

Vishal Kapoor

📧 Email: vishalkapoor9803@gmail.com  
🔗 LinkedIn: https://www.linkedin.com/in/vishal--kapoor/  
🐙 GitHub: https://github.com/Kapoor-Vishal 
