Bengaluru House Price Prediction

This project explores the challenge of predicting residential property prices in Bengaluru, one of India’s fastest-growing real estate markets. The motivation behind this work was to use a publicly available dataset to simulate how data science and machine learning can be applied in the housing domain to bring clarity to buyers, sellers, and real estate stakeholders.

The goal was not just to build a predictive model but to take the reader through the full lifecycle of a data science project: from raw dataset exploration, through rigorous cleaning and feature engineering, to the deployment of an interactive web application that allows end-users to test the model in real time.

Background and Motivation

Property price estimation is a complex problem because it depends on a wide range of factors: size of the property, number of bedrooms and bathrooms, the location, and hidden patterns such as local demand or outlier entries. Buyers and investors often lack transparent tools that combine these variables into a consistent valuation.

The Bengaluru housing dataset from Kaggle provided a valuable playground for addressing this challenge. By cleaning and processing the data carefully, and by applying machine learning, I aimed to demonstrate how structured approaches can lead to practical insights.

Approach

The work began with the raw dataset, which contained many irregularities: missing values, inconsistent location naming, square footage ranges instead of numbers, and unrealistic entries. The first major step was data cleaning — standardizing formats, dropping irrelevant columns, and ensuring core variables (location, size, total_sqft, bath, price) were reliable.

Next, I applied feature engineering:

Bedrooms (BHK) were extracted from text entries like "3 BHK".

Total square footage was normalized, even when given as ranges (e.g., "2100-2850" was averaged).

Price per square foot was calculated to allow comparisons across properties.

Rare locations were grouped into an "other" bucket to reduce noise from sparse one-hot encoding.

With these features in place, I addressed outliers systematically: removing properties with improbably low area per BHK, filtering entries where prices per square foot were far from location means, and correcting logical inconsistencies such as multi-bedroom apartments being priced lower per square foot than smaller ones in the same location. Bathrooms were also sanity-checked (a home with 10 bathrooms and only 2 bedrooms is not realistic).

The cleaned dataset provided a solid foundation for model training. I compared multiple algorithms — Linear Regression, Lasso, and Decision Trees — using cross-validation and GridSearch to evaluate which generalized best. The final choice was a Linear Regression model, which performed well after preprocessing and produced consistent predictions.

Finally, I focused on deployment. Rather than stopping with a Jupyter notebook, I wanted to make the model usable by non-technical stakeholders. For this, I built a Streamlit web application that allows users to input their own property details (square footage, number of bedrooms, bathrooms, and location) and receive an estimated price instantly.

The Web Application

The Streamlit app is the front-end of the project and transforms the static model into an interactive experience.

It provides:

Price Prediction: Users enter details and see the predicted price in Lakhs.

Exploratory Data Analysis (EDA): Summary statistics, BHK distributions, bathroom distributions, price histograms, and location insights are displayed in a clean, dark-themed dashboard.

Price Distribution Analysis: Average prices by location are visualized to highlight the most expensive areas.

Sensitivity Analysis: Users can experiment with how prices shift when varying BHK, bathrooms, or square footage.

Batch Predictions: A CSV upload feature allows testing new batches of properties at once.

PDF Report Generation: Automatically generates downloadable PDF reports containing KPIs and charts for stakeholder sharing.

The app is styled with a consistent dark palette for professional readability and deployable via Streamlit Cloud, giving a shareable URL to anyone interested.

Technical Details

Language & Frameworks: Python, with Streamlit for deployment.

Libraries: Pandas, NumPy, Scikit-learn, Plotly, ReportLab, Kaleido.

Model: Linear Regression (trained on engineered features with one-hot encoded locations).

Artifacts:

banglore_home_prices_model.pickle: the serialized model

columns.json: the ordered list of features (sqft, bath, bhk, and one-hot location columns)

bangalore_home_prices_cleaned.csv: cleaned dataset used for EDA in the app

Running the Project Locally

To run this project on your own system:

Clone the repository and navigate inside it.

Install Python 3.10+ and required dependencies:

pip install -r requirements.txt


Ensure the following files are present:

app.py

banglore_home_prices_model.pickle

columns.json

bangalore_home_prices_cleaned.csv (create from notebook if missing)

Start the app:

streamlit run app.py


Open the app in your browser at http://localhost:8501.

Deployment

The project can be deployed easily to Streamlit Cloud by connecting the GitHub repo, ensuring requirements.txt and .streamlit/config.toml are included. On deployment, Streamlit automatically provisions the environment and provides a shareable link.

Key Learnings

Through this project, I practiced the complete lifecycle of a data science solution:

Rigorous data cleaning and feature engineering to handle real-world messiness.

Building and comparing models, and understanding why simpler models sometimes perform better when features are carefully prepared.

Designing a professional-grade interactive app for end users, moving beyond notebooks into real usability.

Handling deployment, theming, caching, and even PDF reporting to make the solution shareable and business-ready.

Credits

Dataset: Bengaluru House Price Data from Kaggle (Amitabh Ajoy).

Streamlit, Pandas, NumPy, Plotly, Scikit-learn, ReportLab for tooling.
