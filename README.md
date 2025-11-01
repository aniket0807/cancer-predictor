Cancer Predictor
A Python-based project for predicting breast cancer malignancy from cytological data using machine learning. The app leverages logistic regression and a user-friendly Streamlit interface to assist in rapid cancer diagnosis based on tissue measurements.

Features
  Interactive Web App (Streamlit)
    -Input cell nucleus measurements via sliders.
    -Visualize measurements with a radar chart.
    -Predict malignancy (benign/malignant) and display probabilities.
    -Designed to assist medical professionals in diagnosis (not a substitute for professional evaluation).
  Machine Learning Pipeline
    -Data cleaning and preprocessing (mapping, scaling, removing irrelevant columns).
    -Logistic Regression classifier with performance outputs (accuracy, classification report).
    -Trained on breast cancer dataset (data/data.csv).

Project Structure
app/
  main.py : Streamlit front-end, visualization, and prediction logic.
model/
  main.py : Model creation, training, saving pickled model/scaler.
  model.pkl, scaler.pkl : Saved ML model and scaler.
data/
  data.csv : Source data, cell nucleus measurements and diagnosis.
assets/
  style.css : Custom styling for the Streamlit app.
requirements.txt : List of required packages and versions.

Input Features
Thirty key cell nucleus features:
-Mean, Standard Error, and Worst values for:
  -Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave Points, Symmetry, Fractal Dimension
Diagnosis label:
-M = Malignant (1), B = Benign (0)

Live: https://breast-cancer--predictor.streamlit.app/

Installation
bash
# Clone the repository
git clone https://github.com/aniket0807/cancer-predictor.git
cd cancer-predictor

# Install requirements
pip install -r requirements.txt
Usage
Training the Model
bash
python model/main.py

This script:
Loads and cleans data/data.csv
Trains a logistic regression model
Saves model.pkl and scaler.pkl in the model/ folder

Running the Web App
bash
streamlit run app/main.py

Opens an interactive web dashboard for data input and visualization.
Predicts the likelihood of malignancy for entered tissue measurements.
Displays feature radar charts, cluster prediction, and probabilities.

Key Dependencies
numpy
pandas
plotly
scikit-learn
streamlit

See requirements.txt for exact versions.

Example
You can use the web app to input custom measurements or connect actual cytology lab readings for fast AI-aided diagnostics.

Disclaimer
This tool provides computational support for cancer diagnosis but should not be used as a substitute for a professional medical opinion.

Project maintained by aniket0807
Feel free to star 🌟 or fork 🍴 if you find it useful!
