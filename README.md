# Used Car Price Prediction (Streamlit App)

This Streamlit app visualizes Toyota used car data and allows users to interact with the dataset.

## Features

- Uploads and reads `toyota.csv`
- Lets users select numeric columns to analyze
- Displays visualizations like histograms and bar charts based on user input

## How to Use

1. Open the Streamlit app on [Streamlit Cloud](https://YOUR-STREAMLIT-URL-HERE)
2. Select a column from the dropdown
3. View the visualization

## Setup Instructions (Locally)

```bash
git clone https://github.com/eo118/used_car_price_prediction.git
cd used_car_price_prediction
pip install -r requirements.txt
streamlit run streamlit_app.py

## Project Structure
streamlit-app/
├── streamlit_app.py
├── requirements.txt
├── toyota.csv
└── .streamlit/
    └── config.toml


