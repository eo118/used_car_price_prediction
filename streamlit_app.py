import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib  # use this if you want to load a saved model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

st.header("🚗 Used Car Price Analysis & Prediction")

# Load dataset
df = pd.read_csv("toyota.csv")

# ===== Preprocessing & Model Training =====
# For demo: you could also load a pre-trained model using joblib
df_clean = df.dropna()
categorical_features = ['model', 'transmission', 'fuelType']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded = encoder.fit_transform(df_clean[categorical_features])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_features))
df_encoded = pd.concat([df_clean.drop(columns=categorical_features), encoded_df], axis=1)

X = df_encoded.drop(columns='price')
y = df_encoded['price']

model = LinearRegression()
model.fit(X, y)

# ===== Sidebar: Visualization Filters =====
st.sidebar.subheader("📈 Customize Visualization")
group_col = st.sidebar.selectbox("Group by", ['model', 'transmission', 'fuelType', 'year'])
agg_type = st.sidebar.radio("Aggregation type", ['Mean Price', 'Median Price', 'Count'])

if agg_type == 'Mean Price':
    agg_df = df.groupby(group_col)['price'].mean().reset_index()
    y_col = 'price'
elif agg_type == 'Median Price':
    agg_df = df.groupby(group_col)['price'].median().reset_index()
    y_col = 'price'
else:
    agg_df = df[group_col].value_counts().reset_index()
    agg_df.columns = [group_col, 'count']
    y_col = 'count'

fig = px.bar(agg_df, x=group_col, y=y_col, title=f"{agg_type} by {group_col}")
st.plotly_chart(fig, use_container_width=True)

# ===== Prediction Interface =====
st.subheader("🔮 Predict Used Car Price")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        model_input = st.selectbox("Model", df['model'].unique())
        year = st.number_input("Year", min_value=1990, max_value=2024, value=2017)
        transmission = st.selectbox("Transmission", df['transmission'].unique())
    with col2:
        mileage = st.number_input("Mileage", min_value=0, value=30000)
        fuelType = st.selectbox("Fuel Type", df['fuelType'].unique())
        tax = st.number_input("Tax", min_value=0.0, value=150.0)
        mpg = st.number_input("MPG", min_value=0.0, value=55.0)
        engineSize = st.number_input("Engine Size", min_value=0.0, value=1.5)

    submit = st.form_submit_button("Predict Price")

    if submit:
        input_dict = {
            'model': model_input,
            'transmission': transmission,
            'fuelType': fuelType
        }
        input_encoded = encoder.transform([[input_dict['model'], input_dict['transmission'], input_dict['fuelType']]])
        input_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out())

        numerical_features = pd.DataFrame([[year, mileage, tax, mpg, engineSize]],
                                          columns=['year', 'mileage', 'tax', 'mpg', 'engineSize'])

        final_input = pd.concat([numerical_features, input_df], axis=1)

        # Align columns with training data
        missing_cols = set(X.columns) - set(final_input.columns)
        for col in missing_cols:
            final_input[col] = 0
        final_input = final_input[X.columns]  # Reorder

        predicted_price = model.predict(final_input)[0]
        st.success(f"Estimated Car Price: £{predicted_price:,.2f}")

