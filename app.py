from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import os
import kaggle
import zipfile
from flasgger import Swagger

app = Flask(__name__)

# Swagger config
app.config['SWAGGER'] = {
    'title': 'Used Car Price Prediction API',
    'uiversion': 3
}
swagger = Swagger(app)

# SQLite DB setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cars.db'
db = SQLAlchemy(app)

# Define a database model
class Car(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model = db.Column(db.String(50), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Float, nullable=False)
    transmission = db.Column(db.String(50), nullable=False)
    mileage = db.Column(db.Integer, nullable=False)
    fuelType = db.Column(db.String(50), nullable=False)
    tax = db.Column(db.Float, nullable=False)
    mpg = db.Column(db.Float, nullable=False)
    engineSize = db.Column(db.Float, nullable=False)

# Create the database
with app.app_context():
    db.create_all()

def preprocess_data(df):
    """Preprocess data: Drop missing values and one-hot encode categorical features."""
    categorical_features = ['model', 'transmission', 'fuelType']
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_features = encoder.fit_transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
    
    df = pd.concat([df, encoded_df], axis=1).drop(columns=categorical_features)

    df = df.dropna()
    return df, encoder

# Global variables for model and encoder
model = None
encoder = None

@app.route('/reload', methods=['POST'])
def reload_data():
    """
    Reload data from the Toyota used car dataset, clear the database, and train a new model.
    ---
    responses:
      200:
        description: Summary statistics of reloaded data
    """
    global model, encoder

    dataset_name = "adityadesai13/used-car-dataset-ford-and-mercedes"
    file_name = "toyota.csv"

    # Authenticate Kaggle API (ensure you have `kaggle.json` in ~/.kaggle/)
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        return jsonify({"error": "Kaggle API key not found."}), 500

    # Download dataset
    kaggle.api.dataset_download_files(dataset_name, path="./", unzip=True)

    # Check if Toyota dataset is available
    if not os.path.exists(file_name):
        return jsonify({"error": "Toyota dataset not found in Kaggle files."}), 500

    df = pd.read_csv(file_name)
    
    # Clear database
    db.session.query(Car).delete()

    # Insert new data
    for _, row in df.iterrows():
        new_car = Car(
            model=row['model'],
            year=int(row['year']),
            price=float(row['price']),
            transmission=row['transmission'],
            mileage=int(row['mileage']),
            fuelType=row['fuelType'],
            tax=float(row['tax']),
            mpg=float(row['mpg']),
            engineSize=float(row['engineSize'])
        )
        db.session.add(new_car)
    db.session.commit()

    # Preprocess data and train model
    df, encoder = preprocess_data(df)
    X = df.drop(columns='price')
    y = df['price']
    model = LinearRegression()
    model.fit(X, y)

    # Log the availability of the model and encoder
    print(f"Model available: {model is not None}")
    print(f"Encoder available: {encoder is not None}")

    summary = {
        'total_cars': len(df),
        'average_price': df['price'].mean(),
        'min_price': df['price'].min(),
        'max_price': df['price'].max(),
        'average_mileage': df['mileage'].mean(),
        'average_engineSize': df['engineSize'].mean()
    }

    summary = {key: (int(value) if isinstance(value, np.int64) else float(value) if isinstance(value, np.float64) else value)
               for key, value in summary.items()}

    return jsonify(summary)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the price of a used Toyota car.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            model:
              type: string
            year:
              type: integer
            transmission:
              type: string
            mileage:
              type: integer
            fuelType:
              type: string
            tax:
              type: number
            mpg:
              type: number
            engineSize:
              type: number
    responses:
      200:
        description: Predicted car price
    """
    global model, encoder

    if model is None or encoder is None:
        return jsonify({"error": "Model not trained. Please refresh the data by calling the '/reload' endpoint first."}), 400


    data = request.json
    try:
        input_data = {
            'model': data.get('model').strip().lower(),
            'year': pd.to_numeric(data.get('year'), errors='coerce'),
            'transmission': data.get('transmission').strip(),
            'mileage': pd.to_numeric(data.get('mileage'), errors='coerce'),
            'fuelType': data.get('fuelType').strip(),
            'tax': pd.to_numeric(data.get('tax'), errors='coerce'),
            'mpg': pd.to_numeric(data.get('mpg'), errors='coerce'),
            'engineSize': pd.to_numeric(data.get('engineSize'), errors='coerce')
        }
        

        if None in input_data.values():
            return jsonify({"error": "Missing or invalid parameters"}), 400

        # Encode categorical features
        input_df = pd.DataFrame([[input_data['model'], input_data['transmission'], input_data['fuelType']]],
                        columns=['model', 'transmission', 'fuelType'])

        encoded_features = encoder.transform(input_df)
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())
        
        # Prepare input array
        numerical_features = np.array([input_data['year'], input_data['mileage'], input_data['tax'], input_data['mpg'], input_data['engineSize']])
        final_input = np.concatenate((numerical_features, encoded_df.iloc[0].values)).reshape(1, -1)
        
        # Predict price
        predicted_price = model.predict(final_input)[0]

        return jsonify({"predicted_price": predicted_price})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
