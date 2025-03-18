# Used Car Price Prediction API

This is a Flask-based API that predicts used car prices based on several factors like the model, year, transmission type, mileage, fuel type, tax, mpg, and engine size. The API has two main endpoints:
- `/reload`: Reloads the data and trains the model.
- `/predict`: Predicts the price of a used car based on the input parameters.

## Data Source and Prediction Process

### Data Source

The dataset used for this project comes from a Kaggle dataset containing information about used Toyota cars. This dataset includes various features such as:
- **Model**: The model of the car.
- **Year**: The year the car was manufactured.
- **Transmission**: The type of transmission (automatic, manual, etc.).
- **Mileage**: The number of miles the car has been driven.
- **FuelType**: The type of fuel the car uses.
- **Tax**: The road tax for the car.
- **MPG**: The miles per gallon the car achieves.
- **EngineSize**: The size of the car's engine.

You can access and download the dataset from Kaggle using the following link: [Used Car Dataset on Kaggle](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes/data?select=toyota.csv).

### Prediction Process

The application uses a **Linear Regression Model** to predict the price of a used Toyota car based on the features provided. Here's a brief overview of the prediction process:

1. **Data Preprocessing**: The data is cleaned and processed. Missing values are handled, and categorical variables like model, transmission, and fuelType are one-hot encoded to make them suitable for machine learning models.
2. **Model Training**: A Linear Regression model is trained using features like model, year, mileage, fuelType, and others to predict the price.
3. **Prediction**: Once trained, the model can predict the price based on user input, such as the car's model, year, mileage, and other features.

By using this model, the app can provide quick predictions for used car prices based on historical data.

## Prerequisites

Before you can set up and run this app, ensure you have the following software installed:

- **Python 3.9+**
- **pip** (Python package installer)
- **Virtualenv** (Optional but recommended)

## Setting up on macOS and Windows

### 1. Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/eo118/used-car-price-prediction.git
cd used-car-price-prediction
```

### 2. Create a Virtual Environment (Optional but Recommended)

You can create a virtual environment to isolate the project dependencies.

For macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install the Dependencies

Install the required Python dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Flask requires some environment variables to run the app correctly. Create a `.env` file in the project root with the following content:

```bash
FLASK_APP=app.py
FLASK_ENV=development
```

For macOS, you can set the environment variables using the following commands:

```bash
export FLASK_APP=app.py
export FLASK_ENV=development
```

For Windows, you can set the environment variables using the following commands in PowerShell:

```bash
$env:FLASK_APP = "app.py"
$env:FLASK_ENV = "development"
```

### 5. Initialize the SQLite Database

To set up the SQLite database for the first time, run:

```bash
flask shell
```

Inside the shell, run:
```python
from app import db
db.create_all()
exit()
```

### 6. Running the Application

Once everything is set up, you can run the application with the following command:

```bash
flask run
```

By default, the app will run on [http://127.0.0.1:5000](http://127.0.0.1:5000).

### 7. Swagger Documentation

You can access the Swagger documentation for the API at:

```
http://127.0.0.1:5000/apidocs/
```

### 8. Testing the Endpoints

#### Reload Data

To reload the data and train the model, use the `/reload` endpoint:

```bash
curl -X POST http://127.0.0.1:5000/reload
```

#### Predict Price

To predict a rental price, you can use the `/predict` endpoint. Here's an example request:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "corolla",
    "year": 2015,
    "transmission": "manual",
    "mileage": 50000,
    "fuelType": "petrol",
    "tax": 150,
    "mpg": 45,
    "engineSize": 1.6
}'
```

### 9. Stopping the Application

To stop the Flask app, you can press `Ctrl + C` in the terminal window where the app is running.

---

## Troubleshooting

### Common Issues

- **Environment variables not being set**: Ensure you have set the environment variables correctly, especially when switching between macOS and Windows.

- **Database initialization issues**: If the app crashes because of database-related errors, make sure you have run the `flask shell` commands to initialize the database properly.

- **Dependency issues**: Ensure that you are using the correct version of Python (3.9+) and have installed the dependencies using `pip install -r requirements.txt`.

---

## License

This project is licensed under the MIT License.

## Running Tests

We use `pytest` for running tests on this application. Before running the tests, ensure all dependencies are installed and the application is properly set up.

### Setting up for Testing

1. Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

2. Export the `PYTHONPATH` environment variable to ensure Python can locate the app module.

For macOS/Linux:
```bash
export PYTHONPATH=.
```

For Windows (PowerShell):
```bash
$env:PYTHONPATH="."
pytest
```

3. Run the tests:

```bash
pytest
```

This will execute all the tests located in the `tests/` folder and provide feedback on the application behavior.

## Deploying to Heroku

### 1. Install the Heroku CLI

Before deploying the application to Heroku, you need to install the Heroku CLI. You can follow the steps below to install it on your machine.

#### macOS:

You can install the Heroku CLI using Homebrew:
```bash
brew tap heroku/brew && brew install heroku
```

#### Windows:

Download and run the Heroku installer from the [Heroku Dev Center](https://devcenter.heroku.com/articles/heroku-cli).

#### Verify Installation:

Once installed, verify the installation by running:

```bash
heroku --version
```

You should see the version of Heroku CLI installed.

### 2. Log in to Heroku

Log in to your Heroku account from the terminal:

```bash
heroku login
```

This will open a web browser for you to log in to your Heroku account.

### 3. Prepare the App for Deployment

Ensure your `requirements.txt` and `Procfile` are present in the project root.

- **Procfile**: Create a `Procfile` in the root directory with the following content to tell Heroku how to run the app:

```bash
web: flask run --host=0.0.0.0 --port=$PORT
```

### 4. Create a Heroku App

Run the following command to create a new Heroku app:

```bash
heroku create
```

### 5. Deploy to Heroku

After you've created your Heroku app, deploy your app using Git:

```bash
git add .
git commit -m "Initial commit"
git push heroku main
```

### 6. Scale the Application

Heroku apps require at least one running dyno. Scale your app to run one web dyno:

```bash
heroku ps:scale web=1
```
