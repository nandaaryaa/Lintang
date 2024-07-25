from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import sqlite3

app = Flask(__name__, template_folder='templates')

# Load the CSV file globally
data_path = "data/audi.csv"
mobil_data = pd.read_csv(data_path)

# Load the pre-trained model
model = joblib.load('model.pkl')  # Adjust the path as necessary

# Create a numerical transformer pipeline
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Create the database table if it does not exist
def create_table():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY,
                  year INTEGER,
                  mileage REAL,
                  tax REAL,
                  mpg REAL,
                  engineSize REAL,
                  predicted_price REAL)''')
    conn.commit()
    conn.close()
create_table()

# Global variable to store prediction history
prediction_history = []

@app.route('/')
def home():
    return render_template("home.html")

def ValuePredictor(to_predict_list):
    try:
        # Convert the input list to DataFrame
        to_predict = pd.DataFrame([to_predict_list], columns=['year', 'mileage', 'tax', 'mpg', 'engineSize'])
        
        # Apply the same preprocessing
        to_predict = numerical_transformer.fit_transform(to_predict)
        
        # Predict using the model
        result = model.predict(to_predict)
        return result[0]
    except Exception as e:
        print(e)
        return None
    
def save_prediction_to_db(year, mileage, tax, mpg, engineSize, predicted_price):
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute("INSERT INTO predictions (year, mileage, tax, mpg, engineSize, predicted_price) VALUES (?, ?, ?, ?, ?, ?)",
              (year, mileage, tax, mpg, engineSize, predicted_price))
    conn.commit()
    conn.close()

@app.route('/', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        try:
            to_predict_dict = request.form.to_dict()
            to_predict_list = [
                int(to_predict_dict['year']),
                float(to_predict_dict['mileage']),
                float(to_predict_dict['tax']),
                float(to_predict_dict['mpg']),
                float(to_predict_dict['engineSize'])
            ]
            
            result = ValuePredictor(to_predict_list)
            
            if result is not None:
                # Save prediction result to database
                save_prediction_to_db(
                    to_predict_list[0],
                    to_predict_list[1],
                    to_predict_list[2],
                    to_predict_list[3],
                    to_predict_list[4],
                    result
                )
                # Store the prediction result and input data into history
                prediction_history.append({
                    'year': to_predict_list[0],
                    'mileage': to_predict_list[1],
                    'tax': to_predict_list[2],
                    'mpg': to_predict_list[3],
                    'engineSize': to_predict_list[4],
                    'predicted_price': result
                })
                return render_template("home.html", result=f"Price: £ {result}")
            else:
                return render_template("home.html", result="Error in prediction")
        except Exception as e:
            print(e)
            return "Terjadi kesalahan dalam prediksi"

@app.route('/list')
def data_list():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute("SELECT year, mileage, tax, mpg, engineSize, predicted_price FROM predictions")
    rows = c.fetchall()
    conn.close()
    
    rows = mobil_data.tail(10).to_dict(orient='records')  
    # Add prediction history to the rows
    for record in prediction_history:
        rows.append({
            'year': record['year'],
            'mileage': record['mileage'],
            'tax': record['tax'],
            'mpg': record['mpg'],
            'engineSize': record['engineSize'],
            'price': record['predicted_price']
        })
    return render_template("list.html", rows=rows)

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=3002)
