from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import requests

app = Flask(__name__)


def download_model():
    model_url = "https://github.com/sumanth778/Property-Final/releases/download/Model/Random_forest_price_regressor.pkl"
    model_path = "Random_forest_price_regressor.pkl"
    

    if not os.path.exists(model_path):
        print(f"Downloading model from GitHub Releases: {model_url}")
        try:
            response = requests.get(model_url, stream=True)
            response.raise_for_status()  
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Model downloaded successfully!")
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise
    

    return joblib.load(model_path)

try:
    model = download_model()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None

try:
    df = pd.read_csv("balanced_synthetic_real_estate.csv")

    location_le = LabelEncoder()
    location_le.classes_ = df['location'].unique()

    property_type_le = LabelEncoder()
    property_type_le.classes_ = df['property_type'].unique()
    
except FileNotFoundError:
    print("CSV file not found. Using hardcoded categories...")
    location_le = LabelEncoder()
    location_le.classes_ = np.array(['Manhattan', 'Brooklyn', 'Bronx', 'Queens', 'Staten Island', 'Harlem'])
    
    property_type_le = LabelEncoder()
    property_type_le.classes_ = np.array(['Apartment', 'Condo', 'House'])

@app.route('/')
def home():
    """Home page with dropdown options"""
    try:
        locations = sorted(df['location'].unique())
        property_types = sorted(df['property_type'].unique())
    except:
        locations = ['Manhattan', 'Brooklyn', 'Bronx', 'Queens', 'Staten Island', 'Harlem']
        property_types = ['Apartment', 'Condo', 'House']
    
    return render_template('index.html', locations=locations, property_types=property_types)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict house price based on input features"""
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
    
    try:
        data = request.get_json()
        
        location_encoded = location_le.transform([data['location']])[0]
        property_type_encoded = property_type_le.transform([data['property_type']])[0]
        
        input_data = {
            'bedrooms': int(data['bedrooms']),
            'bathrooms': int(data['bathrooms']),
            'square_feet': int(data['square_feet']),
            'location': location_encoded,
            'year_built': int(data['year_built']),
            'garage': int(data['garage']),
            'has_pool': int(data['has_pool']),
            'property_type': property_type_encoded,
            'num_floors': int(data['num_floors']),
            'has_basement': int(data['has_basement'])
        }

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]

        formatted_price = f"${round(float(prediction), 2):,}"
        
        return jsonify({
            'predicted_price': round(float(prediction), 2),
            'formatted_price': formatted_price,
            'success': True
        })

    except KeyError as e:
        return jsonify({'error': f'Missing field: {str(e)}', 'success': False}), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid input value: {str(e)}', 'success': False}), 400
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 400

@app.route('/health')
def health():
    """Health check endpoint for Railway monitoring"""
    return jsonify({
        'status': 'healthy' if model is not None else 'model_not_loaded',
        'model_loaded': model is not None,
        'location_categories': list(location_le.classes_),
        'property_categories': list(property_type_le.classes_)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
