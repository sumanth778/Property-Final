import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

n_samples = 16000
locations = ['Manhattan', 'Brooklyn', 'Bronx', 'Queens', 'Staten Island', 'Harlem']
property_types = ['Apartment', 'Condo', 'House']


location_multiplier = {
    'Manhattan': 200000,
    'Brooklyn': 150000,
    'Queens': 100000,
    'Bronx': 80000,
    'Staten Island': 60000,
    'Harlem': 120000
}

property_type_multiplier = {
    'Apartment': 50000,
    'Condo': 70000,
    'House': 90000
}


df = pd.DataFrame({
    'bedrooms': np.random.randint(1, 7, size=n_samples),
    'bathrooms': np.random.randint(1, 5, size=n_samples),
    'square_feet': np.random.randint(600, 4000, size=n_samples),
    'location': np.random.choice(locations, size=n_samples),
    'year_built': np.random.randint(1980, 2023, size=n_samples),
    'garage': np.random.choice([0, 1], size=n_samples),
    'has_pool': np.random.choice([0, 1], size=n_samples),
    'property_type': np.random.choice(property_types, size=n_samples),
    'num_floors': np.random.randint(1, 4, size=n_samples),
    'has_basement': np.random.choice([0, 1], size=n_samples)
})

def calculate_price(row):
    base_price = (
        row['bedrooms'] * 50000 +
        row['bathrooms'] * 30000 +
        row['square_feet'] * 150 +
        row['garage'] * 25000 +
        row['has_pool'] * 40000 +
        row['num_floors'] * 15000 +
        row['has_basement'] * 20000 +
        location_multiplier[row['location']] +
        property_type_multiplier[row['property_type']]
    )
    noise = np.random.normal(0, 20000)  
    return int(base_price + noise)

df['price'] = df.apply(calculate_price, axis=1)

low_cutoff = df['price'].quantile(0.33)
high_cutoff = df['price'].quantile(0.66)

def price_to_category(price):
    if price < low_cutoff:
        return 'Low'
    elif price < high_cutoff:
        return 'Medium'
    else:
        return 'High'

df['price_category'] = df['price'].apply(price_to_category)


print(df['price_category'].value_counts())


df.to_csv('data.csv', index=False)
print("\nBalanced dataset saved as 'balanced_synthetic_real_estate.csv'")
