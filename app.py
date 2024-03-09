from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from contentfilter import contentfilter

app = Flask(__name__)

# Load the CSV file into a DataFrame
df = pd.read_csv('cars_ds_final_updated.csv')

# Convert 'Ex-Showroom_Price' to numeric type, handling non-numeric values
df['Ex-Showroom_Price'] = pd.to_numeric(df['Ex-Showroom_Price'], errors='coerce')

# Convert 'Length' to numeric type, handling non-numeric values
df['Length'] = pd.to_numeric(df['Length'], errors='coerce')

# Convert 'City_Mileage' to numeric type, handling non-numeric values
df['City_Mileage'] = pd.to_numeric(df['City_Mileage'], errors='coerce')

# Convert 'Displacement' to strings and remove non-numeric characters
df['Displacement'] = df['Displacement'].astype(str).str.replace('[^\d.]', '', regex=True)

# Convert 'Displacement' to numeric type, handling non-numeric values
df['Displacement'] = pd.to_numeric(df['Displacement'], errors='coerce')

# Remove rows with missing values in relevant columns
df.dropna(subset=['Ex-Showroom_Price', 'Body_Type', 'Length', 'Fuel_Type', 'Displacement'], inplace=True)

# Combine relevant features into a single text-based feature for TF-IDF vectorization
df['Features'] = df['Ex-Showroom_Price'].astype(str) + ' ' + df['Body_Type'].astype(str) + ' ' + df['Length'].astype(str) + ' ' + df['Fuel_Type'].astype(str) + ' ' + df['Displacement'].astype(str)

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the TF-IDF vectorizer on the 'Features' column
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Features'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Define the route for getting recommendations
@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    # Get query parameters from the request
    min_price = float(request.args.get('min_price'))
    max_price = float(request.args.get('max_price'))
    body_type = request.args.get('body_type')
    fuel_type = request.args.get('fuel_type')
    displacement_min = float(request.args.get('displacement_min'))
    displacement_max = float(request.args.get('displacement_max'))
    length_filter = request.args.get('length_filter')

    # Get recommendations and the number of available cars within the specified criteria
    recommendations, num_available_cars = contentfilter.get_recommendations(min_price,max_price,body_type,fuel_type,displacement_min,displacement_max,length_filter)

    # Convert recommendations DataFrame to JSON
    #recommendations_json = recommendations.to_json(orient='records')


    # Return recommendations and the number of available cars as JSON response
    return jsonify({'recommendations': recommendations, 'num_available_cars': num_available_cars})


    # Return recommendations and the number of available cars as JSON response
    #return jsonify({'recommendations': recommendations_json, 'num_available_cars': num_available_cars})

'''
if __name__ == '__main__':
    app.run(debug=True)
'''