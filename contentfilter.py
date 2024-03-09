import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class contentfilter():
        

    def get_recommendations(min_price, max_price, body_type, fuel_type, displacement_min, displacement_max, length_filter, num_recommendations=5):
        
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

        
        # Convert the user's input for 'displacement_min' and 'displacement_max' to numeric
        displacement_min = float(displacement_min)
        displacement_max = float(displacement_max)
        
        # Filter cars based on user preferences for Ex-Showroom Price, Body_Type, Fuel_Type, Displacement, and Length
        if length_filter == 'less':
            filtered_cars = df[(df['Ex-Showroom_Price'] >= min_price) & (df['Ex-Showroom_Price'] <= max_price) &
                            (df['Body_Type'].str.lower().str.strip() == body_type.lower().strip()) &
                            (df['Fuel_Type'].str.lower().str.strip() == fuel_type.lower().strip()) &
                            (df['Displacement'] >= displacement_min) &
                            (df['Displacement'] <= displacement_max) &
                            (df['Length'] < 4000)]
        elif length_filter == 'more':
            filtered_cars = df[(df['Ex-Showroom_Price'] >= min_price) & (df['Ex-Showroom_Price'] <= max_price) &
                            (df['Body_Type'].str.lower().str.strip() == body_type.lower().strip()) &
                            (df['Fuel_Type'].str.lower().str.strip() == fuel_type.lower().strip()) &
                            (df['Displacement'] >= displacement_min) &
                            (df['Displacement'] <= displacement_max) &
                            (df['Length'] >= 4000)]
        else:
            # If 'length_filter' is not 'less' or 'more', return an empty DataFrame
            return pd.DataFrame(columns=['Make', 'Model', 'Ex-Showroom_Price', 'Body_Type', 'Length', 'Fuel_Type', 'Displacement']), 0

        num_available_cars = len(filtered_cars)

        if num_available_cars == 0:
            return pd.DataFrame(columns=['Make', 'Model', 'Ex-Showroom_Price', 'Body_Type', 'Length', 'Fuel_Type', 'Displacement']), num_available_cars  # Return an empty DataFrame if no cars match the criteria

        # Extract and display the 'Make', 'Model', 'Ex-Showroom_Price', 'Body_Type', 'Length', 'Fuel_Type', and 'Displacement' columns
        recommended_cars = filtered_cars[['Make', 'Model', 'Ex-Showroom_Price', 'Body_Type', 'Length', 'Fuel_Type', 'Displacement']]

         # Convert the recommendations to a list of dictionaries
        recommendations_list = recommended_cars.to_dict(orient='records')
        
        return recommendations_list, num_available_cars


        #return recommended_cars, num_available_cars


    def get_collabRecommendations():

        # Load the car_ratings.csv dataset
        car_ratings_df = pd.read_csv('maintenance.csv')

        sort_column = input("Enter the column for sorting (e.g., 'Avg_Maintenance', 'Avg_KM', 'Safety_Ratings'): ")
        if sort_column=="Safety_Ratings"or sort_column=="Nearby_Service_Centers" or sort_column=="Reliability" or sort_column=="Avg_KM":
        # Extract the recommended cars from the content-based recommendations
            content_recommendations, _ = contentfilter.get_recommendations(user_min_price, user_max_price, user_body_type, user_fuel_type, user_displacement_min, user_displacement_max, user_length_filter)

        # Filter the ratings for the recommended models
            recommended_models = content_recommendations['Model']
            ratings_for_recommended_models = car_ratings_df[car_ratings_df['Model'].isin(recommended_models)]

        # Merge the content-based recommendations with ratings
            merged_recommendations = content_recommendations.merge(ratings_for_recommended_models, on='Model', how='left')

        # Sort the ratings in descending order
            sorted_ratings = merged_recommendations.sort_values(by=sort_column, ascending=False)

        # Print the sorted ratings including all columns from content-based recommendations
        #print(sorted_ratings[['Make', 'Model','Safety_Ratings', 'Ex-Showroom_Price', 'Body_Type', 'Length', 'Fuel_Type', 'Displacement']])
        # Print the sorted ratings including all columns from content-based recommendations and 'Safety_Ratings'
            print(sorted_ratings[['Model', sort_column, 'Ex-Showroom_Price', 'Body_Type', 'Length', 'Fuel_Type', 'Displacement']])
        elif sort_column=="Avg_Maintenance":
            content_recommendations, _ = contentfilter.get_recommendations(user_min_price, user_max_price, user_body_type, user_fuel_type, user_displacement_min, user_displacement_max, user_length_filter)

        # Filter the ratings for the recommended models
            recommended_models = content_recommendations['Model']
            ratings_for_recommended_models = car_ratings_df[car_ratings_df['Model'].isin(recommended_models)]

        # Merge the content-based recommendations with ratings
            merged_recommendations = content_recommendations.merge(ratings_for_recommended_models, on='Model', how='left')

        # Sort the ratings in descending order
            sorted_ratings = merged_recommendations.sort_values(by='Avg_Maintenance', ascending=False)

        # Print the sorted ratings including all columns from content-based recommendations
        #print(sorted_ratings[['Make', 'Model','Safety_Ratings', 'Ex-Showroom_Price', 'Body_Type', 'Length', 'Fuel_Type', 'Displacement']])
        # Print the sorted ratings including all columns from content-based recommendations and 'Safety_Ratings'
            print(sorted_ratings[['Model', 'Avg_Maintenance', 'Ex-Showroom_Price', 'Body_Type', 'Length', 'Fuel_Type', 'Displacement']])
        sorted_ratings.to_csv('content_based_recommendations.csv', index=False)
        print("Content-based recommendations have been saved to 'content_based_recommendations.csv'.")

        # Get recommendations and the number of available cars within the specified criteria
        recommendations, num_available_cars = contentfilter.get_recommendations(user_min_price, user_max_price, user_body_type, user_fuel_type, user_displacement_min, user_displacement_max, user_length_filter)

        print(f"Number of Cars Available within the Price Range and Length Filter: {num_available_cars}")
        print("Recommended Cars:")
        print(recommendations[['Make', 'Model', 'Ex-Showroom_Price', 'Body_Type', 'Length', 'Fuel_Type', 'Displacement']])




'''

# User input for each feature
user_min_price = float(input("Enter your minimum preferred Ex-Showroom Price: "))
user_max_price = float(input("Enter your maximum preferred Ex-Showroom Price: "))
user_body_type = input("Enter your preferred Body Type (e.g., SUV, Sedan, Hatchback, MPV): ")
user_fuel_type = input("Enter your preferred Fuel Type (e.g., Diesel, Petrol, Electric): ")
user_displacement_min = int(input("Enter your minimum preferred Displacement: "))
user_displacement_max = int(input("Enter your maximum preferred Displacement: "))
user_length_filter = input("Enter 'less' or 'more' for Length filter: ")

myclass= cont_collab()
output= myclass.get_recommendations(user_min_price,user_max_price,user_body_type,user_fuel_type,user_displacement_min,user_displacement_max,user_length_filter)
print(output)

'''