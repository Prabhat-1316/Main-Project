# main/views.py
import pandas as pd
from django.shortcuts import render
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Import Random Forest Regressor
from sklearn import metrics

def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    try:
        # Load and preprocess the data
        data = pd.read_csv(r"C:\Users\haunt\testing\data.csv")
        df = data.dropna()
        columns_to_keep = ['bath', 'balcony', 'price', 'total_sqft_int', 'bhk', 'price_per_sqft']
        clean_data = df[columns_to_keep]

        # Define features and target
        X = clean_data.drop('price', axis=1)
        y = clean_data['price']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=51)

        # Train the Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=51)  # You can adjust hyperparameters
        model.fit(X_train, y_train)

        # Get input values from request
        var1 = float(request.GET.get('n1', 0))
        var2 = float(request.GET.get('n2', 0))
        var3 = float(request.GET.get('n3', 0))
        var4 = float(request.GET.get('n4', 0))
        var5 = float(request.GET.get('n5', 0))

        # Predict the price using Random Forest
        pred = model.predict(np.array([var1, var2, var3, var4, var5]).reshape(1, -1))
        pred = round(pred[0])
        price = "The predicted price using Random Forest is " + str(pred)

        return render(request, 'predict.html', {"result2": price})

    except Exception as e:
        print(f"Error in result view: {e}")
        return render(request, 'predict.html', {"result2": "Error occurred during prediction. Please try again."})
