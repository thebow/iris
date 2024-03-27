import streamlit as st
import pickle

# Define the target names for the Iris dataset
target_names = ['setosa', 'versicolor', 'virginica']

# Create Streamlit app
st.title("Iris Flower Predictor testing..")
st.header("Enter the values for the flower's features:")

# Input fields for sepal length, sepal width, petal length, and petal width
sepal_length = st.slider("Sepal length", 0.0, 10.0, 5.0)
sepal_width = st.slider("Sepal width", 0.0, 10.0, 5.0)
petal_length = st.slider("Petal length", 0.0, 10.0, 5.0)
petal_width = st.slider("Petal width", 0.0, 10.0, 5.0)

# Load the pre-trained Random Forest Classifier model from the pickle file
with open("iris_rf_model.pkl", "rb") as model_file:
    loaded_rf = pickle.load(model_file)

# Define prediction button
if st.button("Predict"):
    # Use the pre-trained classifier to predict the type of iris flower
    pred = loaded_rf.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    flower_type = target_names[pred[0]]  # Use the defined target names
    # Display the predicted type of iris flower
    st.write("The predicted type of iris flower is", flower_type)
