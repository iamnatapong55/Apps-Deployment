import streamlit as st
import pickle
import numpy as np
import time

# func to download model from pkl file
def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


data = load_model()
regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

# func to display on the prediction page
def show_predict_page():
    st.title("The Salary Prediction for Software Engineer")

    st.write("""### Please select the options to predict the salary""")

    countries = (
         'United Kingdom of Great Britain and Northern Ireland',
         'Netherlands', 'United States of America', 'Austria', 'Italy',
         'Canada', 'Germany', 'Poland', 'France', 'Brazil', 'Sweden',
         'Spain', 'Turkey', 'India', 'Russian Federation', 'Switzerland',
         'Australia'

    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)

    expericence = st.slider("Years of Experience", 0, 50, 3)


    finish = st.button("Calculate Salary")
    my_bar = st.progress(0)

    if finish:
        X = np.array([[country, education, expericence]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)

        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)

        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")
        st.snow()