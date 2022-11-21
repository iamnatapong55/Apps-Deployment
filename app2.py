import streamlit as st
import pandas as pd
import streamlit as st
import pickle
import numpy as np
import time


def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map


def clean_experience(x):
    if x == 'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)


def clean_education(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral' in x:
        return 'Post grad'
    return 'Less than a Bachelors'


@st.cache
def load_data():
    df = pd.read_csv("survey_results_public.csv")
    df = df[["Country", "EdLevel", "YearsCodePro", "ConvertedCompYearly"]]
    df = df[df["ConvertedCompYearly"].notnull()]
    df = df.dropna()

    country_map = shorten_categories(df.Country.value_counts(), 400)
    df["Country"] = df["Country"].map(country_map)
    df = df[df["ConvertedCompYearly"] <= 250000]
    df = df[df["ConvertedCompYearly"] >= 10000]
    df = df[df["Country"] != "Other"]

    df["YearsCodePro"] = df["YearsCodePro"].apply(clean_experience)
    df["EdLevel"] = df["EdLevel"].apply(clean_education)
    df = df.rename({"ConvertedCompYearly": "Salary"}, axis=1)
    return df


df = load_data()


def show_explore_page():
    st.title("Explore Software Engineer Salaries")

    st.write(
        """
    ### The data source is from the Stack Overflow Developer Survey 2022
    """
    )

    st.write("""#### Amount of Data from different countries""")
    data = df["Country"].value_counts()
    st.bar_chart(data)
    st.write(
        """
    #### Mean Salary Based On Country
    """
    )

    data = df.groupby(["Country"])["Salary"].mean().sort_values(ascending=True)
    st.bar_chart(data)

    st.write(
        """
    #### Mean Salary Based On Experience
    """
    )

    data = df.groupby(["YearsCodePro"])["Salary"].mean().sort_values(ascending=True)
    st.line_chart(data)

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

page = st.sidebar.selectbox(" Please select the modes from dropdown", ("Predict", "Explore"))

if page == "Predict":
    show_predict_page()
else:
    show_explore_page()
