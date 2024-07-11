# Importing the necessary Python modules.
import streamlit as st
import numpy as np
import pandas as pd
import diabetes_home
import diabetes_predict
import diabetes_plots

# Configure your home page by setting its title and icon that will be displayed in a browser tab.
st.set_page_config(
	page_title = 'Early Diabetes Prediction Web App',
	page_icon = 'random',
	layout = 'wide',
	initial_sidebar_state = 'auto'
)

# Loading the dataset.
@st.cache()
def load_data():
	# Load the Diabetes dataset into DataFrame.
	df = pd.read_csv('diabetes.csv')
	df.head()

	# Rename the column names in the DataFrame.
	df.rename(columns = {"BloodPressure": "Blood_Pressure",}, inplace = True)
	df.rename(columns = {"SkinThickness": "Skin_Thickness",}, inplace = True)
	df.rename(columns = {"DiabetesPedigreeFunction": "Pedigree_Function",}, inplace = True)

	df.head()

	return df

diabetes_df = load_data()

pages_dict = {
	"Home": diabetes_home,
	"Predict Diabetes": diabetes_predict,
	"Visualise Decision Tree": diabetes_plots
}

st.sidebar.title("Navigation")

user_choice = st.sidebar.radio('Go to', tuple(pages_dict.keys()))
selected_page = pages_dict[user_choice]
selected_page.app(diabetes_df)