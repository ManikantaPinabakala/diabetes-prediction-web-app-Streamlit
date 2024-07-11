import streamlit as st

def app(diabetes_df):
  st.title("Early Diabetes Prediction Web App")
  st.markdown('''<p style="color:red;">
                    Diabetes is a chronic (long-lasting) health condition
                    that affects how your body turns food into energy. There
                    isn’t a cure yet for diabetes, but losing weight, eating
                    healthy food, and being active can really help in reducing
                    the impact of diabetes. This Web app will help you to predict
                    whether a person has diabetes or is prone to get diabetes in
                    future by analysing the values of several features using the
                    Decision Tree Classifier.
                </p>''', unsafe_allow_html=True)
  
  st.header("View Data")
  with st.beta_expander("View Full Dataset"):
    st.dataframe(diabetes_df)
  
  st.subheader("Columns Description:")
  beta_col1, beta_col2, beta_col3 = st.beta_columns([1, 1, 1])

  with beta_col1:
    if st.checkbox("Show all columns names"):
      st.table(list(diabetes_df.columns))
  
  with beta_col2:
    if st.checkbox("View column data-types"):
      st.table(diabetes_df.dtypes)
  
  with beta_col3:
    if st.checkbox("View column data"):
      selected_column = st.selectbox('Select the column', list(diabetes_df.columns))
      st.write(diabetes_df[selected_column])
  
  st.subheader("Dataset summary:")
  if st.checkbox("Show summary"):
    st.write(diabetes_df.describe())