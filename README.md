# Diabetes Prediction Multipage Web Application - Streamlit

In this project, I created a Multipage Early Diabetes Prediction Web app using the Streamlit framework.

This web app will do the following:

- **`Home Page:`** Showing raw data to users and metadata.

- **`Predict Diabetes Page:`** Predicting whether a person has diabetes or is prone to get diabetes in future by analysing the values of several features using the Decision Tree Classifier (simple and best classifiers).

- **`Visualise Decision Tree Page:`** Displays the correlation heatmap, confusion matrix plot and a decision tree plot.

### Things covered

1. Designing the **Home page** of the web app to display the raw dataset and other metadata.
  
2. Creating Decision Tree classifier and Optimised Decision Tree using the `GridSearchCV` to predict diabetes based on the values of features.

3. Designing the **Prediction page** of the web app that allows users to select the values of the features and display the prediction results on a click of the `Predict` button.

4. Designing the **Visualise Decision Tree** page that allows users to

 - Get the top correlated features using a `correlation heatmap`.
 - Display the `confusion matrix` plot to get the performance of the classifier.
 - Visualise the classifier with the help of an actual `decision tree plot`.