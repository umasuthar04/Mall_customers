import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

kmeans=joblib.load('Model.pkl')
df= pd.read_csv('Mall_Customers.csv')
x=df[["Annual Income (k$)","Spending Score (1-100)"]]

X_array=x.values

st.set_page_config(page_title="Customer Cluster Prediction",layout="centered")
st.title("Customer Cluster Prediction")
st.write("Enter the Annual Income and Spending Score to predict the cluster of the customer.")

annual_income=st.number_input("Annual Income (k$)", min_value=0, max_value=400, value=50)
spending_score=st.slider("Spending Score (1-100)", 1, 100, 20)

#Predicting the cluster
if st.button("Predict Cluster"):
    input_data = np.array([[annual_income, spending_score]])
    cluster = kmeans.predict(input_data)[0]
    st.write(f"The predicted cluster for the customer is: {cluster}")