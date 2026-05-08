import streamlit as st
import pandas as pd 
import numpy as np
import pickle as pkl


st.title("Car Price Prediction App")
st.write("Welcome to pruthviraj chavan's Project")

model=pkl.load(open("CPQ.pkl","rb+"))
ds=pd.read_csv("cleaned_data.csv")

companies=sorted(ds["company"].unique())
fuel_types=sorted(ds["fuel_type"].unique())

years=[]
for i in range(1995,2025):
    years.append(i)

company=st.sidebar.selectbox("Select Company:",companies,key="company")
names=sorted(ds[ds["company"]==company]["name"].unique())#to get name of that company car
name=st.sidebar.selectbox("Select Name:",names,key="name")#Column in app
year=st.sidebar.selectbox("Select year",years)
kms_driven=st.sidebar.text_input("Enter the Kms Driven","10000")
fuel_type=st.sidebar.selectbox("Select Fuel Type",fuel_types,key="fuel_type")
st.write("You selected",company,name)

if st.sidebar.button("Predict"):
    columns=["company","name","year","kms_driven","fuel_type"]
    # data=[[company,name,year,kms_driven,fuel_type]] --for anaconda prompt--
    # st.write("You Provide Following Information:")
    # st.write("Comapny:",company)
    # st.write("Name:",name)
    # st.write("Year:",year)
    # st.write("Total Kms_driven:",kms_driven)
    # st.write("Fuel Type:",fuel_type)
    #myinput=pd.DataFrame(data=data,columns=columns)
    myinput =pd.DataFrame([[company,name,int(year),int(kms_driven),fuel_type]],columns=columns)
    result=model.predict(myinput)
    st.sucess(f'The predicted price is:{np.round(result[0],2)}')
    st.write("Predicted price:",result[0,0])
