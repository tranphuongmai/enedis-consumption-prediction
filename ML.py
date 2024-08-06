import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split






def predict():
    # Assistant = a
    # User's input = user
    
    a = st.chat_message("assistant")
    
    a.write("Hi there, How are you today?")
    a.write("Thanks for using this application to predict the ***Regional Electricity Consumption*** in France.")
    a.write("We need you to answer some important questions which will provide us enough information for the prediction 🤖")

    user = st.chat_message("user")
    user.write("No problem, I can use my phone to check for the weather indicators.")

    a = st.chat_message("assistant")
    a.write("Thank you. Here we go!")
    a.write("In which ***region*** would you like to predict the electric consumption?")

    # Region
    

    region = st.radio(
                            "Please select one",
                            options = ["***Centre-Val de Loire***", "***Haust-de-France***"],
                            key="region"
                            )
    
    if "region" not in st.session_state:
        st.session_state.region = "Centre-Val de Loire"

    if region == "Centre-Val de Loire":
        region_code = 24
    else:
        region_code = 32

    # Season
    a = st.chat_message("assistant")
    a.write("In which ***season***?")

    

    season = st.radio(
                      "Please select one",
                       options = ["***Spring***", "***Summer***",
                         "***Autumn***", "***Winter***"],
                        key="season"
                            )
    if "season" not in st.session_state:
        st.session_state.season = "Spring"
 
    if season == "Spring":
        season_code = 0
    if season == "Summer":
        season_code = 1
    if season == "Autumn":
        season_code = 2
    else:
        season_code = 3

    # Date
    a = st.chat_message("assistant")
    a.write("Is the predicted date on any ***holiday***?")

    

    hol = st.radio(
                      "Please select one",
                        options = ["***Normal Day***", "***Week-end***",
                         "***National Holidays***", "***School Holidays***"],
                        key="holiday"
                            )
    
    if "holiday" not in st.session_state:
        st.session_state.holiday = "Week-end"
       
    if hol == "Normal Day":
        hol_code = 0
    if hol == "Week-end":
        hol_code = 1
    if hol == "National Holidays":
        hol_code = 2
    else:
        hol_code = 3

    
    
    a = st.chat_message("assistant")
    a.write("Thanks for providing me the location and the moment where you want to predict regional electricity consumption. That helped us a lot. But, I still need your help with the weather indicators.")

    # max_tem
    a.write("What will be ***Maximum Temperature*** readings for P-day? 🌡🥵")
    max_tem = st.text_input(
            "Please enter numbers only",
            "31.05"
        )
    max_tem=float(max_tem)
        
    # Snow
    a = st.chat_message("assistant")
    a.write("""Do you know ***Snow Quantity*** on P-day? 🌨️ 
            If it will not snow, please let me know by entering 0.""")
    snow = st.text_input(
            "Please enter numbers only",
            "0."
                        )
    snow=float(snow)
    
    # Humidity
    a = st.chat_message("assistant")
    a.write("And the ***humidity*** could reach up to how many percent? 💦")
    humidity = st.text_input(
            "Please enter numbers only",
            "60"
        )
    humidity=float(humidity)

    # total_precip
    a = st.chat_message("assistant")
    a.write("The last question, What will the ***total rainfall*** be? 🌧️")

    total_precip = st.text_input(
            "Please enter numbers only",
            "0.", key='total_precip'
            )
    total_precip=float(total_precip)
    
    

    a = st.chat_message("assistant")
    a.write("...... 🤖, Please be patient for a few seconds. I'm working on the prediction.")
    a.write("...... ")
    a.write("...... ")
    a.write("According to your provied indicators, the Regional Electricity Consumption will be:")

    

    ### Train model
    df = pd.read_csv("ener_weather_1462r_new.csv")

    df['total_consum_Mwh'] = round(df['total_consum(Wh)']/1000000,0)
    df = df.drop(['total_consum(Wh)'], axis=1)

    # Create column season
    df["month"] = df["DATE"].apply(lambda x: x[5:7])

    # Add columns season
    spring = ['03','04', '05']
    summer = ['06','07', '08']
    autumn = ['09','10', '11']

    df['season'] = df['month'].apply(lambda x: 0 if x in spring
                                        else 1 if x in summer
                                        else 2 if x in autumn
                                        else 3)

    # Convert column DATE to datetime object
    df["DATE"] = pd.to_datetime(df["DATE"])
    df['day_name'] = df['DATE'].apply(lambda x: x.strftime('%a'))
    # Change datetime object to string
    df["DATE"] = df["DATE"].apply(lambda x: x.strftime( "%Y-%m-%d"))

    # Function to identify school vacation zone B
    def vacation(x):
        if (
                    '2022-04-09'<x<'2022-04-24'
                ) or (
                    '2022-07-07'<x<'2022-08-31' 
                ) or (
                    '2022-10-22'<x<'2022-11-06'
                ) or (
                    '2022-12-16'<x<'2023-01-02' 
                ) or (
                    '2023-02-11'<x<'2023-02-26'
                ) or (
                    '2023-04-15'<x<'2023-05-01' 
                ) or (
                    '2023-07-08'<x<'2023-09-03'
                ) or (
                    '2023-10-21'<x<'2023-11-05'
                ) or (
                    '2023-12-23'<x<='2024-01-07'
                ) or (
                    '2024-02-24'<x<'2024-03-01'
                ): 
            
                return 3
        else:
                return 0

    df["hol_weekend_vac"] = df["DATE"].apply(lambda x: vacation(x))

    # Weekend
    weekend = ["Sat", "Sun"]

    df.loc[df["day_name"].isin(weekend), "hol_weekend_vac"] = 1

    # Setup holiday days
    jour_ferie = ["2022-04-18", "2022-05-01", "2022-05-26", "2022-06-06", "2022-07-14",
                "2022-08-15", "2022-11-01", "2022-11-11", "2022-12-25", "2023-01-01",
                "2023-04-10", "2023-05-01", "2023-05-08", "2023-05-18", "2023-05-29", 
                "2023-07-14", "2023-08-15", "2023-11-01", "2023-11-11", "2023-12-25", 
                "2024-01-01", "2024-04-01"]

    df.loc[df["DATE"].isin(jour_ferie), "hol_weekend_vac"] = 2

    X = df[[
         'hol_weekend_vac', 'region_code', 'season', 'snow',
         'humidity', 'max_tem','total_precip'
          ]]

    y = df['total_consum_Mwh']

    # Split and standardize X
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.8)

    # Standardize X 
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    user_input = [hol_code, region_code, season_code, snow, humidity,  max_tem, total_precip]
    
    user_input_scaled = scaler.transform([user_input])

    ### Apply hyperparams with n_features=7

    modelDTR = DecisionTreeRegressor(min_samples_split = 15, min_samples_leaf = 1, max_depth = 8, random_state=42)
    modelDTR.fit(X_train_scaled, y_train)

    user_predict = modelDTR.predict(user_input_scaled)

    result = round(user_predict[0], 2)
 
    st.write(result, "Mwh")

    a = st.chat_message("assistant")
    a.write("Thanks you for using my application. I hope you will have another nice day as usual 🤖.")

    
