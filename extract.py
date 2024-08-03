import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

def intro():
    st.title(":violet[Regional Electricity Consumption Prediction System]")
    st.header("Introduction of the project, location of analysis and datasets", divider='rainbow')

    st.markdown("""
                  As part of my training at Wild Code School, with the very enthusiastic support from my coach, Mathieu Procis,  my classmate Alexandre and I implented this project which  is about the electricity consumption and how weather affects it in 2 specific region of France: Centre-Val de Loire and Hauts-de-France. In this project, we just work on a small part of Enedis's dataset which has the power allows up to 36Kwh.
                """)
    st.markdown(" ")
    # Create a folium map
    dic = {'region': ["France", "Hauts-de-France", "Centre-Val de Loire"],
           "long_lat": [[46.2276, 2.2137], [49.6636, 2.5281], [47.7516, 1.6751]]}
    map = pd.DataFrame(dic)

    m = folium.Map(location=map["long_lat"][0], zoom_start=5)

    for ll in range(1,3):
        folium.Marker(
        location=map.loc[ll,'long_lat'],
        popup= map.loc[ll, 'region'],
        icon=folium.Icon(color='purple', icon='fa fa-flag', prefix='fa')).add_to(m)
    
    st_data = st_folium(m, height=500, width=700)
    change_font = '<p style="font-family:Corbel; color:#5d0076; font-size: 18px;">(Location of two regions in this project)</p>'
    st.markdown(change_font, unsafe_allow_html=True)
    

    st.markdown("""
         
        Special thanks to opendata of enedis.fr and historique-meteo.net, we could collect and exploit these datasets for our learning and training practice.  
                
        Requirement for the final product is a regional electricity consumption prediction system in general terms, which returns user's amount of regional electricity consumption per day when user inputs certain important weather indicators. At the beginning, we use all the weather features to explain our target variable (electricity consumption), then we tried to reduire them until the ideal numbers to achive a highe enough score while using minimum number components of features. In the framework of this project, we don't need to pay attention to the accuracy of the predicted result, but it is important to explain the selection of features for the model.
                
        After the first model using DecisionTreeRegressor, we also trained another one requiring more detail in the sense of separating the user's profile. We will ask user to input more information as contract's registration power range, season,... to return a more individual prediction to user. Despite we can not get high score due to information missing but we tried to explore a little bit to understand the importance of some feature that we ignored in our global models.
        """)
    st.subheader("Implementation steps")
    st.markdown("""
        The prediction system creation is implemented in 4 principal steps:
        1. Extract data from mentioned sources
        2. Combine and Explorate (Analyze) all the datasets
        3. Clean and feature data in order to prepare for ML
        4. Train and evaluate models (standardize, PCA, RandomSearchCV,...)
                """)
    st.subheader("About the datasets:")          
    st.markdown("""    
        - 2 datasets about total electricity consumption every half hour of 2 regions from 4/2022 to 3/2024, which provides us condensed and macro information of the regional electricity consumption. 
        - 33 datasets about the weather indicators of 11 departments in 2 regions.
                """)
    

def ex():
    st.title(":violet[Data Extraction and Traitement]")
    st.header("Weather datasets", divider='rainbow')

    code = '''
                # Import the packages
                import pandas as pd
                import numpy as np
                
                # Create a function to combine 33 weather files
                def concat_weather(years, lst_city):
                    df_total = pd.DataFrame()
                    
                    for c in lst_city:
                        city = str(c)
                        for y in years:
                        
                            year = str(y)

                            # Define the path 
                            path = "export-" + city + year + ".csv"

                            # Import the file and skip 3 first rows
                            df = pd.read_csv(path, sep=',', skiprows=3)

                            # change columns name to reduce the size of columns
                            df_year = df.rename(columns = {"MAX_TEMPERATURE_C": "max_tem",
                                                    "MIN_TEMPERATURE_C": "min_tem",
                                                        "PRECIP_TOTAL_DAY_MM": "total_precip",
                                                        'TEMPERATURE_MORNING_C': "morning_tem",
                                                        'TEMPERATURE_NOON_C': "noon_tem", 
                                                        'TEMPERATURE_EVENING_C': "evening_tem",                                         
                                                        'HEATINDEX_MAX_C': "heat_index", 
                                                        'WINDTEMP_MAX_C': "wind_tem", 
                                                        'TEMPERATURE_NIGHT_C': "night_tem",
                                                        'WINDSPEED_MAX_KMH': "wind_speed",
                                                        'HUMIDITY_MAX_PERCENT': "humidity",
                                                        'TOTAL_SNOW_MM': "snow",
                                                        'VISIBILITY_AVG_KM': "visibility",
                                                        })

                            # Concatenate them
                            df_total = pd.concat([df_total, df_year])


                    # Return list of necessary columns = only number
                    lst_col = df_total.iloc[:,1:25].select_dtypes('number').columns.to_list()

                    # Groupby for daily unique value
                    df_total = df_total.groupby('DATE')[lst_col].mean().reset_index()

                    # Filter for date
                    df_total = df_total.loc[(df_total["DATE"] > "2022-03-31") & (df_total["DATE"] < "2024-04-01")]
                    
                    return df_total
                '''
    st.code(code, language='python')

    code1 = '''
            # Apply the function 
            lst_year = ['2022', '2023', '2024']

            # HDF
            lst_hdf = ['lille', 'amiens', 'beauvais-oise', 'laon-aisne', 'arras']
            hdf_w = concat_weather(lst_year, lst_hdf)

            # CVDL
            lst_cvdl = ['chartres', 'blois', 'bourges', 'tours', 'chateauroux', 'orleans']
            cvdl_w = concat_weather(lst_year, lst_cvdl)

                '''
    st.code(code1, language='python')

    st.header("Enedis's datasets", divider='rainbow')

    code2 = '''
            # Import csv file of Centre-Val de Loire
            df = pd.read_csv("val_de_loire.csv", sep=";", 
                            usecols=["Horodate", "Code région", "Profil", "Plage de puissance souscrite",
                                    "Nb points soutirage", "Total énergie soutirée (Wh)"])

            # Drop NaN
            df = df.loc[df["Total énergie soutirée (Wh)"].notnull()]
            df = df.loc[df['Plage de puissance souscrite']!='P0: Total <= 36 kVA']

            # Add columns Date, year, month, week
            df["DATE"] = df["Horodate"].apply(lambda x: x[:10])

            # Change columns name to code faster
            df = df.rename(columns = {"Code région": "region_code",
                                    "Profil": "profile",
                                    "Nb points soutirage": "total_point",
                                    "Total énergie soutirée (Wh)": "total_consum(Wh)",
                                    "Plage de puissance souscrite": "power_range"})

            # Group by and sum by date and regional code
            df = df.groupby(["DATE", "region_code"])["total_consum(Wh)"].sum().reset_index()

            cvdl = df.copy()
                '''
    st.code(code2, language='python')

    code3 = '''
            # Import csv file of Hauts-de-France
            df = pd.read_csv("hauts_de_france.csv", sep=";", 
                            usecols=["Horodate", "Code région", "Profil", "Plage de puissance souscrite",
                                    "Nb points soutirage", "Total énergie soutirée (Wh)"])

            # Drop NaN
            df = df.loc[df["Total énergie soutirée (Wh)"].notnull()]
            df = df.loc[df['Plage de puissance souscrite']!='P0: Total <= 36 kVA']

            # Add columns Date, year, month, week
            df["DATE"] = df["Horodate"].apply(lambda x: x[:10])

            # Change columns name to code faster
            df = df.rename(columns = {"Code région": "region_code",
                                    "Profil": "profile",
                                    "Nb points soutirage": "total_point",
                                    "Total énergie soutirée (Wh)": "total_consum(Wh)",
                                    "Plage de puissance souscrite": "power_range"})

            # Group by and sum by date and regional code
            df = df.groupby(["DATE", "region_code"])["total_consum(Wh)"].sum().reset_index()

            hdf = df.copy()
                '''
    st.code(code3, language='python')

    st.header('Combine all for final dataset', divider='rainbow')
    code4 = '''
            # Merge to weather
            cvdl =  pd.merge(cvdl, cvdl_w, how='left', on='DATE')
            hdf =  pd.merge(hdf, hdf_w, how='left', on='DATE')

            # Concat 2df energy+weather for a full dataset
            two_region = pd.concat([cvdl, hdf])

            print(two_region)
                '''
    st.code(code4, language='python')

    df = pd.read_csv(r"ener_weather_1462r_new.csv")

    st.write(df)






       
