import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_palette('Paired')
sns.set_style('whitegrid')

### Import consumption data ###
df = pd.read_csv(r"ener_weather_1462r_new.csv")


df['total_consum_Mwh'] = round(df['total_consum(Wh)']/1000000,0)
df = df.drop(['total_consum(Wh)'], axis=1)

# Create column season
df["month"] = df["DATE"].apply(lambda x: x[5:7])

# Add columns season
spring = ['03','04', '05']
summer = ['06','07', '08']
autumn = ['09','10', '11']

df['season'] = df['month'].apply(lambda x: "Spring" if x in spring
                                      else "Summer" if x in summer
                                      else "Autumn" if x in autumn
                                      else "Winter")

### Create column to identify the weekend, holiday and vacation in zone B of France
# Here, we apply a hierarchy to all days off: the priority is national holidays, next rank is weekend then school holidays.

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
       
        return "School Holidays"
   else:
        return "Normal Day"

df["hol_weekend_vac"] = df["DATE"].apply(lambda x: vacation(x))

# Weekend
weekend = ["Sat", "Sun"]

df.loc[df["day_name"].isin(weekend), "hol_weekend_vac"] = "Week-end"

# Setup holiday days
jour_ferie = ["2022-04-18", "2022-05-01", "2022-05-26", "2022-06-06", "2022-07-14",
              "2022-08-15", "2022-11-01", "2022-11-11", "2022-12-25", "2023-01-01",
              "2023-04-10", "2023-05-01", "2023-05-08", "2023-05-18", "2023-05-29", 
              "2023-07-14", "2023-08-15", "2023-11-01", "2023-11-11", "2023-12-25", 
              "2024-01-01", "2024-04-01"]

df.loc[df["DATE"].isin(jour_ferie), "hol_weekend_vac"] = "National Holidays"



df['DATE'] = pd.to_datetime(df['DATE'])

df["region_code"] = np.where(df["region_code"]==24, "Centre-Val de Loire", "Hauts-de-France")

# Set up a dictionary of font title
font_title = {'fontfamily': 'sans-serif',
              'color':  '#114b98',
              'fontweight': 'bold'}

# Palette for hue of season
palette = {"Spring": "#a95aa1", "Summer": "#85c0f9",
            "Autumn": "#f5793a", "Winter": "#0f2080"}


def bar_line():

    # Set the size of chart
    f, ax = plt.subplots( figsize=(12, 8))

    # Chart of total consumption
    ax = sns.barplot(data = df, y = "total_consum_Mwh",
                x = "region_code", 
                errorbar=None)

    ax.set_title("Regional AVG Total Consumption Per day", pad=10, fontsize = 20, fontdict=font_title)
    ax.set_xlabel("Region", fontweight ='bold' )
    ax.set_ylabel("AVG Total Consumption (MWh)", fontweight ='bold')

    for container in ax.containers:
        ax.bar_label(container, label_type="center", fmt="{:,.0f} MWH",
                        color="black", fontsize=15, fontweight ='bold')
        
    st.pyplot(f) 

    st.markdown(" ")
    st.markdown("""
              - Vice versa, the line chart shows the total consumption by timeline from 04/2022 to 04/2024, which allows to observe the evolution of  consumption more in detail. 
                """)
    # Chart of consumption evolution
    f1, ax1 = plt.subplots(figsize=(12,8 ))
    palette_region = {"Centre-Val de Loire": "#85c0f9", "Hauts-de-France": "#0f2080"}
    ax1 = sns.lineplot(df, x = 'DATE', y = 'total_consum_Mwh', hue = "region_code", style="region_code", palette = palette_region)
    ax1.set_title("Regional Consumption Curve in 24 Months", pad=10, fontsize = 20, fontdict=font_title)
    ax1.set_xlabel("Representative month", fontweight ='bold' )
    ax1.set_ylabel("Total consumption (MWh)", fontweight ='bold')
    ax1.legend(title="Region")

    st.pyplot(f1)  

def season_hol():
     
     # Setup df for season
    season = df.loc[(df["DATE"] >= "2023-03-01") & (df["DATE"] < "2024-03-01")]

   
    # Define the category for subplot: nCat = nSubplot
    n_sub = len(season['region_code'].unique())
    n_col = 1

    fig, axes = plt.subplots(int(n_sub/n_col), n_col, sharex=False, sharey=False, figsize=(16,12))
    axes = np.array(axes)

    fig.suptitle("Seasonal Consumption Trend by Region", fontsize = 30, fontdict=font_title)

    i=0
    for ax in axes.reshape(-1):
        cat = season['region_code'].unique()[i]
        subset = season[season['region_code']==cat].sort_values("DATE")
        sns.lineplot(data=subset, y='total_consum_Mwh', x='DATE', ax=ax, 
                     style='season', hue = "season", palette=palette)
        ax.set_title('Region : {}'.format(cat), pad=8, loc='left')
        ax.set_xlabel(" ")
        ax.set_ylabel("Total Consumption (MWh)")
        i+=1
    st.pyplot(fig)

    st.markdown(" ")
    st.markdown(""" 
                If we tranche the dataset by holidays working days and kid vacations in order to compare the average consumption on each type of date,  people in two region have tendency consume more electircity on working days and weekend in spring and winter, but in autumn national holidays are the dates where they use more electricity. 
                """)
    st.markdown(" ")
    ## Holidays

    n_sub = len(df['region_code'].unique())
    n_col = 1

    fig2, axes = plt.subplots(int(n_sub/n_col), n_col, sharex=False, sharey=False, figsize=(16,8))
    axes = np.array(axes)

    fig2.suptitle("Distribution of Electricity Consumption by Date type and Season", fontdict=font_title, fontsize = 30)

    i=0
    for ax in axes.reshape(-1):
        cat = df['region_code'].unique()[i]
        subset = df[df['region_code']==cat].sort_values("season")
        sns.barplot(data=subset, y='total_consum_Mwh', 
                    x='hol_weekend_vac',  order=['Normal Day', 'Week-end', 'National Holidays', 'School Holidays'],
                    hue = "season", hue_order=['Spring', 'Summer', 'Autumn', 'Winter'], 
                    ax=ax, palette=palette, errorbar=None)
        ax.legend(bbox_to_anchor=(1, 1), loc = 'upper left')
        ax.set_title('Region : {}'.format(cat), pad=8, loc='left')
        ax.set_xlabel(" ")
        ax.set_ylabel("AVG Consumption (MWh)")
        i+=1

        for container in ax.containers:
            ax.bar_label(container, label_type="edge", fmt="{:,.0f}",
                        color="black", fontsize=10, fontweight ='bold')

    st.pyplot(fig2)

def corr_weather():
    
    
    # get correlation data 
    var_corr = df.corr(numeric_only = True)

    f = plt.figure(figsize=(15,10))
    sns.heatmap(var_corr,
                annot = True, cmap = 'vlag', center = 0, 
                annot_kws={'size': 9}, linecolor="white", linewidth=0.5)
    st.pyplot(f)

    ### Scatter max/noon temerature ###
    st.subheader("🔹Max and Evening Temperature")
    st.markdown("""
                The combination of max and evening temperature permits us explaining that the temperature plays an important role to electricity consumption in general terms. Especialy, when the temperature drops below 0 degree, the consumption indicators increase.
                    """)
    st.markdown(" ")
    
    df_hdf = df.loc[df['region_code'] == 'Hauts-de-France']
    df_cvdl = df.loc[df['region_code'] == 'Centre-Val de Loire']
    labels = df_cvdl["season"].unique()

    tab1, tab2 = st.tabs(["Centre-Val de Loire", "Hauts-de-France"])

    with tab1:
        
        buttonsLabels = [dict(label = "All",
                                    method = "update",
                                    visible=True,
                                    args = [
                                        {'x' : [df_cvdl.max_tem]},
                                        {'y' : [df_cvdl.noon_tem]},
                                        {'color': [df_cvdl["total_consum_Mwh"]]},
                                    ]
                                    )]

        for label in labels:
            buttonsLabels.append(dict(label = label,
                                    method = "update",
                                    visible = True,
                                    args = [
                                        {'x' : [df_cvdl.loc[df_cvdl['season'] == label, "max_tem"]]},
                                        {'y' : [df_cvdl.loc[df_cvdl['season'] == label, "evening_tem"]]},
                                        {'color' : [df_cvdl.loc[df_cvdl['season'] == label, "total_consum_Mwh"]]},
                                    ]
                                    ))

        # Display figure
        fig1 = go.Figure(px.scatter(df_cvdl, x="max_tem", y="evening_tem",
                        color="total_consum_Mwh",
                        hover_data= ["total_consum_Mwh"],
                        labels={"max_tem": "Max Temperature",
                                "evening_tem": "Evening Temperature",
                                "total_consum_Mwh": "Consum(Mwh)"
                        },
                        color_continuous_scale='turbo')
                        #range_color=[10,70])

        )

        fig1.update_layout(updatemenus = [dict(buttons = buttonsLabels, showactive = True)], 
                        margin = dict(t=50, l=25, r=25, b=25), 
                        title=dict(text="Distribution of Consumption by Max-Evening Temperature", 
                                    font=dict(size=20))            
                        )
        
        st.plotly_chart(fig1, theme="streamlit")     

    with tab2:
        
        buttonsLabels = [dict(label = "All",
                                    method = "update",
                                    visible=True,
                                    args = [
                                        {'x' : [df_hdf.max_tem]},
                                        {'y' : [df_hdf.evening_tem]},
                                        {'color': [df_hdf["total_consum_Mwh"]]},
                                    ]
                                    )]

        for label in labels:
            buttonsLabels.append(dict(label = label,
                                    method = "update",
                                    visible = True,
                                    args = [
                                        {'x' : [df_hdf.loc[df_hdf['season'] == label, "max_tem"]]},
                                        {'y' : [df_hdf.loc[df_hdf['season'] == label, "evening_tem"]]},
                                        {'color' : [df_hdf.loc[df_hdf['season'] == label, "total_consum_Mwh"]]},
                                    ]
                                    ))

        # Display figure
        fig2 = go.Figure(px.scatter(df_hdf, x="max_tem", y="evening_tem",
                        color="total_consum_Mwh",
                        hover_data= ["total_consum_Mwh"],
                        labels={"max_tem": "Max Temperature",
                                "evening_tem": "Evening Temperature",
                                "total_consum_Mwh": "Consum(Mwh)"
                        },
                        color_continuous_scale='turbo')
                        #range_color=[10,85])

        )

        fig2.update_layout(updatemenus = [dict(buttons = buttonsLabels, showactive = True)], 
                        margin = dict(t=50, l=25, r=25, b=25), 
                        title=dict(text="Distribution of Consumption by Max-Evening Temperature", 
                                    font=dict(size=20))            
                        )
        
        st.plotly_chart(fig2, theme="streamlit")  


def snow_humid():

    ### Snow ###
    # Create boolean value of column 'snow'
    df["snow_str_boo"] = np.where(df["snow"] > 0., "Snow", "No Snow")

    
    ### Humidity ###
    df['humidity'] =  round(df['humidity'],0)
    df['humidity'] = np.where(df['humidity'] <= 70, 'Suitbale Humidity', 'Humidity > 70%')

    

    # Define the category for subplot: nCat = nSubplot
    n_sub = len(df['region_code'].unique())
    n_col = 2

    tab1, tab2 = st.tabs(['Total Snowfall in MM', 'Humidity Percentage'])

    with tab1:
        fig, axes = plt.subplots( int(n_sub/n_col), n_col,  sharex=False, sharey=False, figsize=(14,7))
        axes = np.array(axes)

        fig.suptitle("Distribution of Consumption by Snowfall", fontdict=font_title, fontsize = 30)

        i=0
        for ax in axes.reshape(-1):
            cat = df['region_code'].unique()[i]
            subset = df[df['region_code']==cat].sort_values("snow")
            sns.barplot(data=subset, y='total_consum_Mwh', x='snow_str_boo', ax=ax, errorbar=None)
            ax.set_title('Region : {}'.format(cat), pad=8, loc='left')
            ax.set_xlabel(" ")
            ax.set_ylabel("AVG Consumption (MWh)")
            i+=1

            for container in ax.containers:
                ax.bar_label(container, label_type="center", fmt="{:,.0f} Mwh",
                        color="black", fontsize=12, fontweight ='bold')
        st.pyplot(fig)

  
    with tab2:

        fig1, axes = plt.subplots( int(n_sub/n_col), n_col,  sharex=False, sharey=False, figsize=(14,7))
        axes = np.array(axes)

        fig1.suptitle("Distribution of Consumption by Humidity", fontdict=font_title, fontsize = 30)

        i=0
        for ax in axes.reshape(-1):
            cat = df['region_code'].unique()[i]
            subset = df[df['region_code']==cat].sort_values("humidity", ascending=False)
            sns.barplot(data=subset, y='total_consum_Mwh', x='humidity', ax=ax, errorbar=None)
            ax.set_title('Region : {}'.format(cat), pad=8, loc='left')
            ax.set_xlabel(" ")
            ax.set_ylabel("AVG Consumption (MWh)")
            i+=1

            for container in ax.containers:
                ax.bar_label(container, label_type="center", fmt="{:,.0f} Mwh",
                        color="black", fontsize=12, fontweight ='bold')
        st.pyplot(fig1)



def rain():

    ### Rain ###
    
    # Create boolean value of column 'snow'
    df["rain_str_boo"] = np.where(df["total_precip"] > 0., "Rain", "No Rain")

    # Define the category for subplot: nCat = nSubplot
    n_sub = len(df['region_code'].unique())
    n_col = 2

    fig3, axes = plt.subplots( int(n_sub/n_col), n_col,  sharex=False, sharey=False, figsize=(14,7))
    axes = np.array(axes)

    fig3.suptitle("Distribution of Consumption by Rainfall", fontdict=font_title, fontsize = 30)

    i=0
    for ax in axes.reshape(-1):
        cat = df['region_code'].unique()[i]
        subset = df[df['region_code']==cat].sort_values("total_precip")
        sns.barplot(data=subset, y='total_consum_Mwh', x='rain_str_boo', ax=ax, errorbar=None)
        ax.set_title('Region : {}'.format(cat), pad=8, loc='left')
        ax.set_xlabel(" ")
        ax.set_ylabel("AVG Consumption (MWh)")
        i+=1

        for container in ax.containers:
            ax.bar_label(container, label_type="center", fmt="{:,.0f} Mwh",
                    color="black", fontsize=12, fontweight ='bold')
    st.pyplot(fig3)



    ### SUNHOUR ###

def sunhour():
       
    # Define the category for subplot: nCat = nSubplot
    n_sub = len(df['region_code'].unique())
    n_col = 1

    fig1, axes = plt.subplots( n_col, int(n_sub/n_col), sharex=False, sharey=False, figsize=(14,7))
    axes = np.array(axes)

    fig1.suptitle("Distribution of Consumption by SUNHOUR", fontdict=font_title, fontsize = 30)

    i=0
    for ax in axes.reshape(-1):
        cat = df['region_code'].unique()[i]
        subset = df[df['region_code']==cat].sort_values("DATE")
        sns.scatterplot(data=subset, y='total_consum_Mwh', x='SUNHOUR', ax=ax, palette=palette, hue='season', style='season')
        ax.set_title('Region : {}'.format(cat), pad=8, loc='left')
        ax.set_xlabel("Sunhour")
        ax.set_ylabel("Consumption (MWh)")
        i+=1
    st.pyplot(fig1)


    # Define the category for subplot: nCat = nSubplot
    n_sub = len(df['region_code'].unique())
    n_col = 1

    fig2, axes = plt.subplots( n_col, int(n_sub/n_col), sharex=False, sharey=False, figsize=(14,7))
    axes = np.array(axes)

    fig2.suptitle("Distribution of Max Temperature and SUNHOUR", fontdict=font_title, fontsize = 30)

    i=0
    for ax in axes.reshape(-1):
        cat = df['region_code'].unique()[i]
        subset = df[df['region_code']==cat].sort_values("DATE")
        sns.scatterplot(data=subset, y='max_tem', x='SUNHOUR', ax=ax, palette=palette, hue='season', style='season')
        ax.set_title('Region : {}'.format(cat), pad=8, loc='left')
        ax.set_xlabel("Sunhour")
        ax.set_ylabel("Max Temperature")
        i+=1
    st.pyplot(fig2)






     








    














                

