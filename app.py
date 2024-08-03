import streamlit as st
from eda import bar_line, season_hol, corr_weather, snow_humid, rain, sunhour
from extract import intro, ex
from DS import model
from ML import predict


# Background
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
                                            background-color: #ffee78 !important;
                                            }}
[data-testid="stHeader"] {{
                        background: rgba(0,0,0,0);
                        }}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


# Sidebar's background 
st.markdown("""
<style>
    [data-testid="stSidebar"] {
                                background-image: url("https://img.freepik.com/free-photo/copy-space-paper-ligh-bulb_23-2148519480.jpg?t=st=1720197016~exp=1720200616~hmac=1588127bcac57e9f3416101fc19c457a27f0a130a01bda1c9d3bb84cdd2c06ce&w=2000");
                                background-size: cover;
                                background-position: left;
                                }                           
</style>
""", unsafe_allow_html=True)


# Make bigger and colored
#change_font = '<p style="font-family:Courier; color:Blue; font-size: 20px;">In case want to change color of title</p>'
#st.markdown(change_font, unsafe_allow_html=True)


def main():
         
    menu = ["Introduction", "Data Extraction", "Data Analysis", "Machine Learning","Consumption Prediction"]

    choice = st.sidebar.selectbox("MENU", menu)

    if choice == "Introduction":
        intro()

    if choice == "Data Extraction":
        ex()

    if choice == "Data Analysis":
        

        st.title(":violet[Regional Consupmtion Analysis]")
        st.markdown("(Note: Dataset of all the profiles which have power allows less than 36Kwh.) ")
        st.markdown(" ")
        st.image(r"asset/intro.png")
        st.markdown("Source: enedis.fr")
        change_font = '<p style="font-family:Corbel; color:#5d0076; font-size: 25px;">DID YOU KNOW? 😎</p>'
        st.markdown(change_font, unsafe_allow_html=True)
        change_font = '<p style="font-family:Corbel; color:#5d0076; font-size: 18px;">"The average annual consumption of electricity and gas per household in France is respectively 4.5MWh and 9.8MWh in 2022." (enedis.fr)</p>'
        st.markdown(change_font, unsafe_allow_html=True)
        st.markdown("")

        
        # Basic info about 2 region
        st.subheader("🔹Hauts-de-France")
        st.markdown("""
        Hauts-de-France is the northernmost region of France. With 5,709,571 inhabitants as of 2018 and a population density of 189 inhabitants per km2, it is the third most populous region in France and the second-most densely populated in metropolitan France after its southern neighbour Île-de-France. 
                    """)
        st.subheader("🔹Centre-Val de Loire")
        st.markdown("""
        Centre-Val de Loire or Centre Region, as it was known until 2015, is one of the eighteen administrative regions of France. It straddles the middle Loire Valley in the interior of the country, with a population of 2,380,990 as of 2018. Its prefecture is Orléans, and its largest city is Tours.
                            """) 
       
    
        ### Graphiques of introduction ###

        st.markdown(" ")
        st.header("General information", divider='rainbow')
        st.markdown(" ")

        

        st.markdown("""
                - The pie chart gives us a brief visualization of 3 principal categories of all the profiles which occurence in the dataset. So, the profile 'Residence' accounts more than 50% of the dataset, ortherwise the 'Company' just plays a small role in the regional consumption.
                                             """) 
        st.markdown(" ")
        st.image(r"asset/segment.png")
        st.markdown(" ")

        st.markdown("""
            - The bar chart gives us an overview of total average daily electricity consumption of the two regions. The purpose is not to compare the indicators of 2 areas but to observe and analyze the elements which could be important for our model after. 
        
            - The region HDF has 3,3 millions more inhabitants  than the region CVDL (2,39 times) but its total electricity consumption is 1.8 times higher than CVDL's.  In the other hand, HDF consums 7% less electricity for heating than CVDL. From this perspective, the size of house/ appartement, type of heating energy should be considered as very important factors in electricity consumption.         
                    """)


        bar_line()    

        st.markdown(" ")
        st.header("Time Factor Analysis", divider='rainbow')
        st.markdown(" ")
        
        st.markdown("""
                - The seasonal line charts explains very well the impact of seasonal elements on electricity consumption. More than that, we can easily guess that the temperature always palys un important role in human energy consumtion. 
                - Apparently, consumption tendency  decreases as summer approaches and increases as the weather turns colder.       
                """)
        st.markdown(" ")
        season_hol()
        
        st.markdown(" ")
        st.header("Weather Factor Analysis", divider='rainbow') 
        st.markdown(" ")
        st.subheader("🔹Correlation Score of all features")
        st.markdown("""
            This graphic allows us to identify the groups of features in the dataset, which will help us to filter the necessary feature for our model.
            
            The correlation score between each pair of features explains somehow their positive or negative causal relationship. In this sence, the group having red color is identified 'temperature' group, and the grey one is 'weather_code' group. So, the rest are individual variables that we need to alanyze in order to see if they have big influence on the electricity consumption or not.     
                    """)
        corr_weather()

        st.markdown(" ")
        st.subheader("🔹Snowfall and Humidity Influence")
        st.markdown("""
                Since the indicators of Humidity and Snowfall are very sporadic and discreet, we can not observe clearly the variance of the consumption influenced by them. So we tried to change the type of data from number to booleans. 
                - The average consumption on a normal day and a snowy day.  We observe a big difference in the consumption behavior.
                - Knowing that people feel comfortable when the humidity is 40-70%, and we feel bad when this indicator is higher than 70%. So we cut the value of this column at the threshold of 70%. The average consumption with comfortable and uncomfortable humidity .  
                    """)
        snow_humid()

        st.markdown(" ")
        st.subheader("🔹Rainfall Influence")
        st.markdown("""
            Observing that there is not much rain during 24 months in both regions, we decided to classify data into boolean types: Rain/No Rain to compare the consumption on a normal day and a rainy day. Apparently, rainfall does not have much effect on electricity consumption.
                """)
        st.markdown(" ")
        rain()

        #st.markdown(" ")
        #st.subheader("🔹Sunhour Influence")
        #sunhour()

    if choice == "Machine Learning":
        st.title(":violet[Machine Learning Session]")
        st.markdown(" ")
        st.markdown("""
                    This session is to explain our ML workflow and how did we filter all the variables to obtain the last product. 
                    """)
        model()

    if choice == "Consumption Prediction":
        st.title(":violet[Prediction Session]")
        st.markdown(" ")
        predict()
        
            



main()