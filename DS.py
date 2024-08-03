import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


sns.set_palette('Paired')
sns.set_style('whitegrid')

def model():
    df = pd.read_csv(r"ener_weather_1462r_new.csv")
    

    # Change scale from Wh to Mwh
    df['total_consum_Mwh'] = round(df['total_consum(Wh)']/1000000,0)

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
    df = df.drop(['total_consum(Wh)', 'month'], axis=1)

    # Display DF
    st.header("Overview of Dataframe", divider='rainbow')
    st.markdown(' ')
    st.markdown("""
                - 1462 rows equivalent to 365 days x 2 years x 2 regions
                - Target variable: 'total_consum_Mwh'
                - Explanatory variables are all the columns except of 'DATE' and target column
                """)
    st.markdown(' ')

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

    st.markdown(" ")
    st.markdown("This is the dataset after the step Feature Engineering")
    st.write(df)


    ### Check Correlation between variables ###
    st.header("Check Correlation between variables", divider='rainbow')
    # get correlation data 
    var_corr = df.corr(numeric_only = True)

    # build a matrix of booleans with same size and shape of var_corr
    ones_corr = np.ones_like(var_corr, dtype=bool)

    # create a mask to take only the half values of the table by using np.triu
    mask = np.triu(ones_corr)

    # adjust the mask and correlation data  to remove the 1st row and last column
    adjusted_mask = mask[1:,:-1]
    adjusted_corr = var_corr.iloc[1:,:-1]

    fig = plt.figure(figsize=(15,10))
    sns.heatmap(adjusted_corr, mask = adjusted_mask,
                annot = True, cmap = 'vlag', center = 0, 
                annot_kws={'size': 7}, linecolor="white", linewidth=0.5)
    st.pyplot(fig) 


    st.header("ML PIPELINE", divider='rainbow')
    st.image(r"asset/pipeline.png")
    st.markdown('(Source: internet)')
    st.markdown(' ')
    st.subheader('Machine Learning workflow step by step:')
    st.markdown("""
                0. Extract and preprocess data
                1. Initialze X (explanatory variables), y (target variable)
                2. Split train, test then standardize X_train, X_test
                3. Train 2 models: LR, DTR and evaluate models using merics score on test/train set, RMSE
                4. Apply PCA to reduce not very important demensions (explanatory variables)
                5. Get columns names of important components then train models again with new numbers of explanatory variables
                6. Apply RandommizedSearchCV to get good hyperparameters for model DTR. Then, train DTR again tuning hyperparameters
                7. Final evaluation: vizualize test-score and RMSE on bar chart to compare and choose final model
                    """)
    

    st.header("Metrics vizualization", divider='rainbow')
    # Create df of models with theirs scores

    dico_models = {
                    'model': ['LinearRegression', 'DecisionTreeRegressor', 'DecisionTreeRegressor with params', 
                            'LinearRegression', 'DecisionTreeRegressor', 'DecisionTreeRegressor with params', 
                            'LinearRegression', 'DecisionTreeRegressor', 'DecisionTreeRegressor with params',
                            'LinearRegression', 'DecisionTreeRegressor', 'DecisionTreeRegressor with params'],
                    'n_components': [24, 24, 24, 13, 13, 13, 8, 8, 8, 7, 7, 7],
                    'score_test': [0.901, 0.938, 0.945, 
                                0.880, 0.925, 0.939, 
                                0.877, 0.934, 0.948,
                                0.876, 0.935, 0.956],
                    'RMSE (Mwh)': [4343, 3426, 3240, 
                                4784, 3776, 3415,
                                4838, 3535, 3141,
                                4864, 3517, 2884]
                }

    df_models = pd.DataFrame(dico_models)

    fig2, ax = plt.subplots(2, 1, figsize=(12, 8))

    sns.barplot(df_models, x = 'n_components', y = 'score_test', hue = 'model', ax=ax[0], order=[24,13,8,7])
    sns.move_legend(ax[0], "upper left", bbox_to_anchor=(1, 1))
    ax[0].set_title("Test-set Score by Num of Features and Model", fontweight ='bold', pad=8, loc='left')
    ax[0].set_xlabel(" ", fontweight ='bold' )
    ax[0].set_ylabel("Accuracy score on Test set")

    for container in ax[0].containers:
        ax[0].bar_label(container, label_type="edge",
                        color="r", fontsize=8)


    sns.barplot(df_models, x = 'n_components', y = 'RMSE (Mwh)', hue = 'model', ax=ax[1], order=[24,13,8,7])
    ax[1].legend_.remove()
    ax[1].set_title("RMSE (Mwh) by Num of Features and Model", fontweight ='bold', pad=8, loc='left')
    ax[1].set_xlabel("Num of Features" )
    ax[1].set_ylabel("Root Mean Squared Error")

    for container in ax[1].containers:
        ax[1].bar_label(container, label_type="edge",
                        color="r", fontsize=8)
        
    fig2.suptitle("METRICS EVALUATION", fontsize = 25, fontweight ='bold')
    st.pyplot(fig2)

    st.markdown(" ")
    st.title(":violet[Detailed Explanation]")
    
    df1 = df.copy()

    st.header("Train and fit X with 24 features", divider='rainbow')
    
    # Initialize X, y
    # 1st reound: X has 24 features
    X = df1[['region_code','max_tem','min_tem','wind_speed','morning_tem','noon_tem',
             'evening_tem','total_precip', 'humidity', 'visibility','PRESSURE_MAX_MB',
             'CLOUDCOVER_AVG_PERCENT','heat_index','DEWPOINT_MAX_C','wind_tem',
            'WEATHER_CODE_MORNING', 'WEATHER_CODE_NOON','WEATHER_CODE_EVENING', 'snow', 
            'UV_INDEX','SUNHOUR','night_tem','season', 'hol_weekend_vac']]

    y = df1['total_consum_Mwh']

    # Split and standardize X
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.8)

    # Standardize X 
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    code = '''
                # Initialize X, y
                # 1st reound: X has 24 features
                X = df1[['region_code','max_tem','min_tem','wind_speed','morning_tem','noon_tem',
                        'evening_tem','total_precip', 'humidity', 'visibility','PRESSURE_MAX_MB',
                        'CLOUDCOVER_AVG_PERCENT','heat_index','DEWPOINT_MAX_C','wind_tem',
                        'WEATHER_CODE_MORNING', 'WEATHER_CODE_NOON','WEATHER_CODE_EVENING',  
                        'snow','UV_INDEX','SUNHOUR','night_tem','season', 'hol_weekend_vac']]

                y = df1['total_consum_Mwh']

                # Split and standardize X
                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, 
                                                                        train_size=0.8)

                # Standardize X 
                scaler = StandardScaler().fit(X_train)
                X_train_scaled = scaler.transform(X_train)
                X_test_scaled = scaler.transform(X_test)
        
                '''
    st.subheader("Initialize-split X,y and standardize data")
    st.code(code, language='python')

    ### Create a function for model training ###

    def train_model(n_feature, X_train_scaled, y_train, X_test_scaled, y_test):
        ### Linear Regression ###
        modelLR = LinearRegression().fit(X_train_scaled, y_train)
        y_test_predictLR = modelLR.predict(X_test_scaled)

        st.write("METRICS for MODEL Linear Regression with", n_feature, "features")
        st.write("\nScore for the Train-set :", modelLR.score(X_train_scaled, y_train))
        st.write("\nScore for the Test-set :", modelLR.score(X_test_scaled, y_test))
        st.write('\nRMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predictLR)), 'Mwh')

        ### DecisionTreeGressor ###
        # Without hyperparams
        modelDTR = DecisionTreeRegressor(random_state=42)
        modelDTR.fit(X_train_scaled, y_train)

        y_test_predictDTR = modelDTR.predict(X_test_scaled)
        st.write("\n\nMETRICS for MODEL DecisionTreeRegressor with", n_feature, "features")
        st.write("\nScore for the Train-set :", modelDTR.score(X_train_scaled, y_train))
        st.write("\nScore for the Test-set :", modelDTR.score(X_test_scaled, y_test))
        st.write('\nRMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predictDTR)), 'Mwh')

    model_train = '''
        def train_model(n_feature, X_train_scaled, y_train, X_test_scaled, y_test):
            ### Linear Regression ###
            modelLR = LinearRegression().fit(X_train_scaled, y_train)
            y_test_predictLR = modelLR.predict(X_test_scaled)

            print("METRICS for MODEL Linear Regression with", n_feature, "features")
            print("Score for the Train-set :", modelLR.score(X_train_scaled, y_train))
            print("Score for the Test-set :", modelLR.score(X_test_scaled, y_test))
            print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predictLR)), 'Mwh')

            ### DecisionTreeGressor ###
            # Without hyperparams
            modelDTR = DecisionTreeRegressor(random_state=42)
            modelDTR.fit(X_train_scaled, y_train)
            y_test_predictDTR = modelDTR.predict(X_test_scaled)

            print("METRICS for MODEL DecisionTreeRegressor with", n_feature, "features")
            print("Score for the Train-set :", modelDTR.score(X_train_scaled, y_train))
            print("Score for the Test-set :", modelDTR.score(X_test_scaled, y_test))
            print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predictDTR)), 'Mwh')
        
                '''
    st.subheader("Create a function for model training")
    st.code(model_train, language='python')

    st.subheader("Metrics using 24 features")
    train_model(24, X_train_scaled, y_train, X_test_scaled, y_test)

    
    ### DecisionTreeGressor ###
    # With hyperparams
    st.subheader("Decision Tree Regressor with Hyperparameters tuning ")
    modelDTR = DecisionTreeRegressor(min_samples_split = 3, min_samples_leaf = 8, max_depth = 16, random_state=42)
    modelDTR.fit(X_train_scaled, y_train)

    y_test_predict = modelDTR.predict(X_test_scaled)

    st.write("\nScore for the Train-set :", modelDTR.score(X_train_scaled, y_train))
    st.write("\nScore for the Test-set :", modelDTR.score(X_test_scaled, y_test))
    st.write('\nRMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)), 'Mwh')

    st.header("Apply PCA to reduce dimensions", divider='rainbow')

    # Initialze CPA to find the n_components
    pca = PCA().fit(X_train_scaled)
    
    code_pca = '''
                # Initialize CPA to find the n_components
                pca = PCA().fit(X_train_scaled)
                pca.explained_variance_ratio_                
                    '''
    st.code(code_pca, language='python')
    
    st.write("The ratio of variance explained for each of the new dimensions:")
    st.write('''
                array([4.51049596e-01, 1.95768925e-01, 5.56937422e-02, 4.78274602e-02,
                4.62318867e-02, 4.12659287e-02, 2.72215019e-02, 2.60161248e-02,
                2.25145404e-02, 2.06654975e-02, 1.86878139e-02, 1.65270679e-02,
                1.10700210e-02, 6.54114745e-03, 4.69892872e-03, 3.85676352e-03,
                2.02685927e-03, 6.88213298e-04, 5.78587316e-04, 4.89023372e-04,
                3.27246555e-04, 1.58630930e-04, 9.44941697e-05])
             ''')
    
    st.markdown(" ")
    st.subheader('Project into a line chart to see how n_components explaines variance of dataset')
    # Plot to see how n_components explaines variance
    plt.rcParams["figure.figsize"] = (12,6)

    fig1, ax = plt.subplots()
    xi = np.arange(1, 25, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)

    plt.ylim(0.0,1.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')

    plt.xlabel('Number of Components')
    plt.xticks(np.arange(0, 25, step=1)) #change from 0-based array index to 1-based human-readable label
    plt.ylabel('Cumulative variance (%)')
    plt.title('The number of components needed to explain variance')

    # 95% cut-off threshold
    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.text(0.5, 0.9, '95% cut-off threshold', color = 'red', fontsize=12)

    # 85% cut-off threshold
    plt.axhline(y=0.80, color='r', linestyle='-')
    plt.text(5.5, 0.75, '80% cut-off threshold', color = 'red', fontsize=12)

    ax.grid(axis='x')
    st.pyplot(fig1)

    st.subheader('Create a Dataframe to identify variance indicators of each variable')
    st.markdown("🔹With n_components = 12")

    # Get columns name of important columns by applying n_components = 11
    pca_ncompo =  PCA(n_components=12).fit(X_train_scaled)
    
    # Create a df to identify important columns
    pca_df = pd.DataFrame(pca_ncompo.components_, columns=X_train.columns, index=['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 
                                                                                    'c9', 'c10', 'c11', 'c12'])
    st.write(pca_df)
    st.markdown('Get the most 12 variables having big variance in the datset')
    important_components = pca_df.abs().sum().sort_values(ascending=False)
    cod1 = '''
                 important_components = pca_df.abs().sum().sort_values(ascending=False)
                 important_components[0:12].index
                 print(important_components)
                '''
    st.code(cod1, language='python')

    st.write(important_components)
    st.markdown("""
                lst_13_var = ['wind_speed', 'humidity', 'PRESSURE_MAX_MB', 'visibility', 'snow',
                            'region_code', 'WEATHER_CODE_EVENING', 'season', 'hol_weekend_vac',
                            'WEATHER_CODE_MORNING', 'total_precip', 'WEATHER_CODE_NOON', 'max_tem'] 
                
                Note: according to the requirements of this project, two required variables are: recipitation and temperature. In this case, we need to add one temperature variable, and we choose 'max_tem'
                """)
    
    st.markdown(" ")
    st.markdown("🔹With n_components = 6")
    # Get columns name of important columns by applying n_components = 8
    pca_ncompo =  PCA(n_components=6).fit(X_train_scaled)
    
    # Create a df to identify important columns
    pca_df = pd.DataFrame(pca_ncompo.components_, columns=X_train.columns, index=['c1', 'c2', 'c3', 'c4', 'c5', 'c6'])
    st.write(pca_df)
    st.markdown('Get the most 6 variables having big variance in the datset')
    important_components = pca_df.abs().sum().sort_values(ascending=False)
    cod2 = '''
                 important_components = pca_df.abs().sum().sort_values(ascending=False)
                 important_components[0:6].index
                 print(important_components)
                '''
    st.code(cod2, language='python')

    st.write(important_components)
    st.markdown("""
                lst_8_var = [region_code', 'season', 'hol_weekend_vac', 'snow', 'humidity', 'wind_speed', 'total_precip', 'max_tem'] 
                
                Note: add 2 features 'total_precip' and 'max_tem'
                """)

   
    st.header("RandomizedSearchCV for best params of model DTR", divider='rainbow')
    code2 = '''
                dico = {'max_depth': range(1, 20),
                'min_samples_split': range(2, 20),
                'min_samples_leaf': range(1, 20)}
                dtree_reg = DecisionTreeRegressor(random_state=42)
                rando = RandomizedSearchCV(dtree_reg, param_distributions=dico , 
                                        n_iter=100, cv=10, random_state=42)
                rando.fit(X_train_scaled, y_train)
        '''
    st.code(code2, language='python')

    st.markdown(" ")
    st.markdown('''
            With n_features = 24
            - Best Parameters: {'min_samples_split': 3, 'min_samples_leaf': 8, 'max_depth': 16}
            - Best Score: 0.9475967392494054
            ''')
    st.markdown(" ")
    st.markdown('''
            With n_features = 13
            - Best Parameters: {'min_samples_split': 6, 'min_samples_leaf': 11, 'max_depth': 8}
            - Best Score: 0.9460895085686358
            ''')
    st.markdown(" ")
    st.markdown('''
            With n_features = 8
            - Best Parameters: {'min_samples_split': 9, 'min_samples_leaf': 10, 'max_depth': 6}
            - Best Score: 0.914556926679847
            ''')
    st.markdown(" ")
    st.markdown('''
            With n_features = 7
            - Best Parameters: {'min_samples_split': 15, 'min_samples_leaf': 1, 'max_depth': 8}
            - Best Score: 0.914556926679847
            ''')


    st.header("Metrics using 13 features", divider='rainbow')

    ### Round 2: after aplying PCS, I have 12 features + 1 temperature feature ###
    X = df1[['wind_speed', 'PRESSURE_MAX_MB', 'visibility', 'humidity', 'snow',
        'WEATHER_CODE_EVENING', 'season', 'region_code', 'WEATHER_CODE_MORNING',
        'total_precip', 'WEATHER_CODE_NOON', 'max_tem']]

    y = df1['total_consum_Mwh']

    # Split and standardize X
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.8)

    # Standardize X 
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_model(13,X_train_scaled, y_train, X_test_scaled, y_test)

    ### DecisionTreeGressor ###
    # With hyperparams
    st.subheader("Decision Tree Regressor with Hyperparameters tuning ")
    modelDTR = DecisionTreeRegressor(min_samples_split = 6, min_samples_leaf = 11, max_depth = 8, random_state=42)
    modelDTR.fit(X_train_scaled, y_train)

    y_test_predict = modelDTR.predict(X_test_scaled)

    st.write("\nScore for the Train-set :", modelDTR.score(X_train_scaled, y_train))
    st.write("\nScore for the Test-set :", modelDTR.score(X_test_scaled, y_test))
    st.write('\nRMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)), 'Mwh')

    st.header("Metrics using 8 features", divider='rainbow')

     ### Round 3: Try with 80% cut-off for n_coponents = 5 then plus 2 required variables ###
    X = df1[['region_code', 'season', 'hol_weekend_vac', 'snow', 'humidity', 'wind_speed', 'max_tem', 'total_precip']]

    y = df1['total_consum_Mwh']

    # Split and standardize X
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.8)

    # Standardize X 
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_model(8,X_train_scaled, y_train, X_test_scaled, y_test)

    ### DecisionTreeGressor ###
    # With hyperparams
    st.subheader("Decision Tree Regressor with Hyperparameters tuning ")
    modelDTR = DecisionTreeRegressor(min_samples_split = 9, min_samples_leaf = 10, max_depth = 6, random_state=42)
    modelDTR.fit(X_train_scaled, y_train)

    y_test_predict = modelDTR.predict(X_test_scaled)

    st.write("\nScore for the Train-set :", modelDTR.score(X_train_scaled, y_train))
    st.write("\nScore for the Test-set :", modelDTR.score(X_test_scaled, y_test))
    st.write('\nRMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)), 'Mwh')

    st.header("Metrics using 7 features", divider='rainbow')
    st.markdown("Following our weather influence analysis, we noticed that the feature 'wind_speed' does not have much impact on regional electricity consumption so we chose to remove this variable.")
    st.markdown(" ")
    ### Round 4: 7 variables ###

    X = df1[['region_code', 'season', 'hol_weekend_vac', 'snow', 'humidity', 'max_tem', 'total_precip']]

    y = df1['total_consum_Mwh']

    # Split and standardize X
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.8)

    # Standardize X 
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_model(7,X_train_scaled, y_train, X_test_scaled, y_test)

    ### DecisionTreeGressor ###
    # With hyperparams
    st.subheader("Decision Tree Regressor with Hyperparameters tuning ")
    modelDTR = DecisionTreeRegressor(min_samples_split = 15, min_samples_leaf = 1, max_depth = 8, random_state=42)
    modelDTR.fit(X_train_scaled, y_train)

    y_test_predict = modelDTR.predict(X_test_scaled)

    st.write("\nScore for the Train-set :", modelDTR.score(X_train_scaled, y_train))
    st.write("\nScore for the Test-set :", modelDTR.score(X_test_scaled, y_test))
    st.write('\nRMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)), 'Mwh')

    st.header("Conclusion", divider='rainbow')
    st.markdown("""
                According to the METRICS EVALUATION graphics, we decided to keep the model using 7 features input for our prediction system:
                - 'region_code': the code to identify the specific region
                - 'season': a specific season
                - 'hol_weekend_vac': a specific type of date (working days, weekend, school holidays or national holidays)
                - 'snow': total quantity of snowfall in mm
                - 'humidity': the percentage of hulidity
                - 'max_tem': the maximum teperature readings
                - 'total_precip': total quantity of rainfall in mm
                """)
    



    