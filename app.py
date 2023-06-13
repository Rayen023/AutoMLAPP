import streamlit as st
import pandas as pd
import os 
import pandas_profiling
import matplotlib.pyplot as plt


from streamlit_pandas_profiling import st_profile_report # installed via pip not conda

from pycaret.classification import setup, compare_models, pull, save_model, load_model


with st.sidebar : 
    st.image("./images/pngwing.png")
    st.info(" This app allows automated ml using streamlit, pandas profiling and pycaret")
    st.title("AutoMLAPP")
    choice = st.radio("Navigation" , ['Upload','Profiling','Modeling'])



if choice == "Upload":
    st.title("Upload")
    file = st.file_uploader('Uplad Dataset')
    if file :
        df = pd.read_csv(file, index_col = 0)
        st.session_state.df = df
        st.dataframe(df)
        try :
            os.mkdir('./Datasets/')
        except:
            pass
        df.to_csv('./Datasets/source_data.csv' ,index = None)

        chosen_target = st.selectbox('Choose the Target Column', df.columns )
        st.bar_chart(df[str(chosen_target)].value_counts())

        chosen_pie_target = st.selectbox('Choose the Target Column Pie', df.columns )
        fig1, ax1 = plt.subplots()
        plt.rcParams['figure.facecolor'] = 'black'
        ax1.pie(x=df[str(chosen_pie_target)].value_counts(),
                labels=df[str(chosen_pie_target)].value_counts().index,
                autopct='%1.1f%%',shadow=True,startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig1)


        
if choice == "Profiling":
    
    st.title("Exploratory Data Analysis")       
    try :
        
        df = st.session_state.df
        st.info('Uploaded Dataframe head : ')
        st.dataframe(df.head())
        if st.button('Run Profiling'): 
            st.session_state.pr_df = df.profile_report()        
            st_profile_report(st.session_state.pr_df)

        elif 'pr_df' in st.session_state:
            st_profile_report(st.session_state.pr_df)
    except : 
        st.info('Dataframe not uploaded')


if choice == "Modeling":
    st.title("Modeling")

    df = st.session_state.df 
    try :
        df.drop('ID' , axis = 1 , inplace = True)
        df.drop('Time' , axis = 1 , inplace = True)
    except: 
        pass
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target, silent=True , fix_imbalance= True, outliers_threshold= 0.001 , normalize= True)
        setup_df = pull()
        st.dataframe(setup_df , use_container_width= True)
        st.session_state.setup_df = setup_df
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df , use_container_width= True)
        st.session_state.compare_df = compare_df
        save_model(best_model, 'best_model') # saved remotely
        st.session_state.best_model = best_model
        st.info(st.session_state.best_model)
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download Top Model', f, file_name="best_model_d.pkl") #saved locally

    else : 
        if 'setup_df' in st.session_state:
            st.dataframe(st.session_state.setup_df)
        if 'compare_df' in st.session_state:
            st.dataframe(st.session_state.compare_df)
        if 'best_model' in st.session_state:
            st.info(st.session_state.best_model)
            with open('best_model.pkl', 'rb') as f:
                st.download_button('Download Top Model', f, file_name="best_model_d.pkl") #saved locally



