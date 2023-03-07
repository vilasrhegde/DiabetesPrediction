import numpy as np
import pickle
import streamlit as st
from datetime import date

loaded_model = pickle.load(open('./trained_dibetes_model.sav','rb'))


#creating the function to prediction
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="random",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://vilasrhegde.github.io/',
        'Report a bug': "https://www.linkedin.com/in/vilasrhegde/",
        'About': "# Made by Vilas Hegde!"
    }
)
def prediction(input_data):
    
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we are predicting for one instance
    #our model is trained for 700+ elements
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)

    if (prediction[0]==0):
        return 'Not diabetic'
    else:
        return 'Diabetic!'


def main():

    #giving the title for webpage
    st.title('Diabetes Prediction')
    st.write("Enter examined details:")

    #getting the input data from the user
    col1,col2,col3,col4=st.columns(4)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies',value=3)
    with col2:
        Glucose = st.text_input('Blood-Glucose level',value=148)       
    with col3:
        BloodPressure = st.text_input('Blood Pressure level',value=72)    
    with col4:
        SkinThickness = st.text_input('Skin Thickness value',value=35)   
    with col1:
        Insulin = st.text_input('Insulin level',value=0)   
    with col2:
        BMI = st.text_input('BMI value',value=33.6)   
    with col3:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value',value=0.63)   
    with col4:
        Age = st.text_input('Age of the Person',value=52)   

    #code for prediction
    diagnosis = ''

    #creating a button for prediction
    if st.button('Test'):
        diagnosis = prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,
        Insulin,BMI,DiabetesPedigreeFunction,Age])

    # st.warning(diagnosis)
    # st.balloons()
        st.balloons()

        if(diagnosis!="Diabetic!"):
            st.info(diagnosis)
        else:
            st.error(diagnosis)
        
    st.markdown(f":red[©️ Vilas Hegde - {date.today().year}]")


if __name__ == '__main__':
    main()

#open anaconda
# go to  environments and run with terminal
#type  streamlit run "path for this file with same backslash"
