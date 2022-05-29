import streamlit as st
import pickle
import numpy as np

#create out streamlit app title

# loading the saved model
my_model=pickle.load(open('C:/Users/Julien/Desktop/projectFellowship/myfile_model.sav', 'rb'))

# creating a function for prediction

def drones_prediction(input_data):

    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = my_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]== 0):
        return 'in this month there is no much request of drones'
    else:
        return 'in this month there is much request of drones'

#Create imput widgets
def main():
    st.title("PREDICTING THE DEMAND FOR DRONE FLIGHTS IN IRRIGATION")

    Month = st.selectbox("select the Month",[1,2,3,4,5,6,7,8,9,10,11,12])
    Location= st.number_input('Enter the Location')
    MinTemp = st.number_input('Enter the MinTemp')
    MaxTemp = st.number_input('Enter the MaxTemp')
    Rainfall = st.number_input('Enter the Rainfall')
    RainToday =  st.number_input("did it rain today?")

    ckeck_rain= ''

    
    if st.button('drones test result'):
        check_rain =drones_prediction([Month, Location,MinTemp,MaxTemp,Rainfall,RainToday, ]) 


        st.success(check_rain)


if __name__ == '__main__':
    main()  
