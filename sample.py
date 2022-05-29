import numpy as np
import pickle

my_model = my_model=pickle.load(open('C:/Users/Julien/Desktop/projectFellowship/myfile_model.sav', 'rb'))  

input_data= [2,5,6,7,8,9,0,2,3,4,5,6,7,8,9,0,1,23,45,67,34,56,76,54,32]

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = my_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
    print( 'there is no much rain for this month')
else:
    print ('there is much rain this comming month')