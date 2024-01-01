import numpy as np
import pickle

with open("G:/MACHINE LEARNING/Diabetes/trained_model.sav", "rb") as file:
    loaded_model = pickle.load(file, encoding='latin1')

#MAKING PREDICTIVE SYStem

input_data=(5,166,72,19,175,25.0,0.507,51)
asnumpy=np.asarray(input_data)
reshaped=asnumpy.reshape(1,-1)

#print(std_data)
prediction=loaded_model.predict(reshaped)
print(prediction)
if(prediction[0]==0):
  print("the patient is non diabetic")
else:
  print("diabetic")