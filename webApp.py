import numpy as np
import pickle
import streamlit as st

with open("G:/MACHINE LEARNING/Diabetes/trained_model.sav", "rb") as file:
    loaded_model = pickle.load(file, encoding='latin1')

def prediction(input_data):
    
    asnumpy=np.asarray(input_data)
    reshaped=asnumpy.reshape(1,-1)

    #print(std_data)
    prediction=loaded_model.predict(reshaped)
    print(prediction)
    if(prediction[0]==0):
        return("the person is non diabetic")
    else:
        return("the person is diabetic")

def main():
    #TITLE

# Set page title and favicon
    st.set_page_config(page_title="Diabetes Prediction App", page_icon=":bar_chart:")

    # Add a header with an image
   

    st.title("DIABETES PREDICTION WEB APP")

    # Sidebar with additional information or options
    st.sidebar.title("User Guide")
    st.sidebar.write("Enter the required information and click the 'Diabetes Test Results' button to get predictions.")
    st.sidebar.write("Feel free to explore other sections in the sidebar.")

    # Getting input data
    Pregnancies = st.text_input("Number of Pregnancies:")
    Glucose = st.text_input("Value of Glucose:")
    BloodPressure = st.text_input("Value of Blood Pressure:")
    SkinThickness = st.text_input("Value of Skin Thickness:")
    Insulin = st.text_input("Value of Insulin:")
    BMI = st.text_input("Value of BMI:")
    DiabetesPedigreeFunction = st.text_input("Value of Diabetes Pedigree Function:")
    Age = st.text_input("Person's age:")

    # Code for prediction
    diagnosis = ""

    # Creating button
    if st.button("Diabetes Test Results"):
        # Add your prediction function here (assuming prediction is a function)
        # diagnosis = prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        diagnosis = "Positive"  # Replace this with your actual prediction result

    # Display the diagnosis result
    st.subheader("Diabetes Prediction Result:")
    if diagnosis:
        st.success(f"The prediction is: {diagnosis}")
    else:
        st.info("Please enter the required information and click the button to get predictions.")



if __name__=="__main__":
    main()