import streamlit as st

st.sidebar.title("Select Activity")
activity = st.sidebar.selectbox("Select Activity", ["Home", "Webcam Face Detection", "About"])

if activity == "Home":
    st.markdown(
        '''<div style="background-color:#6D7B8D;padding:10px">
        <h4 style="color:white;text-align:center;">Face Emotion detection application using OpenCV, Custom CNN model and Streamlit</h4>
        </div><br>''', unsafe_allow_html=True)
    st.write("The application has two functionalities:\n\n1. Real time face detection using webcam.\n2. Real time face emotion recognition.")

elif activity == "About":
    st.subheader("About this app")
    st.write("This app is developed to demonstrate real-time face emotion recognition.")

elif activity == "Webcam Face Detection":
    st.write("Webcam functionality goes here.")
