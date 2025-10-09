import streamlit as st

st.title("👋 Hello Streamlit App")

name = st.text_input("Enter your name:")

if st.button("Submit"):
    st.success(f"Hello, {name}!")