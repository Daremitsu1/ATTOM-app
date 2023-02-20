import streamlit as st
import os
from arcgis.learn.text import TextClassifier, SequenceToSequence
import pickle

# Load the text classifier from the saved files
model_folder = "models/text-classifier"
model = TextClassifier.load(model_folder, "text-classifier")

# Save the text classifier to a .pkl file
with open("text_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

with st.sidebar:
    st.image('https://www.attomdata.com/wp-content/uploads/2021/05/ATTOM-main-full-1000.jpg')
    st.title("AutoAttom")
    st.info("This project application will help in text classification and sequence to sequence labelling")

# Text Classifier Section
st.title("Text Classifier")
user_input = st.text_input(""" """)
if user_input:
    model_folder = os.path.join("models", "text-classifier")
    model = TextClassifier()
    model.load(model_folder, 'text-classifier')
    st.write(model.predict(user_input))

# Sequence to Sequence Section
st.title("Sequence-To-Sequence")
user_input = st.text_input("")
if user_input:
    model_folder = os.path.join("models", "seq2seq_unfrozen8E_bleu_88")
    model = SequenceToSequence(model_folder)
    st.write(model.predict(user_input))