import streamlit as st
from image_search import ImageSearch

text_prompt = st.text_input("Enter the text to search for images")
isearch = ImageSearch(dataset="EduardoPacheco/FoodSeg103")
isearch.embed_images()
results = isearch.search(text_prompt)
submit = st.button("Search")    

if submit:
    for result in results:
         st.image(result["image"])