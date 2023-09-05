import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
import pickle
from sklearn.pipeline import Pipeline
import time


# Configurar la información personalizada en la sección "About"
about_text = """
**F5 Airlines. Grupo 1**

**Coders:**
- María López Jiménez
- Karla Lamus
- Javi Navarro
- Sandra Gómez Santamaría

[Repositorio del proyecto](https://github.com/AI-School-F5-P2/F5_airlines-G1.git)
"""
# Page Configuration
st.set_page_config(
    page_title="F5 Airlines Predict App",
    page_icon="🛫",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'About': about_text
    }
)

image = 'plane_blue.png'
img_width = 250
img_height = 250


col1, col2 = st.columns([0.3,0.7],)
with st.container():
    with col1:
        st.image(image, width=img_width)
    with col2:
        st.markdown("# Proyecto F5 Airlines")
        st.write('Aprendizaje Supervisado. Clasificación')


st.write('**La Airline F5 App**, recoge los datos de un cliente nuevo y realiza una predicción sobre su grado de satisfacción.')
multi = ''' Cupcake ipsum dolor sit amet danish. Muffin sesame snaps cupcake I love gingerbread biscuit lemon drops
 
soufflé cupcake. I love I love topping liquorice bonbon gummies 
lemon drops. Icing muffin chocolate cake jelly-o muffin halvah. Macaroon I love pudding toffee topping pudding fruitcake. 

Topping muffin fruitcake I love topping lollipop I love jelly.
Ice cream I love bear claw oat cake biscuit. Pastry bear claw brownie chupa chups gingerbread biscuit. 
I love tart muffin cake I love tiramisu.
'''
st.markdown(multi)








