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
- Sandra Gómez S.

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


st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
multi = ''' Estamos encantados de tenerlo a bordo y nos encantaría saber más sobre su experiencia.
¿Por qué? Porque usted importa. Nos apasiona ofrecer el mejor servicio posible, y su opinión nos ayuda a 
elevar nuestros estándares.

Simplemente haga click en la pestaña **F5 Airline Survey** y complete el formulario. 
No olvide pulsar el botón **'ENVIAR'**. ¡Así de fácil! .
'''
st.markdown(multi)








