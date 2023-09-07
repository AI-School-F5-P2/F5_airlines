import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
import pickle
from sklearn.pipeline import Pipeline
import time

import streamlit_survey as ss


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
survey = ss.StreamlitSurvey("F5 Airlines")

# Cargar el pipeline desde el archivo
with open('data_pipeline.pkl', 'rb') as file:
     loaded_pipeline = pickle.load(file)



# Definir los campos del formulario. Tienen que estar en el mismo orden y escritos exactamente igual que en el modelo
features = ["Gender", "Customer Type", "Age", "Type of Travel", "Class", "Flight Distance", "Inflight wifi service",
            "Departure/Arrival time convenient", "Ease of Online booking", "Gate location", "Food and drink",
            "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service",
            "Baggage handling", "Checkin service", "Inflight service", "Cleanliness", "Departure Delay in Minutes",
            "Arrival Delay in Minutes"]

st.title('F5 Airlines')
st.header('Encuesta de satisfacción')
st.write('Cuál es su satisfacción, del 0 al 5, con:')


# Crear una lista para almacenar los datos
data = []


# Definir una función de devolución de llamada para guardar los datos en CSV
def save_one_data_csv(data):
    try:
        with open('one_data.csv', "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(features)
            writer.writerow(data)

    except FileNotFoundError as e:
        print(f"Error al abrir el archivo: {e}.")

    df_pred = pd.read_csv('one_data.csv')
    return df_pred


def save_new_data_csv(df, pred):
    try:
        df['satisfaction'] = pred
        df.to_csv('new_data.csv', mode="a", header=False, index=False)
    except FileNotFoundError as e:
        print(f"Error al abrir el archivo: {e}.")


def delete_File(file_to_delete):
    try:
        os.remove(file_to_delete)
        print(f"El archivo {file_to_delete} ha sido eliminado correctamente")
    except FileNotFoundError:
        print(f"El archivo {file_to_delete} no existe en la ubicación especificada")
    except Exception as e:
        print(f"Ocurrió un error al intentar eliminar el archivo {e}")


def execute_pipeline(df_data, data):
    # Aplicar el pipeline a los datos de entrada
    y_pred = loaded_pipeline.predict(df_data)

    # Guardar las predicciones en el DataFrame
    df_data['satisfaction'] = y_pred
    df_data.to_csv('new_data.csv', mode="a", header=False, index=False)
    return y_pred

# Crear un objeto Survey. Se instancia el formulario
survey = st.form('F5_Airlines')

opc_val = ["0", "1", "2", "3", "4", "5"]

with survey:
    q1 = survey.radio('Inflight Wifi Service', options=opc_val, horizontal=True )
    q2 = survey.radio('Departure/Arrival time convenient', options=opc_val, horizontal=True)
    q3 = survey.radio('Ease of Online booking', options=opc_val, horizontal=True)
    q4 = survey.radio('Gate location', options=opc_val, horizontal=True)
    q5 = survey.radio('Food and drink', options=opc_val, horizontal=True)
    q6 = survey.radio('Online boarding', options=opc_val, horizontal=True)
    q7 = survey.radio('Seat comfort', options=opc_val, horizontal=True)
    q8 = survey.radio('Inflight entertainment', options=opc_val, horizontal=True)
    q9 = survey.radio('On-board service', options=opc_val, horizontal=True)
    q10 = survey.radio('Leg room service', options=opc_val, horizontal=True)
    q11 = survey.radio('Baggage handling', options=opc_val, horizontal=True)
    q12 = survey.radio('Checkin service', options=opc_val, horizontal=True)
    q13 = survey.radio('Inflight service', options=opc_val, horizontal=True)
    q14 = survey.radio('Cleanliness', options=opc_val, horizontal=True)
    q15 = survey.selectbox("Customer Type", options=["Loyal Customer", "disloyal Customer"])
    q16 = survey.selectbox("Type of Travel", options=["Personal Travel", "Business travel"])
    q17 = survey.selectbox("Class", options=["Eco Plus", "Business", "Eco"])
    q18 = st.number_input("Flight Distance", min_value=0, max_value= 1000000, value=0, key=18)
    q19 = st.number_input("Departure Delay in Minutes", min_value=0, value=0, max_value=720, key=19)
    q20 = st.number_input("Arrival Delay in Minutes:", min_value=0, value=0)
    q21 = st.number_input("Age:", min_value=0, max_value=106, value=0)
    q22 = survey.radio('Gender', options=["Female", "Male"], horizontal=True)
    q23 = survey.radio('Como considera que se siente con relación a la Aerolínea.:',
                       options=["satisfied", "neutral or dissatisfied"], horizontal=True)
    submit = st.form_submit_button('Predict')
    if submit:
        # Recopilar datos en una lista
        data = [q22, q15, q21, q16, q17, q18, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q19, q20]

        # Llamar a la función callback para guardar los datos
        df_pred = save_one_data_csv(data)
        #st.dataframe(df_pred, use_container_width=True)
        predicción = execute_pipeline(df_pred, data)
        if predicción == "neutral or dissatisfied":
            # st.snow()
            st.error(f"Prediction: {predicción}")
        else:
            st.success(f"Prediction: {predicción}")
