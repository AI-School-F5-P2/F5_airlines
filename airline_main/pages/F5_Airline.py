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

# Cargar el pipeline desde el archivo
with open('data_pipeline.pkl', 'rb') as file:
    loaded_pipeline = pickle.load(file)


# Crear un objeto Survey
survey = ss.StreamlitSurvey("F5 Airlines")

# Definir los campos del formulario. Tienen que estar en el mismo orden y escritos exactamente igual que en el modelo
features = ["Inflight wifi service", "Food and drink", "Customer Type", "Gender"]

st.title('F5 Airlines')
st.header('Encuesta de satisfacción')
st.write('Cuál es su satisfacción, del 0 al 5, con:')

# Inicializo el formulario
#survey = st.form('f5_airline')


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


col1, col2 = st.columns(2)

# Aquí empieza el formularioairline

with st.form('f5_survey'):
    with col1:
        q1 = survey.radio('Inflight Wifi Service', options=["0", "1", "2", "3", "4", "5"], horizontal=True,  id= 1)
        q3 = survey.radio("Customer Type:", options=["Loyal Customer", "Disloyal Customer"], id='customer_type')

    with col2:
        q2 = survey.radio('Food and Drink', options=["0", "1", "2", "3", "4", "5"], horizontal=True, id=2)
        q4 = survey.radio("Pick your gender:", options=["Female", "Male"], id = 'gender')


    submit = st.form_submit_button('Predict')

    if submit:
        # Recopilar datos en una lista
        data = [q1, q2, q3, q4]

        # Llamar a la función callback para guardar los datos
        df_pred = save_one_data_csv(data)
        st.dataframe(df_pred, use_container_width=True)
        predicción = execute_pipeline(df_pred, data)
        if predicción == "neutral or dissatisfied":
            #st.snow()
            st.error(f"Prediction: {predicción}")
        else:
            st.success(f"Prediction: {predicción}")
            #st.balloons()
