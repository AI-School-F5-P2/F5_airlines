FROM python:3.10.9

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia los archivos necesarios al contenedor
COPY airline_main/ .
#COPY airline_passenger_satisfaction.csv .
#COPY app.py .
#COPY Columnas_en_orden.txt .
#COPY data_pipeline.pkl .
#COPY feedback_data.csv .
#COPY main.ipynb .
#COPY new_data.csv .
#COPY one_data.csv .
#COPY plane_blue.png .
#COPY test_sgs.ipynb .
COPY .gitignore .
COPY README.md .
COPY requirements.txt .


# librerias instaladas
# ADD requirements.txt ./requirements.txt

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt && rm -rf requirements.txt


# Instala spacy y sus modelos
RUN pip install spacy
RUN pip install streamlit_survey
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download en_core_web_md
RUN python -m spacy download de_core_news_sm
# Comando para ejecutar la aplicación Streamlit
CMD ["streamlit", "run", "F5_Airline_App.py"]

