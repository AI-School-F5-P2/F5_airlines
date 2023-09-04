import streamlit as st

# Inicializa variables de estado para controlar la página actual y almacenar respuestas
if 'page' not in st.session_state:
    st.session_state.page = 1

# Define las preguntas y tipos de inputs para cada página
pages = {
    1: [
        {'question': 'Pregunta 1 (Slider)', 'input_type': 'slider'},
        {'question': 'Pregunta 2 (Radio)', 'input_type': 'radio'},
        {'question': 'Pregunta 3 (Slider)', 'input_type': 'slider'},
    ],
    2: [
        {'question': 'Pregunta 4 (Radio)', 'input_type': 'radio'},
        {'question': 'Pregunta 5 (Slider)', 'input_type': 'slider'},
    ],
    3: [
        {'question': 'Pregunta 6 (Radio)', 'input_type': 'radio'},
        {'question': 'Pregunta 7 (Slider)', 'input_type': 'slider'},
    ],
}

# Función para mostrar y procesar el formulario de la página actual
def show_page(page_number):
    st.write(f'Página {page_number}')
    with st.form(f'page_{page_number}_form'):
        for question_info in pages.get(page_number, []):
            question = question_info['question']
            input_type = question_info['input_type']
            if input_type == 'slider':
                response = st.slider(question, 0, 10)
            elif input_type == 'radio':
                options = ['Opción A', 'Opción B', 'Opción C']
                response = st.radio(question, options)
        if st.form_submit_button('Siguiente página'):
            st.session_state.page += 1

# Mostrar el formulario de la página actual
if st.session_state.page <= len(pages):
    show_page(st.session_state.page)
