import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration, pipeline
import matplotlib.pyplot as plt
from modelo import Modelo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner="⏳ Cargando modelos…")
def cargar_modelos(model_name_emociones: str, model_name_qa: str):
    tok_emociones = AutoTokenizer.from_pretrained(model_name_emociones)
    mod_emociones = AutoModelForSequenceClassification.from_pretrained(model_name_emociones)
    pipeline_emociones = pipeline("text-classification", model=mod_emociones, tokenizer=tok_emociones)

    tok_qa = AutoTokenizer.from_pretrained(model_name_qa)
    mod_qa = T5ForConditionalGeneration.from_pretrained(model_name_qa).to(device)

    return pipeline_emociones, tok_qa, mod_qa

st.set_page_config(page_title="Análisis de comentarios", layout="wide")
st.title("Análisis de comentarios")
st.write("Esta página está creada para el análisis de comentarios de películas o productos.")
st.write("Seleccione el archivo CSV con los comentarios que desea analizar.")

uploaded_file = st.file_uploader("Elija el archivo CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Esta es la información que se encuentra en el archivo:")
    st.dataframe(df.head())

    column = st.selectbox("Seleccione la columna donde se encuentran los comentarios a analizar", df.columns)

    if column != "Unnamed: 0":
        modelo = Modelo(df, column)

        modelo.procesamiento()

        nube = modelo.nubedepalabras()
        st.pyplot(nube)

        grafico = modelo.graficoDeBarras()
        st.pyplot(grafico)

        pipeline_emociones, tok_qa, mod_qa = cargar_modelos("boltuix/bert-emotion", "mrm8488/spanish-t5-small-sqac-for-qa")

        modelo.clasificacion(pipeline_emociones)

        graficoResultado = modelo.graficoDeresultados()
        st.pyplot(graficoResultado)

        pregunta = st.text_input("Escriba la pregunta que desea hacerle al modelo sobre los comentarios:")

        if st.button("Hacer pregunta"):
            respuesta = modelo.pregunta(tok_qa, mod_qa, pregunta)

            if respuesta:
                st.write("Respuesta del modelo: ", respuesta)
            else:
                st.write("Respondiendo pregunta...")

    else:
        st.write("Por favor, seleccione una columna válida.")
else:
    st.write("Por favor, seleccione un archivo CSV para el análisis.")
