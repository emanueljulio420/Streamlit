import re
import string
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


class Modelo:
    def __init__(self, data, columnName):
        self.columnName = columnName
        self.data = data

        self.stopword_es = set(stopwords.words('spanish'))
        self.stemmer_es = SnowballStemmer("spanish")

    def procesamiento(self):

        self.data.head()

        self.data["clean_text"] = self.data[self.columnName].apply(self.limpieza)

        self.data["clean_text"] = self.data["clean_text"].apply(self.clean_with_stopwords_and_stemming_regex)

    def clean_with_stopwords_and_stemming_regex(self, text):
   
        stopword_es = set(stopwords.words('spanish'))
        stemmer_es = SnowballStemmer("spanish")
        tokens = re.findall(r'\b\w+\b', text.lower(), flags=re.UNICODE)
        stemmed_tokens = [stemmer_es.stem(token) for token in tokens if token not in stopword_es]

        return " ".join(stemmed_tokens).strip()


    def limpieza(self, text):

        text = str(text).lower()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\w*\d\w*', '', text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = text.strip()

        return text
    
    def nubedepalabras(self):

        text = " ".join(review for review in self.data["clean_text"])
        wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", width=800, height=400).generate(text)

        figura, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title("Nube de Palabras - Reseñas Limpias")

        return figura
    
    def graficoDeBarras(self):


        texto_completo = ' '.join(self.data["clean_text"].str.lower())
        
        tokens = re.findall(r'\b\w+\b', texto_completo)
        
        cuenta = Counter(tokens)
        palabras, frecs = zip(*cuenta.most_common(10))

        figura, ax = plt.subplots()
        ax.barh(palabras[::-1], frecs[::-1]) 
        ax.set_title("10 palabras más frecuentes")
        ax.set_xlabel("Frecuencia")
        figura.tight_layout()

        return figura
    
    def clasificacion(self, modelo_emociones):

        self.data["clasificacion"] = self.data["clean_text"].apply(lambda x: modelo_emociones(x)[0]['label'])
        self.data["clasificacion"] = self.data["clasificacion"].astype("category")

    def pregunta(self, tok, mod, pregunta):
        context = self.data[self.columnName]
        prompt = f"question: {pregunta}  context: {context}"
        
        inputs = tok(prompt, max_length=512, truncation=True, return_tensors="pt").to(mod.device)
        ids = mod.generate(**inputs, max_length=64, num_beams=4)
        respuesta = tok.decode(ids[0], skip_special_tokens=True)

        return respuesta
        

    def graficoDeresultados(self):
        conteo = self.data["clasificacion"].value_counts()

        figura, ax = plt.subplots()
        conteo.plot(kind='bar', ax=ax)
        ax.set_title("Clasificación de Reseñas")
        ax.set_xlabel("Clasificación")
        ax.set_ylabel("Frecuencia")

        return figura

export = Modelo