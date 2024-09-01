from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
import nltk
import time
from num2words import num2words
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Descarga de recursos NLTK
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

app = Flask(__name__)

# Inicializar variables globales para TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = None

def convert_lower_case(data):
    return np.char.lower(data)

def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text

def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data

def remove_apostrophe(data):
    return np.char.replace(data, "'", "")

def stemming(data):
    stemmer = PorterStemmer()
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text

def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text

def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data)  # remove comma separately
    data = remove_apostrophe(data)
    data = remove_stop_words(data)
    data = convert_numbers(data)
    data = stemming(data)
    data = remove_punctuation(data)
    data = convert_numbers(data)
    data = stemming(data)  # needed again as we need to stem the words
    data = remove_punctuation(data)  # needed again as num2word is giving few hyphens and commas forty-one
    data = remove_stop_words(data)  # needed again as num2word is giving stop words 101 - one hundred and one
    return data

def cosine_sim(query, tfidf_matrix):
    query_tfidf = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    return cosine_similarities

def buscar_documentos_mas_relevantes(query, dataset, top_n=5):
    # Preprocesa la consulta
    query_original = query
    query = preprocess(query)
    
    # Calcula la similitud coseno entre la consulta y la matriz TF-IDF
    similitudes = cosine_sim(query, tfidf_matrix)
    
    # Obtén los índices de los top N documentos más similares
    indices_mas_relevantes = np.argsort(similitudes)[::-1][:top_n]
    
    # Extrae los documentos más relevantes y sus similitudes correspondientes
    documentos_mas_relevantes = [
        {
            "path": dataset[i]['path'],  # Ajuste aquí para acceder al 'path' en un diccionario
            "title": dataset[i].get('title', 'Sin título'),  # Acceso al 'title'
            "similitud": similitudes[i]
        }
        for i in indices_mas_relevantes
    ]
    resultado = {"recomendaciones": documentos_mas_relevantes, "query":query_original}
    
    return resultado

def procesar_html_bruto(lista_html):
    # Lista para almacenar el contenido textual relevante
    dataset = []

    # Leer el contenido de cada HTML bruto y extraer el texto
    for indice, html_bruto in enumerate(lista_html):
        try:
            # Usar BeautifulSoup para analizar el HTML
            soup = BeautifulSoup(html_bruto, 'lxml')

	    # Extraer el título del HTML si existe
            titulo = soup.title.string if soup.title else "Sin título"
            
            # Eliminar todas las etiquetas <script> y <style>
            for script in soup(['script', 'style']):
                script.decompose()
            
            # Seleccionar los elementos de contenido principal: títulos, párrafos, etc.
            articulos = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p'])
            
            # Concatenar el texto de estos elementos
            texto_articulos = ' '.join([articulo.get_text(separator=' ', strip=True) for articulo in articulos])
            
            # Agregar el texto relevante a la lista 
            dataset.append((titulo, texto_articulos))
        
        except Exception as e:
            # Manejar cualquier error de procesamiento
            print(f"Error al procesar el HTML en el índice {indice}: {e}")
    
    return dataset
@app.route('/')
def worker_status_check():
    return "Worker esta bien", 200
    
@app.route('/procesar-html', methods=['POST'])
def procesar_html():
    datos = request.get_json()
    start_time = time.time()
    
    # Registro del contenido recibido para depuración
    print("Received data:", request.data.decode('utf-8'), flush=True)
    
    if not datos:
        return jsonify({"error": "No JSON data received"}), 400
    
    dataset = datos.get('dataset', []) 
    query = datos.get('query', '')

    dataset_tuplas = [(item.get('id'), item.get('path'), item.get('content')) for item in dataset]
    
    # Verificación de que el dataset no esté vacío
    if not dataset_tuplas:
        return jsonify({"error": "Dataset is empty or not provided"}), 400

    # Procesar los HTMLs
    html_list = [x[2] for x in dataset_tuplas]
    
    processed_data = procesar_html_bruto(html_list)

    processed_titles = [item[0] for item in processed_data]
    processed_texts = [item[1] for item in processed_data]

    global tfidf_matrix  # Necesitamos usar la variable global aquí
    tfidf_matrix = vectorizer.fit_transform(processed_texts)

    dataset_tuplas = [
        {"id": item[0], "path": item[1], "title": title, "content": item[2]} 
        for item, title in zip(dataset_tuplas, processed_titles)
    ]
    
    top_documentos = buscar_documentos_mas_relevantes(query, dataset_tuplas)
    end_time = time.time()
    execution_time = end_time - start_time
    
    top_documentos['time']=execution_time
    
    return jsonify(top_documentos)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
