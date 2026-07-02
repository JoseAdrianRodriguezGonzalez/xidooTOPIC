class TopicFormatter:
    def format(self, topics_dict):
        lista_keywords = []
        texto_formateado = ""

        for topic_id, (keywords, resumen) in topics_dict.items():
            aux=keywords
            lista_keywords.append(aux)
            #obtener_lista_palabras(keywords)

            bloque = (
                f"Tópico: {topic_id}\n"
                f"Keywords: {', '.join(aux)}\n"
                f"Documento:\n{resumen}\n"
                f"{'-'*30}\n"
            )

            texto_formateado += bloque

        return lista_keywords, texto_formateado
prompt = """You are a data analyst specialized in topic modeling evaluation.

Your task is to analyze the output of a topic modeling process based strictly and exclusively on the information provided.

You will receive:
1. A list of keywords extracted for a topic.
2. A set of representative tourist reviews associated with that topic.

IMPORTANT RULES:

1. You must base your analysis ONLY on the keywords and documents provided.
2. You must NOT assume, infer, or introduce any geographical locations, destinations, cities, landmarks, or contextual information that are not explicitly mentioned in the provided text.
3. If the documents mention a location explicitly, you may refer to it. Otherwise, do not introduce any location.
4. If the keywords and documents do not appear semantically coherent or do not share a clear common theme, you must classify the topic as: "INCOHERENT – DISCARD".
5. Do not speculate.
6. Do not add external knowledge.
7. Be concise, analytical, and objective.

For each topic, provide:

- Topic ID:
- Decision: (VALID / INCOHERENT – DISCARD)
- Proposed Topic Label (max 6 words, neutral and descriptive):
- Short Analytical Description (2–3 sentences strictly grounded in the provided text)

After analyzing all topics:

- Identify if any VALID topics are semantically overlapping.
- If so, propose a merged topic label and indicate which Topic IDs should be merged.
- Do not propose to merge topics unless there is strong semantic overlap explicitly visible in the data.
- If you are uncertain, propose to DISCARD rather than inventing a thematic interpretation.

Output must follow this structure exactly.

below are the input information from the topic modeling process:

"""

def obtener_lista_palabras(lista_tuplas):
    # Extraemos solo el primer elemento (la palabra) de cada tupla en cada lista del diccionario
   return [palabra for palabra, valor in lista_tuplas]
def format_llm_input_string(cluster_keywords_dict, representative_docs_dict):
    strin = ""
    # Get unique sorted cluster IDs that are present in both dictionaries
    cluster_ids = sorted(list(set(cluster_keywords_dict.keys()) & set(representative_docs_dict.keys())))

    for topic_id in cluster_ids:
        # Convert np.int64 to standard int for f-string formatting if necessary
        strin += f"Topic ID: {int(topic_id)}:\n"

        # Get keywords for this topic_id
        # cluster_keywords_dict[topic_id] is a list of tuples: [('keyword', score), ...]
        keywords_list = [item[0] for item in cluster_keywords_dict[topic_id]]
        strin += f"Keywords:\n{', '.join(keywords_list)}\n"

        strin += f"Documents:\n"
        # representative_docs_dict[topic_id] is a list of tuples: [('document_text', score), ...]
        for i, (doc_text, doc_score) in enumerate(representative_docs_dict[topic_id]):
            strin += f"{i+1}: {doc_text}\n"
        strin += "\n" # Add an extra newline for better separation

    return strin

def obtener_keyw(diccionario):
    listas_resultado = []

    # Iteramos sobre los ítems del diccionario (clave y su lista de datos)
    # Usamos sorted() por si quieres que el resultado respete el orden numérico de las claves
    for clave in sorted(diccionario.keys()):
        datos = diccionario[clave]

        # Extraemos solo el texto (primer elemento de cada sublista)
        # Python resuelve automáticamente el Unicode (ej: \u00f3 -> ó) al procesar la cadena
        palabras = [item[0] for item in datos]

        # Unimos las palabras con comas y las añadimos a nuestra lista final
        cadena_palabras = ", ".join(palabras)
        listas_resultado.append(cadena_palabras)

    return listas_resultado
