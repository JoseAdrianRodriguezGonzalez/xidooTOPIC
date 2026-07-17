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

