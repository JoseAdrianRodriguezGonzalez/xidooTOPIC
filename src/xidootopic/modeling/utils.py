def obtener_lista_palabras(lista_tuplas):
    # Extraemos solo el primer elemento (la palabra) de cada tupla en cada lista del diccionario
   return [palabra for palabra, valor in lista_tuplas]
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

