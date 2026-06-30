from typing import TextIO


class TextChunker:
    def __init__(self,max_palabras:int=512,overlap:int=50):
        self.max_palabras=max_palabras
        self.overlap=overlap 
    def recortar_head_tail_con_separador(self,texto:str,separador:str=" [ ... ]"): 
        palabras =texto.split()
        if len(palabras)<=self.max_palabras:
            return texto 
        mitad=self.max_palabras//2
        palabras_inicio=palabras[:mitad]
        palabras_fin=palabras[-mitad:]
        texto_recortado = ' '.join(palabras_inicio) + separador + ' '.join(palabras_fin)
        return texto_recortado
    def recortaTodo(self,texto):
        return [self.recortar_head_tail_con_separador(d) for d in texto]
    


