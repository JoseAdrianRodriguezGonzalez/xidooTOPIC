import unicodedata
import re

class TextCleaner:
    def __init__(self,min_words:int=3,keep_number:bool=True,keep_emoji:bool=True) -> None:
        self.min_words=min_words
        self.keep_number=keep_number
        self.keep_emoji=keep_emoji
        
    def transform(self,reviews:list[str]):
        cleaned_reviews=[]
        indexes=[]
        i=0

        for text in reviews:
            if not isinstance(text, str):
                continue
            text=unicodedata.normalize("NFKC",text)
            text=text.lower()

            text=re.sub(r"http\S+|www\S","",text)
            text = re.sub(r"[^a-záéíóúüñ\s]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

            if len(text.split()) >= self.min_words:
                cleaned_reviews.append(text)
                indexes.append(i)
            i = i+1
        return cleaned_reviews,indexes


