from preprocessing.chunks import TextChunker
def test_chunking_splits_long_text():
    text = " ".join(["palabra"] * 1000)

    chunker = TextChunker(max_palabras=100, overlap=20)
    chunks = chunker.recortar_head_tail_con_separador(text)

    assert len(chunks) > 1
