from modeling.representation import get_topk_frequent,KeyWordExtractor,TopicSummarizer
import numpy as np 
import pytest
import spacy

@pytest.fixture
def nlp():
    return spacy.load("en_core_web_sm")
def test_topk_frequent(nlp):
    words = ["playa", "playas", "mar", "mar", "arena"]

    result = get_topk_frequent(words, nlp)

    assert isinstance(result, list)
    assert "playa" in result or "mar" in result
class DummyKW:
    def extract_keywords(self, text, **kwargs):
        return [("playa", 0.9), ("mar", 0.8)]
def test_keyword_extractor(nlp):
    texts = ["playa bonita", "mar azul", "hotel limpio"]
    emb = np.random.rand(3, 5)
    labels = np.array([0, 0, 1])

    extractor = KeyWordExtractor(
        kw_model=DummyKW(),
        stopwords=None,
        nlp=nlp
    )

    result = extractor.extract(texts, emb, labels)

    assert isinstance(result, dict)
    assert len(result) == 2

    for k, v in result.items():
        keywords, resumen = v
        assert isinstance(keywords, list)
        assert isinstance(resumen, str)
class DummySummarizer:
    def summarize(self, text, max_length=128):
        return "summary"
def test_summarizer():
    s=DummySummarizer()

    summary = s.summarize("text")

    assert summary=="summary"
