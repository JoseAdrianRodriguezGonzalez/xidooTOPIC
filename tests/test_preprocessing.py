from preprocessing.base import TextCleaner
def test_clean_reviews_basic():
    texts = [
        "Este es un texto!!! con URL http://test.com",
        "ok",
        None
    ]
    cleaner = TextCleaner(min_words=3)
    cleaned, idx = cleaner.transform(texts)
    print(cleaned)
    assert len(cleaned) == 1
    assert "http" not in cleaned[0]
