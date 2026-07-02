def get_tokenized_grouped_texts(texts: list[str], labels: np.ndarray) -> dict:
    """
    Groups texts by their cluster labels and tokenizes each text.

    Args:
        texts (list[str]): A list of cleaned text documents.
        labels (np.ndarray): An array of cluster labels corresponding to the texts.

    Returns:
        dict: A dictionary where keys are cluster IDs and values are lists of
              tokenized documents belonging to that cluster.
    """
    # Group texts by cluster
    grouped_texts = group_by_cluster(texts, labels)

    # Tokenize each text within the grouped clusters
    tokenized_grouped_texts = {}
    for cid, docs_in_cluster in grouped_texts.items():
        tokenized_grouped_texts[cid] = [text.split() for text in docs_in_cluster]

    return tokenized_grouped_texts

