"""
TF-IDF Vectorizer — FROM SCRATCH
==================================
TF-IDF (Term Frequency - Inverse Document Frequency) converts text documents
into numerical vectors, enabling mathematical comparison between them.

The final TF-IDF score for a word in a document combines:
    - TF: How often the word appears in THIS document (local importance)
    - IDF: How rare the word is across ALL documents (global importance)

Formula:
    TF-IDF(t, d) = TF(t, d) × IDF(t)

A word gets a HIGH TF-IDF score when it:
    - Appears frequently in the current document (high TF)
    - Appears rarely across other documents (high IDF)

A word gets a LOW TF-IDF score when it:
    - Appears in many documents (low IDF → word is not distinctive)
    - Appears infrequently in this document (low TF)

Implementation: Combines tf.py and idf.py modules. No sklearn.TfidfVectorizer!
"""

from .tf import compute_tf
from .idf import compute_idf


def compute_tfidf(tokens, idf_scores):
    """
    Calculate the TF-IDF vector for a single document.

    Formula: TF-IDF(t, d) = TF(t, d) × IDF(t)

    Args:
        tokens (list[str]): Preprocessed tokens from one document
        idf_scores (dict[str, float]): Pre-computed IDF scores for all words
                                        in the corpus (from compute_idf)

    Returns:
        dict[str, float]: TF-IDF vector — mapping of each word to its TF-IDF score

    Example:
        >>> tokens = ['python', 'ml', 'python', 'data']
        >>> idf = {'python': 0.0, 'ml': 0.405, 'data': 0.405}
        >>> tfidf = compute_tfidf(tokens, idf)
        >>> tfidf['python']  # common word → TF-IDF ≈ 0
        0.0
        >>> tfidf['data']    # rare word → TF-IDF > 0
        0.101
    """
    if not tokens:
        return {}

    # Step 1: Calculate TF for this document
    tf = compute_tf(tokens)

    # Step 2: Multiply TF × IDF for each word
    tfidf = {}
    for word, tf_score in tf.items():
        idf = idf_scores.get(word, 0.0)  # If word not in corpus, IDF = 0
        tfidf[word] = tf_score * idf

    return tfidf


def build_tfidf_matrix(documents):
    """
    Build TF-IDF vectors for ALL documents at once.

    This is the main function that orchestrates the entire TF-IDF pipeline:
        1. Compute IDF across all documents (global statistics)
        2. Compute TF-IDF vector for each individual document

    Args:
        documents (list[list[str]]): List of documents, where each document
                                      is a list of preprocessed tokens

    Returns:
        tuple: (tfidf_vectors, idf_scores)
            - tfidf_vectors (list[dict]): TF-IDF vector for each document
            - idf_scores (dict): IDF scores (useful for new documents later)

    Example:
        >>> docs = [['python', 'ml'], ['java', 'ml'], ['python', 'web']]
        >>> vectors, idf = build_tfidf_matrix(docs)
        >>> len(vectors)  # One vector per document
        3
    """
    if not documents:
        return [], {}

    # Step 1: Compute IDF across ALL documents
    idf_scores = compute_idf(documents)

    # Step 2: Compute TF-IDF for each document
    tfidf_vectors = []
    for doc_tokens in documents:
        vector = compute_tfidf(doc_tokens, idf_scores)
        tfidf_vectors.append(vector)

    return tfidf_vectors, idf_scores


def get_top_terms(tfidf_vector, n=10):
    """
    Get the top N most important terms from a TF-IDF vector.

    Useful for understanding why a document was matched and for
    displaying the most distinctive words in the Streamlit UI.

    Args:
        tfidf_vector (dict[str, float]): A document's TF-IDF vector
        n (int): Number of top terms to return

    Returns:
        list[tuple[str, float]]: Top N (word, score) pairs, sorted by score descending

    Example:
        >>> vec = {'python': 0.3, 'ml': 0.5, 'data': 0.1, 'java': 0.0}
        >>> get_top_terms(vec, n=2)
        [('ml', 0.5), ('python', 0.3)]
    """
    sorted_terms = sorted(tfidf_vector.items(), key=lambda x: x[1], reverse=True)
    return sorted_terms[:n]
