"""
Inverse Document Frequency (IDF) — FROM SCRATCH
=================================================
IDF measures how RARE or UNIQUE a word is across all documents.
It's the second half of the TF-IDF calculation.

The key insight: Words that appear in many documents are less useful
for distinguishing between them. IDF penalizes common words and
rewards rare, distinctive ones.

Formula:
    IDF(t) = log(N / (1 + df(t)))

    Where:
        N    = Total number of documents in the corpus
        df(t) = Number of documents that contain term t
        +1   = Smoothing factor to prevent division by zero
        log  = Natural logarithm (compresses the scale)

Example (3 documents):
    Doc1: "python machine learning"
    Doc2: "python data science"
    Doc3: "java machine learning"

    N = 3 documents

    "python"   appears in Doc1, Doc2     → df=2 → IDF = log(3/3) = 0.0
    "machine"  appears in Doc1, Doc3     → df=2 → IDF = log(3/3) = 0.0
    "java"     appears in Doc3 only      → df=1 → IDF = log(3/2) = 0.405
    "data"     appears in Doc2 only      → df=1 → IDF = log(3/2) = 0.405
    "science"  appears in Doc2 only      → df=1 → IDF = log(3/2) = 0.405

    → "java", "data", "science" are MORE important (rare = distinctive!)
    → "python", "machine" are LESS important (common across docs)

Implementation: Uses only math.log. No sklearn!
"""

import math


def compute_idf(documents):
    """
    Calculate Inverse Document Frequency for every word across all documents.

    Formula: IDF(t) = log(N / (1 + df(t)))

    Words that appear in fewer documents get higher IDF scores,
    meaning they are more useful for distinguishing documents.

    Args:
        documents (list[list[str]]): List of documents, where each document
                                      is a list of preprocessed tokens.
                                      Example: [['python', 'ml'], ['java', 'ml']]

    Returns:
        dict[str, float]: Mapping of each word to its IDF score

    Example:
        >>> docs = [['python', 'ml'], ['java', 'ml'], ['python', 'web']]
        >>> idf = compute_idf(docs)
        >>> idf['ml']     # appears in 2/3 docs → lower IDF
        0.0
        >>> idf['java']   # appears in 1/3 docs → higher IDF
        0.405
    """
    if not documents:
        return {}

    N = len(documents)  # Total number of documents

    # Step 1: Count Document Frequency (df) — how many documents contain each word
    # IMPORTANT: Count each word only ONCE per document (use set)
    doc_freq = {}
    for doc_tokens in documents:
        unique_words = set(doc_tokens)  # Deduplicate within document
        for word in unique_words:
            doc_freq[word] = doc_freq.get(word, 0) + 1

    # Step 2: Calculate IDF for each word
    idf = {}
    for word, df in doc_freq.items():
        # Formula: IDF(t) = log(N / (1 + df(t)))
        # The +1 prevents division by zero
        # log() is natural logarithm
        idf[word] = math.log(N / (1 + df))

    return idf


def compute_document_frequency(documents):
    """
    Calculate raw Document Frequency — just the count of documents containing each word.

    Useful for debugging and for the Jupyter notebook walkthrough.

    Args:
        documents (list[list[str]]): List of token lists

    Returns:
        dict[str, int]: Mapping of each word to number of documents it appears in

    Example:
        >>> docs = [['python', 'ml'], ['java', 'ml']]
        >>> compute_document_frequency(docs)
        {'python': 1, 'ml': 2, 'java': 1}
    """
    doc_freq = {}
    for doc_tokens in documents:
        unique_words = set(doc_tokens)
        for word in unique_words:
            doc_freq[word] = doc_freq.get(word, 0) + 1
    return doc_freq
