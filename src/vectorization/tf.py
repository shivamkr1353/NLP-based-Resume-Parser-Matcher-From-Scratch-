"""
Term Frequency (TF) — FROM SCRATCH
====================================
Term Frequency measures how often a word appears in a single document.
It's the first half of the TF-IDF calculation.

Formula:
    TF(t, d) = count(t in d) / total_terms(d)

    Where:
        t = a specific term (word)
        d = a specific document
        count(t in d) = number of times term t appears in document d
        total_terms(d) = total number of terms in document d

Example:
    Document: "python machine learning python data python"
    Total terms = 6

    TF("python")   = 3/6 = 0.500
    TF("machine")  = 1/6 = 0.167
    TF("learning") = 1/6 = 0.167
    TF("data")     = 1/6 = 0.167

Implementation: Uses only collections.Counter for counting. No sklearn!
"""

from collections import Counter


def compute_tf(tokens):
    """
    Calculate Term Frequency for each word in a document.

    Formula: TF(t, d) = count(t in d) / total_terms(d)

    A higher TF means the word appears more frequently in this document.
    TF is normalized by document length so that longer documents don't
    get artificially higher scores.

    Args:
        tokens (list[str]): List of preprocessed word tokens from one document

    Returns:
        dict[str, float]: Mapping of each word to its TF score

    Example:
        >>> compute_tf(['python', 'machine', 'learning', 'python', 'data', 'python'])
        {'python': 0.5, 'machine': 0.1667, 'learning': 0.1667, 'data': 0.1667}
    """
    if not tokens:
        return {}

    # Count occurrences of each word
    word_counts = Counter(tokens)

    # Total number of words in the document
    total_words = len(tokens)

    # Calculate TF: count / total
    tf = {}
    for word, count in word_counts.items():
        tf[word] = count / total_words

    return tf


def compute_raw_tf(tokens):
    """
    Calculate raw (unnormalized) Term Frequency — just the counts.

    This is useful for debugging and for the Jupyter notebook walkthrough
    to show the intermediate step before normalization.

    Args:
        tokens (list[str]): List of preprocessed word tokens

    Returns:
        dict[str, int]: Mapping of each word to its raw count

    Example:
        >>> compute_raw_tf(['python', 'ml', 'python', 'data'])
        {'python': 2, 'ml': 1, 'data': 1}
    """
    return dict(Counter(tokens))
