"""
Cosine Similarity — FROM SCRATCH
==================================
Cosine Similarity measures how similar two documents are by comparing
the angle between their TF-IDF vectors.

Formula:
    cos(A, B) = (A · B) / (|A| × |B|)

    Where:
        A · B  = Dot Product = Σ(Aᵢ × Bᵢ)  — multiply matching dimensions and sum
        |A|    = Magnitude   = √(Σ Aᵢ²)     — the "length" of the vector
        |B|    = Magnitude   = √(Σ Bᵢ²)

    Result range: 0 (completely different) to 1 (identical)

Visual Intuition:
    Imagine two arrows (vectors) in multi-dimensional space:
    - Same direction  → angle = 0°   → cos(0°) = 1.0  → PERFECT MATCH
    - Perpendicular   → angle = 90°  → cos(90°) = 0.0 → NO MATCH
    - Opposite        → angle = 180° → cos(180°) = -1  → OPPOSITE

    In practice, TF-IDF vectors are always non-negative, so
    cosine similarity ranges from 0 to 1.

Implementation: Uses only math.sqrt. No sklearn.metrics.pairwise!
"""

import math


def cosine_similarity(vec_a, vec_b):
    """
    Calculate cosine similarity between two TF-IDF vectors.

    Formula: cos(A, B) = (A · B) / (|A| × |B|)

    Both vectors are represented as dictionaries {word: tfidf_score}.
    Words missing from one vector are treated as 0.

    Args:
        vec_a (dict[str, float]): TF-IDF vector for document A (e.g., Job Description)
        vec_b (dict[str, float]): TF-IDF vector for document B (e.g., Resume)

    Returns:
        float: Similarity score between 0.0 (no match) and 1.0 (perfect match)

    Example:
        >>> jd  = {'python': 0.3, 'ml': 0.5, 'data': 0.2}
        >>> res = {'python': 0.4, 'ml': 0.3, 'java': 0.6}
        >>> cosine_similarity(jd, res)
        0.561  # 56.1% match
    """
    if not vec_a or not vec_b:
        return 0.0

    # Step 1: Find all unique words across both vectors
    all_words = set(vec_a.keys()) | set(vec_b.keys())

    # Step 2: Calculate Dot Product (A · B)
    # Formula: Σ(Aᵢ × Bᵢ) for all dimensions
    dot_product = 0.0
    for word in all_words:
        a_val = vec_a.get(word, 0.0)  # 0 if word not in this document
        b_val = vec_b.get(word, 0.0)
        dot_product += a_val * b_val

    # Step 3: Calculate Magnitude (|A|)
    # Formula: √(Σ Aᵢ²)
    magnitude_a = math.sqrt(sum(val ** 2 for val in vec_a.values()))

    # Step 4: Calculate Magnitude (|B|)
    magnitude_b = math.sqrt(sum(val ** 2 for val in vec_b.values()))

    # Step 5: Handle edge case — avoid division by zero
    # (happens if a document has no meaningful words)
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    # Step 6: Cosine Similarity = dot_product / (mag_a × mag_b)
    similarity = dot_product / (magnitude_a * magnitude_b)

    return similarity


def cosine_similarity_detailed(vec_a, vec_b):
    """
    Calculate cosine similarity WITH intermediate steps (for the "How It Works" tab).

    Returns the same similarity score plus all intermediate calculations,
    which can be displayed in the Streamlit UI to show the math.

    Args:
        vec_a (dict[str, float]): TF-IDF vector for document A
        vec_b (dict[str, float]): TF-IDF vector for document B

    Returns:
        dict: {
            'similarity': float,        # Final cosine similarity
            'dot_product': float,       # A · B value
            'magnitude_a': float,       # |A| value
            'magnitude_b': float,       # |B| value
            'common_terms': list,       # Words present in both vectors
            'term_contributions': dict, # Each word's contribution to dot product
        }
    """
    if not vec_a or not vec_b:
        return {
            'similarity': 0.0,
            'dot_product': 0.0,
            'magnitude_a': 0.0,
            'magnitude_b': 0.0,
            'common_terms': [],
            'term_contributions': {},
        }

    all_words = set(vec_a.keys()) | set(vec_b.keys())

    # Calculate dot product with per-term breakdown
    dot_product = 0.0
    term_contributions = {}
    common_terms = []

    for word in all_words:
        a_val = vec_a.get(word, 0.0)
        b_val = vec_b.get(word, 0.0)
        contribution = a_val * b_val
        dot_product += contribution

        if a_val > 0 and b_val > 0:
            common_terms.append(word)
            term_contributions[word] = {
                'a_val': round(a_val, 4),
                'b_val': round(b_val, 4),
                'contribution': round(contribution, 4),
            }

    magnitude_a = math.sqrt(sum(val ** 2 for val in vec_a.values()))
    magnitude_b = math.sqrt(sum(val ** 2 for val in vec_b.values()))

    if magnitude_a == 0 or magnitude_b == 0:
        similarity = 0.0
    else:
        similarity = dot_product / (magnitude_a * magnitude_b)

    # Sort common terms by contribution (highest first)
    common_terms.sort(key=lambda w: term_contributions.get(w, {}).get('contribution', 0), reverse=True)

    return {
        'similarity': round(similarity, 6),
        'dot_product': round(dot_product, 6),
        'magnitude_a': round(magnitude_a, 6),
        'magnitude_b': round(magnitude_b, 6),
        'common_terms': common_terms,
        'term_contributions': term_contributions,
    }
