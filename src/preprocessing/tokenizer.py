"""
Tokenizer — FROM SCRATCH
========================
Tokenization is the process of splitting raw text into individual words (tokens).
This is the first step in any NLP pipeline.

Implementation: Uses only Python's built-in `re` module for regex-based text cleaning.
NO external NLP libraries (no spaCy, no NLTK).

Example:
    >>> tokenize("Machine Learning Engineer, 5+ years!")
    ['machine', 'learning', 'engineer', '5', 'years']
"""

import re


def tokenize(text):
    """
    Split raw text into clean word tokens.

    Steps:
        1. Convert to lowercase (case normalization)
        2. Remove URLs (http/https links)
        3. Remove email addresses
        4. Remove phone numbers
        5. Remove special characters (keep only letters, digits, spaces)
        6. Collapse multiple spaces
        7. Split into individual word tokens
        8. Filter out empty strings and single-character tokens (except meaningful ones)

    Args:
        text (str): Raw input text (resume or job description)

    Returns:
        list[str]: List of clean, lowercase word tokens

    Example:
        >>> tokenize("Senior Python Developer | john@email.com | 555-1234")
        ['senior', 'python', 'developer']
    """
    if not text or not isinstance(text, str):
        return []

    # Step 1: Convert to lowercase
    text = text.lower()

    # Step 2: Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Step 3: Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', ' ', text)

    # Step 4: Remove phone numbers (various formats)
    text = re.sub(r'[\+]?[\d]{1,3}[-.\s]?[\d]{3}[-.\s]?[\d]{3,4}[-.\s]?[\d]{3,4}', ' ', text)

    # Step 5: Remove special characters — keep only letters, digits, spaces
    # This removes: @, #, $, %, ^, &, *, (, ), etc.
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Step 6: Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 7: Split into tokens
    tokens = text.split()

    # Step 8: Filter out very short tokens (single chars) except meaningful ones
    # Keep single digits and single meaningful letters
    meaningful_singles = {'c', 'r', 'j'}  # C (language), R (language), J (Java shorthand)
    tokens = [t for t in tokens if len(t) > 1 or t in meaningful_singles]

    return tokens


def tokenize_preserve_phrases(text):
    """
    Tokenize but also detect common multi-word technical phrases.

    This helps match phrases like "machine learning" as a single concept
    rather than two separate words.

    Args:
        text (str): Raw input text

    Returns:
        tuple: (single_tokens, detected_phrases)

    Example:
        >>> tokens, phrases = tokenize_preserve_phrases("Experience in machine learning and deep learning")
        >>> phrases
        ['machine learning', 'deep learning']
    """
    # Common multi-word technical phrases in tech/ML domain
    PHRASES = [
        'machine learning', 'deep learning', 'natural language processing',
        'computer vision', 'data science', 'data analysis', 'data engineering',
        'artificial intelligence', 'neural network', 'neural networks',
        'web development', 'full stack', 'front end', 'back end',
        'project management', 'version control', 'continuous integration',
        'continuous deployment', 'object oriented', 'test driven',
        'agile methodology', 'cloud computing', 'big data',
        'business intelligence', 'quality assurance', 'user experience',
        'software engineering', 'software development',
    ]

    text_lower = text.lower() if text else ""

    # Detect phrases
    detected_phrases = []
    for phrase in PHRASES:
        if phrase in text_lower:
            detected_phrases.append(phrase)

    # Also return normal single tokens
    single_tokens = tokenize(text)

    return single_tokens, detected_phrases
