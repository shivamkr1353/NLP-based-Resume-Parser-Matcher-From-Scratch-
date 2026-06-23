"""
Stop Words — FROM SCRATCH
=========================
Stop words are common words that appear frequently in text but carry little
meaningful information for document comparison (e.g., "the", "is", "and").

Removing them helps the TF-IDF algorithm focus on words that actually
distinguish one document from another.

Implementation: Hand-crafted list of ~120 stop words. NO nltk.corpus.stopwords!

Why hand-craft?
    - Professor requires from-scratch implementation
    - We can customize for the recruiter/resume domain
    - We add domain-specific stop words like "resume", "responsibilities"
"""


# ============================================================
# HAND-CRAFTED STOP WORDS LIST
# Organized by category for readability and maintainability
# ============================================================

# Standard English stop words
_ARTICLES = {'a', 'an', 'the'}

_PRONOUNS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
}

_PREPOSITIONS = {
    'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of',
    'about', 'above', 'after', 'before', 'between', 'into', 'through',
    'during', 'under', 'over', 'below', 'up', 'down', 'out', 'off',
    'against', 'along', 'across', 'around', 'among', 'upon', 'within',
}

_CONJUNCTIONS = {
    'and', 'or', 'but', 'nor', 'so', 'yet', 'both', 'either', 'neither',
}

_AUXILIARY_VERBS = {
    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing',
    'will', 'would', 'shall', 'should',
    'may', 'might', 'must', 'can', 'could',
}

_DETERMINERS = {
    'this', 'that', 'these', 'those',
    'each', 'every', 'all', 'any', 'few', 'more', 'most',
    'some', 'such', 'no', 'not', 'only', 'own', 'same',
    'other', 'another', 'much', 'many',
}

_ADVERBS = {
    'very', 'just', 'also', 'too', 'quite', 'rather',
    'already', 'always', 'never', 'often', 'sometimes',
    'here', 'there', 'where', 'when', 'how', 'why', 'what', 'which', 'who', 'whom',
    'then', 'than', 'now', 'well', 'even', 'still',
}

_MISC = {
    'if', 'because', 'as', 'until', 'while', 'although',
    'though', 'since', 'unless', 'whether',
    'again', 'once', 'further',
    'etc', 'eg', 'ie',
}

# ----- DOMAIN-SPECIFIC: Resume/Recruiter stop words -----
# These words appear in almost every resume/JD and don't help distinguish candidates
_RESUME_STOP_WORDS = {
    'resume', 'cv', 'curriculum', 'vitae',
    'responsibilities', 'responsible', 'responsibility',
    'requirements', 'required', 'require',
    'qualifications', 'qualified',
    'experience', 'experienced',
    'skills', 'skill',
    'education', 'university', 'college', 'degree',
    'work', 'working', 'worked',
    'role', 'position', 'job',
    'company', 'organization', 'firm',
    'team', 'teams',
    'ability', 'able',
    'including', 'include', 'includes',
    'using', 'use', 'used', 'utilize', 'utilized',
    'based', 'related',
    'strong', 'excellent', 'good', 'great',
    'new', 'various', 'multiple',
    'ensure', 'ensuring',
    'provide', 'providing', 'provided',
    'develop', 'developing', 'developed', 'development',
    'manage', 'managing', 'managed', 'management',
    'support', 'supporting', 'supported',
    'year', 'years', 'month', 'months',
    'knowledge', 'understanding',
    'environment', 'environments',
    'professional', 'proficiency', 'proficient',
}

# Combine all into one master set
STOP_WORDS = (
    _ARTICLES | _PRONOUNS | _PREPOSITIONS | _CONJUNCTIONS |
    _AUXILIARY_VERBS | _DETERMINERS | _ADVERBS | _MISC | _RESUME_STOP_WORDS
)


def remove_stopwords(tokens):
    """
    Remove stop words from a list of tokens.

    Stop words are common words that appear frequently but carry little
    semantic meaning for document comparison. Removing them allows TF-IDF
    to focus on distinctive, meaningful terms.

    Args:
        tokens (list[str]): List of word tokens (should be lowercase)

    Returns:
        list[str]: Filtered list with stop words removed

    Example:
        >>> remove_stopwords(['the', 'python', 'developer', 'is', 'experienced', 'in', 'ml'])
        ['python', 'developer', 'ml']
    """
    return [word for word in tokens if word not in STOP_WORDS]


def get_stop_words():
    """Return the full set of stop words (useful for inspection/debugging)."""
    return STOP_WORDS.copy()


def add_custom_stop_words(words):
    """
    Add custom stop words to the global set.

    Args:
        words (list[str]): Additional words to treat as stop words

    Example:
        >>> add_custom_stop_words(['internship', 'intern'])
    """
    STOP_WORDS.update(words)
