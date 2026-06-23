"""
Porter Stemmer — FROM SCRATCH
==============================
Stemming reduces words to their root/base form so that different variations
of the same word are treated as identical during TF-IDF comparison.

    "running"     → "run"
    "machines"    → "machin"
    "engineering" → "engineer"
    "experienced" → "experienc"

Algorithm: Simplified Porter Stemmer with ~25 suffix-stripping rules.
The full Porter Stemmer has ~60 rules across 5 steps. We implement the
most impactful rules that cover the vast majority of English word forms.

Implementation: Pure Python string operations. NO nltk.stem.PorterStemmer!

Reference: Porter, M.F. (1980) "An algorithm for suffix stripping"
"""


def _measure(word):
    """
    Calculate the 'measure' of a word — the number of vowel-consonant sequences.

    The measure (m) is used to determine if a stem is long enough after
    stripping a suffix. A higher measure means a longer stem.

    Pattern: [C](VC){m}[V]
        where C = consonant sequence, V = vowel sequence

    Examples:
        "tr"      → m=0  (just consonants)
        "tree"    → m=0  (C-V, no VC pair)
        "trouble" → m=1  (C-V-C-V = one VC pair)
        "troubles"→ m=2

    Args:
        word (str): The word to measure

    Returns:
        int: The measure value
    """
    vowels = 'aeiou'
    # Build a pattern string: 'v' for vowels, 'c' for consonants
    # Handle 'y' — it's a vowel when preceded by a consonant
    pattern = []
    for i, char in enumerate(word):
        if char in vowels:
            pattern.append('v')
        elif char == 'y' and i > 0 and word[i - 1] not in vowels:
            pattern.append('v')
        else:
            pattern.append('c')

    # Count VC transitions
    pattern_str = ''.join(pattern)
    # Collapse consecutive same letters
    collapsed = ''
    for char in pattern_str:
        if not collapsed or collapsed[-1] != char:
            collapsed += char

    # Count 'vc' pairs in the collapsed pattern
    m = collapsed.count('vc')
    return m


def _has_vowel(word):
    """Check if the word contains at least one vowel."""
    vowels = 'aeiou'
    for i, char in enumerate(word):
        if char in vowels:
            return True
        if char == 'y' and i > 0:
            return True
    return False


def _ends_double_consonant(word):
    """Check if the word ends with a double consonant (e.g., 'll', 'ss', 'zz')."""
    if len(word) < 2:
        return False
    return word[-1] == word[-2] and word[-1] not in 'aeiou'


def _ends_cvc(word):
    """
    Check if the word ends with consonant-vowel-consonant
    where the last consonant is NOT w, x, or y.
    """
    vowels = 'aeiou'
    if len(word) < 3:
        return False
    c1 = word[-3] not in vowels  # consonant
    v = word[-2] in vowels       # vowel
    c2 = word[-1] not in vowels and word[-1] not in 'wxy'  # consonant (not w/x/y)
    return c1 and v and c2


def stem(word):
    """
    Apply simplified Porter Stemmer rules to reduce a word to its stem.

    The algorithm processes the word through multiple steps, each applying
    specific suffix-stripping rules. A suffix is only removed if the
    remaining stem is long enough (measured by the 'measure' function).

    Rules are applied in order from longest suffix to shortest within
    each step, ensuring greedy matching.

    Args:
        word (str): A single lowercase word token

    Returns:
        str: The stemmed word

    Examples:
        >>> stem("running")
        'run'
        >>> stem("machines")
        'machin'
        >>> stem("generalization")
        'gener'
    """
    if len(word) <= 2:
        return word

    # ============================================================
    # STEP 1: Handle plurals and past participles
    # ============================================================

    # Rule: "sses" → "ss"  (e.g., "caresses" → "caress")
    if word.endswith('sses'):
        word = word[:-2]

    # Rule: "ies" → "i"  (e.g., "ponies" → "poni")
    elif word.endswith('ies'):
        word = word[:-2]

    # Rule: "ss" → "ss"  (e.g., "caress" → "caress" — no change)
    elif word.endswith('ss'):
        pass

    # Rule: "s" → ""  (e.g., "cats" → "cat")
    elif word.endswith('s'):
        word = word[:-1]

    # ============================================================
    # STEP 2: Handle "-ed" and "-ing" endings
    # ============================================================

    # Rule: "eed" → "ee" if measure > 0
    if word.endswith('eed'):
        stem_part = word[:-3]
        if _measure(stem_part) > 0:
            word = word[:-1]  # Remove just the 'd'

    # Rule: "ed" → "" if stem contains vowel
    elif word.endswith('ed'):
        stem_part = word[:-2]
        if _has_vowel(stem_part):
            word = stem_part
            # Post-processing after removing -ed
            if word.endswith('at') or word.endswith('bl') or word.endswith('iz'):
                word += 'e'  # "conflated" → "conflat" → "conflate"
            elif _ends_double_consonant(word) and not word.endswith(('l', 's', 'z')):
                word = word[:-1]  # "hopping" → "hopp" → "hop"
            elif _measure(word) == 1 and _ends_cvc(word):
                word += 'e'

    # Rule: "ing" → "" if stem contains vowel
    elif word.endswith('ing'):
        stem_part = word[:-3]
        if _has_vowel(stem_part):
            word = stem_part
            # Same post-processing as -ed
            if word.endswith('at') or word.endswith('bl') or word.endswith('iz'):
                word += 'e'
            elif _ends_double_consonant(word) and not word.endswith(('l', 's', 'z')):
                word = word[:-1]
            elif _measure(word) == 1 and _ends_cvc(word):
                word += 'e'

    # ============================================================
    # STEP 3: Handle "y" → "i"
    # ============================================================
    if word.endswith('y') and _has_vowel(word[:-1]) and len(word) > 2:
        word = word[:-1] + 'i'

    # ============================================================
    # STEP 4: Handle longer suffixes (map to shorter forms)
    # Condition: Only apply if measure of stem > 0
    # ============================================================
    step4_rules = [
        ('ational', 'ate'),    # "relational" → "relate"
        ('tional', 'tion'),    # "conditional" → "condition"
        ('enci', 'ence'),      # "valenci" → "valence"
        ('anci', 'ance'),      # "hesitanci" → "hesitance"
        ('izer', 'ize'),       # "customizer" → "customize"
        ('ization', 'ize'),    # "digitization" → "digitize"
        ('ation', 'ate'),      # "activation" → "activate"
        ('ator', 'ate'),       # "operator" → "operate"
        ('alism', 'al'),       # "feudalism" → "feudal"
        ('iveness', 'ive'),    # "effectiveness" → "effective"
        ('fulness', 'ful'),    # "hopefulness" → "hopeful"
        ('ousness', 'ous'),    # "callousness" → "callous"
        ('aliti', 'al'),       # "formaliti" → "formal"
        ('iviti', 'ive'),      # "sensitiviti" → "sensitive"
        ('biliti', 'ble'),     # "sensibiliti" → "sensible"
    ]

    for suffix, replacement in step4_rules:
        if word.endswith(suffix):
            stem_part = word[:-len(suffix)]
            if _measure(stem_part) > 0:
                word = stem_part + replacement
            break  # Only apply first matching rule

    # ============================================================
    # STEP 5: Remove common suffixes
    # Condition: Only apply if measure of stem > 0
    # ============================================================
    step5_rules = [
        ('icate', 'ic'),   # "triplicate" → "triplic"
        ('ative', ''),     # "formative" → "form"
        ('alize', 'al'),   # "formalize" → "formal"
        ('iciti', 'ic'),   # "electriciti" → "electric"
        ('ical', 'ic'),    # "electrical" → "electric"
        ('ful', ''),       # "hopeful" → "hope"
        ('ness', ''),      # "goodness" → "good"
    ]

    for suffix, replacement in step5_rules:
        if word.endswith(suffix):
            stem_part = word[:-len(suffix)]
            if _measure(stem_part) > 0:
                word = stem_part + replacement
            break

    # ============================================================
    # STEP 6: Final cleanup — remove trailing 'e'
    # Condition: measure > 1, or (measure == 1 and NOT cvc pattern)
    # ============================================================
    if word.endswith('e'):
        stem_part = word[:-1]
        m = _measure(stem_part)
        if m > 1:
            word = stem_part
        elif m == 1 and not _ends_cvc(stem_part):
            word = stem_part

    # Remove double consonant at end if measure > 1
    if _ends_double_consonant(word) and _measure(word) > 1:
        if word.endswith('l'):
            word = word[:-1]

    return word


def stem_tokens(tokens):
    """
    Apply stemming to a list of tokens.

    Args:
        tokens (list[str]): List of word tokens

    Returns:
        list[str]: List of stemmed tokens

    Example:
        >>> stem_tokens(['running', 'machines', 'learning'])
        ['run', 'machin', 'learn']
    """
    return [stem(token) for token in tokens]
