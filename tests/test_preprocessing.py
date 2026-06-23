"""Tests for the preprocessing pipeline (tokenizer, stopwords, stemmer)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing.tokenizer import tokenize, tokenize_preserve_phrases
from src.preprocessing.stopwords import remove_stopwords, STOP_WORDS
from src.preprocessing.stemmer import stem, stem_tokens


class TestTokenizer:
    """Tests for the from-scratch tokenizer."""

    def test_basic_tokenization(self):
        result = tokenize("Machine Learning Engineer")
        assert result == ['machine', 'learning', 'engineer']

    def test_removes_special_chars(self):
        result = tokenize("Python/Java — C++ | SQL!")
        # C++ becomes c, which is kept as meaningful single
        assert 'python' in result
        assert 'java' in result
        assert 'sql' in result

    def test_removes_urls(self):
        result = tokenize("Visit https://example.com for more info")
        assert 'https' not in result
        assert 'example' not in result
        assert 'visit' in result

    def test_removes_emails(self):
        result = tokenize("Contact john@email.com for details")
        assert 'john' not in result
        assert 'email' not in result
        assert 'contact' in result

    def test_handles_empty_input(self):
        assert tokenize("") == []
        assert tokenize(None) == []

    def test_handles_numbers(self):
        result = tokenize("5 years of experience in Python 3")
        assert '5' not in result  # single char filtered
        assert 'years' in result
        assert 'python' in result

    def test_lowercase(self):
        result = tokenize("PYTHON Django Flask")
        assert all(t.islower() for t in result)

    def test_phrase_detection(self):
        tokens, phrases = tokenize_preserve_phrases("Experience in machine learning and deep learning")
        assert 'machine learning' in phrases
        assert 'deep learning' in phrases


class TestStopWords:
    """Tests for the from-scratch stop words removal."""

    def test_removes_common_words(self):
        tokens = ['the', 'python', 'developer', 'is', 'in', 'ml']
        result = remove_stopwords(tokens)
        assert 'the' not in result
        assert 'is' not in result
        assert 'in' not in result
        assert 'python' in result
        assert 'ml' in result

    def test_removes_resume_specific_words(self):
        tokens = ['experience', 'skills', 'python', 'responsibilities', 'tensorflow']
        result = remove_stopwords(tokens)
        assert 'experience' not in result  # resume stop word
        assert 'skills' not in result      # resume stop word
        assert 'python' in result
        assert 'tensorflow' in result

    def test_preserves_technical_terms(self):
        tokens = ['python', 'tensorflow', 'kubernetes', 'docker', 'react']
        result = remove_stopwords(tokens)
        assert result == tokens  # None should be removed

    def test_stop_words_set_exists(self):
        assert len(STOP_WORDS) > 50  # We should have a substantial list

    def test_handles_empty_list(self):
        assert remove_stopwords([]) == []


class TestStemmer:
    """Tests for the from-scratch Porter Stemmer."""

    def test_plural_s(self):
        assert stem('cats') == 'cat'
        assert stem('dogs') == 'dog'

    def test_plural_ies(self):
        assert stem('ponies') == 'poni'

    def test_plural_sses(self):
        assert stem('caresses') == 'caress'

    def test_ed_suffix(self):
        result = stem('walked')
        assert 'walk' in result or result == 'walk'

    def test_ing_suffix(self):
        result = stem('running')
        assert 'run' in result or result == 'run'

    def test_short_words_unchanged(self):
        assert stem('ml') == 'ml'
        assert stem('ai') == 'ai'
        assert stem('go') == 'go'

    def test_stem_tokens_list(self):
        result = stem_tokens(['running', 'machines', 'learning'])
        assert len(result) == 3
        # All should be different from original (stemmed)
        assert result != ['running', 'machines', 'learning']

    def test_tion_suffix(self):
        result = stem('activation')
        # Should be stemmed somehow
        assert len(result) < len('activation')

    def test_ness_suffix(self):
        result = stem('goodness')
        assert result == 'good'


# Run tests
if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
