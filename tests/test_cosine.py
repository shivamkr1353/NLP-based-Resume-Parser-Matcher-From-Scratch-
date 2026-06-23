"""Tests for from-scratch Cosine Similarity."""

import sys
import os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.similarity.cosine import cosine_similarity, cosine_similarity_detailed


class TestCosineSimilarity:
    """Tests for the from-scratch cosine similarity implementation."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity = 1.0"""
        vec = {'python': 0.5, 'ml': 0.3, 'data': 0.2}
        assert abs(cosine_similarity(vec, vec) - 1.0) < 1e-10

    def test_orthogonal_vectors(self):
        """Vectors with no common words should have similarity = 0.0"""
        vec_a = {'python': 0.5, 'ml': 0.3}
        vec_b = {'java': 0.4, 'spring': 0.6}
        assert cosine_similarity(vec_a, vec_b) == 0.0

    def test_partial_overlap(self):
        """Vectors with some common words should have 0 < similarity < 1"""
        vec_a = {'python': 0.3, 'ml': 0.5, 'data': 0.2}
        vec_b = {'python': 0.4, 'ml': 0.3, 'java': 0.6}
        sim = cosine_similarity(vec_a, vec_b)
        assert 0.0 < sim < 1.0

    def test_worked_example_from_research_notes(self):
        """
        Verify against the worked example from research_notes.md:
        JD  = {'python': 0.3, 'ml': 0.5, 'data': 0.2}
        Res = {'python': 0.4, 'ml': 0.3, 'java': 0.6}

        Dot product = 0.3*0.4 + 0.5*0.3 + 0.2*0 + 0*0.6 = 0.12 + 0.15 = 0.27
        Mag JD  = sqrt(0.09 + 0.25 + 0.04) = sqrt(0.38)
        Mag Res = sqrt(0.16 + 0.09 + 0.36) = sqrt(0.61)
        Similarity = 0.27 / (sqrt(0.38) * sqrt(0.61))
        """
        jd = {'python': 0.3, 'ml': 0.5, 'data': 0.2}
        resume = {'python': 0.4, 'ml': 0.3, 'java': 0.6}

        expected_dot = 0.3 * 0.4 + 0.5 * 0.3  # = 0.27
        expected_mag_a = math.sqrt(0.3**2 + 0.5**2 + 0.2**2)
        expected_mag_b = math.sqrt(0.4**2 + 0.3**2 + 0.6**2)
        expected = expected_dot / (expected_mag_a * expected_mag_b)

        result = cosine_similarity(jd, resume)
        assert abs(result - expected) < 1e-6

    def test_empty_vectors(self):
        """Empty vectors should return 0.0"""
        assert cosine_similarity({}, {}) == 0.0
        assert cosine_similarity({'python': 0.5}, {}) == 0.0
        assert cosine_similarity({}, {'java': 0.3}) == 0.0

    def test_zero_values(self):
        """Vectors with all zero values should return 0.0"""
        assert cosine_similarity({'a': 0}, {'a': 0}) == 0.0

    def test_symmetry(self):
        """cos(A, B) should equal cos(B, A)"""
        vec_a = {'python': 0.3, 'ml': 0.5}
        vec_b = {'python': 0.4, 'java': 0.6}
        assert abs(cosine_similarity(vec_a, vec_b) - cosine_similarity(vec_b, vec_a)) < 1e-10


class TestCosineSimilarityDetailed:
    """Tests for the detailed cosine similarity (with intermediate values)."""

    def test_returns_all_fields(self):
        vec_a = {'python': 0.3, 'ml': 0.5}
        vec_b = {'python': 0.4, 'java': 0.6}
        result = cosine_similarity_detailed(vec_a, vec_b)

        assert 'similarity' in result
        assert 'dot_product' in result
        assert 'magnitude_a' in result
        assert 'magnitude_b' in result
        assert 'common_terms' in result
        assert 'term_contributions' in result

    def test_common_terms_detected(self):
        vec_a = {'python': 0.3, 'ml': 0.5, 'data': 0.2}
        vec_b = {'python': 0.4, 'ml': 0.3, 'java': 0.6}
        result = cosine_similarity_detailed(vec_a, vec_b)

        assert 'python' in result['common_terms']
        assert 'ml' in result['common_terms']
        assert 'java' not in result['common_terms']
        assert 'data' not in result['common_terms']

    def test_detailed_matches_simple(self):
        """Detailed and simple versions should return the same similarity."""
        vec_a = {'python': 0.3, 'ml': 0.5, 'data': 0.2}
        vec_b = {'python': 0.4, 'ml': 0.3, 'java': 0.6}

        simple = cosine_similarity(vec_a, vec_b)
        detailed = cosine_similarity_detailed(vec_a, vec_b)

        assert abs(simple - detailed['similarity']) < 1e-4


# Run tests
if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
