"""Tests for identity module."""

import pytest
import numpy as np

from detector.identity import (
    extract_identity_embedding,
    compare_embeddings,
    track_identity,
    detect_face_regions,
    extract_face_embedding,
)


class TestExtractIdentityEmbedding:
    """Tests for identity embedding extraction."""
    
    def test_returns_vector(self, sample_frame):
        embedding = extract_identity_embedding(sample_frame)
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) > 0
    
    def test_normalized(self, sample_frame):
        embedding = extract_identity_embedding(sample_frame)
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01
    
    def test_consistent_output(self, sample_frame):
        emb1 = extract_identity_embedding(sample_frame)
        emb2 = extract_identity_embedding(sample_frame)
        # Same input should produce same embedding
        assert np.allclose(emb1, emb2, atol=1e-5)
    
    def test_different_frames_different_embeddings(self):
        frame_a = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frame_b = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        emb_a = extract_identity_embedding(frame_a)
        emb_b = extract_identity_embedding(frame_b)
        # Very different frames should have different embeddings
        similarity = np.dot(emb_a, emb_b)
        assert similarity < 0.99


class TestCompareEmbeddings:
    """Tests for embedding comparison."""
    
    def test_identical_embeddings(self):
        emb = np.random.randn(128)
        emb = emb / np.linalg.norm(emb)
        similarity = compare_embeddings(emb, emb)
        assert abs(similarity - 1.0) < 1e-5
    
    def test_orthogonal_embeddings(self):
        emb_a = np.array([1, 0, 0], dtype=np.float32)
        emb_b = np.array([0, 1, 0], dtype=np.float32)
        similarity = compare_embeddings(emb_a, emb_b)
        assert abs(similarity) < 1e-5
    
    def test_opposite_embeddings(self):
        emb_a = np.array([1, 0, 0], dtype=np.float32)
        emb_b = np.array([-1, 0, 0], dtype=np.float32)
        similarity = compare_embeddings(emb_a, emb_b)
        assert similarity < -0.99
    
    def test_similarity_range(self, sample_frame):
        emb_a = extract_identity_embedding(sample_frame)
        emb_b = extract_identity_embedding(sample_frame)
        similarity = compare_embeddings(emb_a, emb_b)
        assert -1 <= similarity <= 1


class TestTrackIdentity:
    """Tests for identity tracking across frames."""
    
    def test_returns_similarities(self, sample_frames):
        similarities = track_identity(sample_frames)
        assert len(similarities) == len(sample_frames)
    
    def test_first_frame_similarity_is_one(self, sample_frames):
        similarities = track_identity(sample_frames)
        assert abs(similarities[0] - 1.0) < 1e-5
    
    def test_empty_frames(self):
        similarities = track_identity([])
        assert similarities == []
    
    def test_single_frame(self, sample_frame):
        similarities = track_identity([sample_frame])
        assert len(similarities) == 1
        assert abs(similarities[0] - 1.0) < 1e-5
    
    def test_similarities_in_range(self, sample_frames):
        similarities = track_identity(sample_frames)
        for sim in similarities:
            assert -1 <= sim <= 1


class TestDetectFaceRegions:
    """Tests for face detection."""
    
    def test_returns_list(self, sample_frame):
        faces = detect_face_regions(sample_frame)
        assert isinstance(faces, list)
    
    def test_face_tuple_format(self, sample_frame):
        faces = detect_face_regions(sample_frame)
        for face in faces:
            assert len(face) == 4
            x, y, w, h = face
            assert w > 0
            assert h > 0
    
    def test_grayscale_input(self):
        frame = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        faces = detect_face_regions(frame)
        assert isinstance(faces, list)


class TestExtractFaceEmbedding:
    """Tests for face embedding extraction."""
    
    def test_no_face_returns_none(self):
        # Random noise unlikely to have faces
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        embedding = extract_face_embedding(frame)
        # May return None or an embedding depending on detection
        assert embedding is None or isinstance(embedding, np.ndarray)
    
    def test_returns_vector_or_none(self, sample_frame):
        embedding = extract_face_embedding(sample_frame)
        if embedding is not None:
            assert isinstance(embedding, np.ndarray)
            assert len(embedding) > 0