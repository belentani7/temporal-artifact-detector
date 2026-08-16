"""Identity embedding and tracking for drift detection."""

import numpy as np
import cv2


def _mock_extract_embedding(image: np.ndarray) -> np.ndarray:
    """Mock identity embedding using color histogram and edge features.
    
    Real implementation would use ArcFace or similar face recognition model.
    This provides a reasonable approximation for testing.
    """
    # Resize to standard size
    resized = cv2.resize(image, (112, 112))
    
    # Compute color histogram
    if len(resized.shape) == 3:
        hist_b = cv2.calcHist([resized], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([resized], [1], None, [32], [0, 256])
        hist_r = cv2.calcHist([resized], [2], None, [32], [0, 256])
        hist = np.concatenate([hist_b, hist_g, hist_r]).flatten()
    else:
        hist = cv2.calcHist([resized], [0], None, [64], [0, 256]).flatten()
    
    # Normalize histogram
    hist = hist / (np.sum(hist) + 1e-6)
    
    # Compute edge features
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    edges = cv2.Canny(gray, 50, 150)
    edge_hist = cv2.calcHist([edges], [0], None, [16], [0, 256]).flatten()
    edge_hist = edge_hist / (np.sum(edge_hist) + 1e-6)
    
    # Compute texture features using LBP-like approach
    texture = cv2.resize(gray, (16, 16)).flatten().astype(np.float32) / 255.0
    
    # Combine features
    embedding = np.concatenate([hist, edge_hist, texture])
    
    # Normalize to unit vector
    embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
    
    return embedding


def _get_embedding_function():
    """Get embedding function - real or mock based on availability."""
    try:
        from facenet_pytorch import InceptionResnetV1
        import torch
        
        model = InceptionResnetV1(pretrained='vggface2').eval()
        
        def real_embedding(image: np.ndarray) -> np.ndarray:
            # Preprocess for FaceNet
            resized = cv2.resize(image, (160, 160))
            
            # BGR to RGB
            if len(resized.shape) == 3:
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            else:
                rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            
            # Normalize to [-1, 1]
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
            tensor = (tensor - 127.5) / 128.0
            
            with torch.no_grad():
                embedding = model(tensor)
            
            return embedding.numpy().flatten()
        
        return real_embedding
    except ImportError:
        return _mock_extract_embedding


_embedding_func = None


def extract_identity_embedding(image: np.ndarray) -> np.ndarray:
    """Extract identity embedding from an image.
    
    Args:
        image: Input image (H, W, 3) uint8
    
    Returns:
        Identity embedding vector (normalized)
    """
    global _embedding_func
    if _embedding_func is None:
        _embedding_func = _get_embedding_function()
    return _embedding_func(image)


def compare_embeddings(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Compute cosine similarity between two identity embeddings.
    
    Args:
        emb_a: First embedding vector
        emb_b: Second embedding vector
    
    Returns:
        Cosine similarity (-1 to 1, 1 = identical)
    """
    # Normalize if not already
    emb_a = emb_a / (np.linalg.norm(emb_a) + 1e-6)
    emb_b = emb_b / (np.linalg.norm(emb_b) + 1e-6)
    
    return float(np.dot(emb_a, emb_b))


def track_identity(frames: list[np.ndarray]) -> list[float]:
    """Track identity consistency across a sequence of frames.
    
    Args:
        frames: List of frames (H, W, 3) uint8
    
    Returns:
        List of cosine similarities relative to the first frame
    """
    if not frames:
        return []
    
    # Extract embedding for first frame
    ref_embedding = extract_identity_embedding(frames[0])
    
    similarities = []
    for frame in frames:
        embedding = extract_identity_embedding(frame)
        similarity = compare_embeddings(ref_embedding, embedding)
        similarities.append(similarity)
    
    return similarities


def detect_face_regions(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect face regions in an image.
    
    Uses OpenCV's Haar cascade for basic face detection.
    
    Args:
        image: Input image (H, W, 3) uint8
    
    Returns:
        List of (x, y, w, h) bounding boxes
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Load Haar cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    
    return [(x, y, w, h) for (x, y, w, h) in faces]


def extract_face_embedding(image: np.ndarray) -> np.ndarray | None:
    """Extract identity embedding from the largest face in an image.
    
    Args:
        image: Input image (H, W, 3) uint8
    
    Returns:
        Identity embedding vector, or None if no face detected
    """
    faces = detect_face_regions(image)
    
    if not faces:
        return None
    
    # Get largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    
    # Crop and extract embedding
    face_crop = image[y:y+h, x:x+w]
    return extract_identity_embedding(face_crop)