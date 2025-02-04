# from io import BytesIO
# from typing import Optional
# import logging
# from fastapi import Form, HTTPException, Request
# import numpy as np
# from sqlalchemy.exc import IntegrityError
# from sqlalchemy.orm import Session
# from PIL import Image, ImageEnhance, ImageOps
# from facenet_pytorch import MTCNN, InceptionResnetV1
# from torchvision import transforms
# import torch
# import os
# from sklearn.metrics.pairwise import cosine_similarity
# from datetime import datetime
# from models import User
# import time
# import psutil
# import platform
# import json
# import statistics

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Initialize face detection and recognition models
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# mtcnn = MTCNN(keep_all=False, device=device)  # Single face detection
# arcface = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# SIMILARITY_THRESHOLD = 0.8  # Adjust as needed
# SOFT_SIMILARITY_THRESHOLD = 0.7  # For handling cases with facial changes
# AUTHORIZED_EMBEDDINGS_FILE = "./dataset/authorized_embeddings.npy"
# TARGET_SIZE = (300, 300)

# # Ensure dataset path exists
# os.makedirs(os.path.dirname(AUTHORIZED_EMBEDDINGS_FILE), exist_ok=True)
# if not os.path.exists(AUTHORIZED_EMBEDDINGS_FILE):
#     np.save(AUTHORIZED_EMBEDDINGS_FILE, np.array([]))

# # Performance tracking
# performance_data = []  # Store timing and accuracy data for analysis

# def log_performance_metrics(name: str, time_taken: float, similarity_score: Optional[float] = None):
#     global performance_data
#     entry = {
#         "operation": name,
#         "time_taken": time_taken,
#         "similarity_score": float(similarity_score) if similarity_score is not None else None,  # Convert here,
#         "timestamp": datetime.utcnow().isoformat()
#     }
#     performance_data.append(entry)
#     logger.info(f"Performance Metrics Logged: {json.dumps(entry)}")

# # Performance measurement decorator
# def measure_performance(func):
#     def wrapper(*args, **kwargs):
#         start_time = time.time()
#         process = psutil.Process()
#         cpu_start = process.cpu_percent(interval=None)
#         gpu_utilization = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0

#         result = func(*args, **kwargs)

#         elapsed_time = time.time() - start_time
#         cpu_end = process.cpu_percent(interval=None)
#         logger.info(
#             f"{func.__name__} - Time: {elapsed_time:.4f}s | CPU: {cpu_end - cpu_start:.2f}% | GPU: {gpu_utilization} bytes"
#         )
#         log_performance_metrics(func.__name__, elapsed_time)
#         return result
#     return wrapper

# # Track environmental factors
# def track_environment(request: Request, image: Image.Image):
#     user_agent = request.headers.get("user-agent", "unknown")
#     device_type = "smartphone" if "mobi" in user_agent.lower() else "desktop"
#     resolution = image.size
#     brightness = np.mean(np.asarray(image.convert('L')))

#     logger.info(f"Device Type: {device_type}")
#     logger.info(f"Image Resolution: {resolution}")
#     logger.info(f"Average Brightness: {brightness:.2f}")

#     return {
#         "device_type": device_type,
#         "resolution": resolution,
#         "brightness": brightness
#     }

# # Image preprocessing and enhancement
# def enhance_image(image: Image.Image) -> Image.Image:
#     image = ImageOps.autocontrast(image)
#     image = ImageEnhance.Contrast(image).enhance(2)
#     image = ImageEnhance.Sharpness(image).enhance(1.5)
#     return image

# # Utility: Resize with Padding
# def resize_with_padding(image: Image.Image, target_size: tuple) -> Image.Image:
#     old_size = image.size
#     ratio = min(target_size[0] / old_size[0], target_size[1] / old_size[1])
#     new_size = (int(old_size[0] * ratio), int(old_size[1] * ratio))

#     # Use LANCZOS resampling for resizing
#     image = image.resize(new_size, Image.Resampling.LANCZOS)
#     new_image = Image.new("RGB", target_size, (0, 0, 0))
#     paste_position = ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2)
#     new_image.paste(image, paste_position)
#     return new_image

# # Extract embeddings from image
# @measure_performance
# def extract_face_embeddings(image_bytes: bytes) -> Optional[np.ndarray]:
#     try:
#         image = Image.open(BytesIO(image_bytes)).convert('RGB')
#         logger.info(f"Image loaded successfully. Size: {image.size}")

#         image = enhance_image(image)
#         logger.info("Image enhancement completed.")

#         image = resize_with_padding(image, TARGET_SIZE)
#         logger.info(f"Image resized to: {TARGET_SIZE}")
#         # image = image.resize(TARGET_SIZE)

#          # Detect face using MTCNN
#         boxes, _ = mtcnn.detect(image)
#         if boxes is None:
#             logger.warning("No face detected in the image.")
#             return None

#         # Crop the first detected face
#         box = boxes[0]  # Focus on the first face detected
#         face = image.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))


#          # Normalize cropped face
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Resize(TARGET_SIZE),
#             transforms.Normalize([0.5], [0.5])
#         ])

#         face_tensor = transform(image).unsqueeze(0).to(device)
#         logger.info("Image tensor created successfully.")
#         # face_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
#         with torch.no_grad():
#             # embedding = arcface(face_tensor).cpu().numpy()
#             embedding = arcface(face_tensor).cpu().numpy().flatten()
        
#         logger.info("Face embedding extracted successfully.")
#         return embedding
#     except Exception as e:
#         logger.error(f"Error in extracting face embeddings: {e}")
#         return None

# # Compare embeddings
# @measure_performance
# def compare_embeddings(stored_embedding: np.ndarray, captured_embedding: np.ndarray) -> bool:
#     similarity = cosine_similarity([stored_embedding], [captured_embedding])[0][0]
#     similarity = round(float(similarity), 1)  # Convert to float and round to 2 decimals
#     logger.info(f"Similarity Score: {similarity}")
#     # logger.info(f"Similarity Score: {float(similarity)}")  # Convert to float
#     log_performance_metrics("compare_embeddings", 0, similarity_score=similarity)
#     if similarity >= SIMILARITY_THRESHOLD:
#         logger.info("Strong match detected.")
#         return True
#     elif similarity >= SOFT_SIMILARITY_THRESHOLD:
#         logger.warning("Soft match detected. Additional checks may be needed.")
#         return True
#     logger.warning("No match found.")
#     return False

# # Load and save embeddings
# def load_authorized_embeddings() -> np.ndarray:
#     # return np.load(AUTHORIZED_EMBEDDINGS_FILE, allow_pickle=True)
#     try:
#         embeddings = np.load(AUTHORIZED_EMBEDDINGS_FILE, allow_pickle=True)
#         if not isinstance(embeddings, np.ndarray):
#             embeddings = np.array([])
#         logger.info(f"Loaded embeddings with shape: {embeddings.shape}")
#         return embeddings
#     except Exception as e:
#         logger.error(f"Error loading embeddings: {e}")
#         return np.array([])
    

# def save_authorized_embeddings(embeddings: np.ndarray):
#     np.save(AUTHORIZED_EMBEDDINGS_FILE, embeddings)

# # CRUD operations
# @measure_performance
# def create_user(db: Session, username: str, image_bytes: bytes, request: Request) -> User:
#     if not username.strip():
#         raise HTTPException(status_code=400, detail="Username cannot be empty")

#     existing_user = db.query(User).filter(User.username == username).first()
#     if existing_user:
#         raise HTTPException(status_code=400, detail="Username already exists")

#     embedding = extract_face_embeddings(image_bytes)
#     print("embedding:: ", embedding)
#     if embedding is None:
#         raise HTTPException(status_code=400, detail="Face not detected or invalid image")

#     # Ensure embedding shape is (512,)
#     # embedding = embedding.flatten()

#     # Track environment
#     image = Image.open(BytesIO(image_bytes)).convert('RGB')
#     env_factors = track_environment(request, image)
#     logger.info(f"Environmental Factors: {json.dumps(env_factors)}")

#     # Load existing embeddings
#     global_embeddings = load_authorized_embeddings()

#     # Verify embeddings shape consistency
#     if len(global_embeddings) > 0:
#         global_embeddings = np.array(global_embeddings)  # Ensure array
#         if global_embeddings.ndim == 1:  # Handle single embedding case
#             global_embeddings = global_embeddings.reshape(1, -1)

#     # Compare embeddings to detect duplicates
#     if any(compare_embeddings(embedding, stored) for stored in global_embeddings):
#         raise HTTPException(status_code=400, detail="A similar face is already registered")

#     # Save the new user
#     user = User(username=username, embedding=embedding.tobytes(), created_at=datetime.utcnow())
#     db.add(user)
#     db.commit()
#     db.refresh(user)

#     # Update global embeddings
#     global_embeddings = np.vstack([global_embeddings, embedding]) if len(global_embeddings) > 0 else embedding[None, :]
#     save_authorized_embeddings(global_embeddings)

#     return user


# @measure_performance
# def authenticate_user(db: Session, username: str, image_bytes: bytes, request: Request) -> dict:
#     user = db.query(User).filter(User.username == username).first()
#     if not user:
#         logger.warning("Authentication failed: User not found.")
#         raise HTTPException(status_code=404, detail="User not found")

#     stored_embedding = np.frombuffer(user.embedding, dtype=np.float32)
#     captured_embedding = extract_face_embeddings(image_bytes)
#     if captured_embedding is None:
#         logger.warning("Authentication failed: Face not detected or invalid image.")
#         raise HTTPException(status_code=400, detail="Face not detected or invalid image")

#     # Track environment
#     image = Image.open(BytesIO(image_bytes)).convert('RGB')
#     env_factors = track_environment(request, image)
#     logger.info(f"Environmental Factors: {json.dumps(env_factors)}")

#     if compare_embeddings(stored_embedding, captured_embedding):
#         logger.info(f"Authentication successful for user: {username}.")
#         return {"message": "Authentication successful", "username": username, "env_factors": env_factors}
#     else:
#         logger.warning("Authentication failed: Embeddings do not match.")
#         raise HTTPException(status_code=401, detail="User authentication failed due to mismatch")

# # @measure_performance
# # def update_user(db: Session, username: str, new_username: Optional[str], image_bytes: Optional[bytes], request: Request) -> User:
# #     user = db.query(User).filter(User.username == username).first()
# #     if not user:
# #         raise HTTPException(status_code=404, detail="User not found")

# #     if new_username:
# #         if db.query(User).filter(User.username == new_username).first():
# #             raise HTTPException(status_code=400, detail="New username already exists")
# #         user.username = new_username

# #     if image_bytes:
# #         embedding = extract_face_embeddings(image_bytes)
# #         if embedding is None:
# #             raise HTTPException(status_code=400, detail="Face not detected or invalid image")
# #         user.embedding = embedding.tobytes()

# #         # Update global embeddings
# #         global_embeddings = load_authorized_embeddings()
# #         stored_embedding = np.frombuffer(user.embedding, dtype=np.float32)
# #         updated_embeddings = [emb for emb in global_embeddings if not np.array_equal(emb, stored_embedding)]
# #         updated_embeddings.append(embedding)
# #         save_authorized_embeddings(np.array(updated_embeddings))

# #     user.updated_at = datetime.utcnow()
# #     db.commit()
# #     db.refresh(user)
# #     return user


# @measure_performance
# def update_user(
#     db: Session,
#     username: str,
#     new_username: Optional[str],
#     image_bytes: Optional[bytes],
#     request: Request,
# ) -> User:
#     # Fetch the user
#     user = db.query(User).filter(User.username == username).first()
#     if not user:
#         raise HTTPException(status_code=404, detail="User not found")

#     # Validate and update username only if it's different from the current one
#     if new_username and new_username != username:
#         if db.query(User).filter(User.username == new_username).first():
#             raise HTTPException(status_code=400, detail="New username already exists")
#         user.username = new_username

#     # Update facial embedding if `image_bytes` is provided
#     if image_bytes:
#         embedding = extract_face_embeddings(image_bytes)
#         if embedding is None:
#             raise HTTPException(status_code=400, detail="Face not detected or invalid image")
#         user.embedding = embedding.tobytes()

#         # Update global embeddings
#         global_embeddings = load_authorized_embeddings()
#         stored_embedding = np.frombuffer(user.embedding, dtype=np.float32)
#         updated_embeddings = [
#             emb for emb in global_embeddings if not np.array_equal(emb, stored_embedding)
#         ]
#         updated_embeddings.append(embedding)
#         save_authorized_embeddings(np.array(updated_embeddings))

#     # Commit changes
#     user.updated_at = datetime.utcnow()
#     db.commit()
#     db.refresh(user)
#     return user





# @measure_performance
# def delete_user(db: Session, username: str):
#     user = db.query(User).filter(User.username == username).first()
#     if not user:
#         raise HTTPException(status_code=404, detail="User not found")

#     # Remove embedding from global storage
#     global_embeddings = load_authorized_embeddings()
#     stored_embedding = np.frombuffer(user.embedding, dtype=np.float32)
#     updated_embeddings = [emb for emb in global_embeddings if not np.array_equal(emb, stored_embedding)]
#     save_authorized_embeddings(np.array(updated_embeddings))

#     db.delete(user)
#     db.commit()
#     return {"message": f"User {username} deleted successfully"}



# @measure_performance
# def update_user(db: Session, username: str, new_username: Optional[str], image_bytes: Optional[bytes]) -> User:
#     user = db.query(User).filter(User.username == username).first()
#     if not user:
#         raise HTTPException(status_code=404, detail="User not found.")
#     if new_username and db.query(User).filter(User.username == new_username).first():
#         raise HTTPException(status_code=400, detail="New username already exists.")
#     user.username = new_username or user.username
#     if image_bytes:
#         embedding = extract_embeddings(image_bytes)
#         if embedding is None:
#             raise HTTPException(status_code=400, detail="No valid face detected.")
#         user.embedding = embedding.tobytes()
#         update_global_embeddings(user.embedding, embedding)
#     db.commit()
#     db.refresh(user)
#     return user

# @measure_performance
# def delete_user(db: Session, username: str):
#     user = db.query(User).filter(User.username == username).first()
#     if not user:
#         raise HTTPException(status_code=404, detail="User not found.")
#     remove_global_embedding(user.embedding)
#     db.delete(user)
#     db.commit()
#     return {"message": f"User {username} deleted successfully"}




import os
import csv
import time
import psutil
import json
import logging
import numpy as np
from datetime import datetime
from io import BytesIO
from typing import Optional
from PIL import Image, ImageEnhance, ImageOps
from fastapi import HTTPException, Request
from sqlalchemy.orm import Session
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from retinaface import RetinaFace
from models import User
import torch

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and directories
AUTHORIZED_EMBEDDINGS_FILE = "./dataset/authorized_embeddings.npy"
REPORTS_DIR = "./reports"
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(AUTHORIZED_EMBEDDINGS_FILE), exist_ok=True)
if not os.path.exists(AUTHORIZED_EMBEDDINGS_FILE):
    np.save(AUTHORIZED_EMBEDDINGS_FILE, np.array([]))

# Models and thresholds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
arcface = InceptionResnetV1(pretrained="vggface2").eval().to(device)
mtcnn = MTCNN(keep_all=False, device=device)
SIMILARITY_THRESHOLD = 0.85
EUCLIDEAN_THRESHOLD = 0.6
TARGET_SIZE = (300, 300)

# Performance tracking
performance_data = []  # Store timing and accuracy data for analysis

def log_performance_metrics(name: str, time_taken: float, similarity_score: Optional[float] = None):
    global performance_data
    entry = {
        "operation": name,
        "time_taken": time_taken,
        "similarity_score": similarity_score if similarity_score is not None else None,
        "timestamp": datetime.utcnow().isoformat(),
    }
    performance_data.append(entry)
    logger.info(f"Performance Metrics Logged: {json.dumps(entry)}")

def measure_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        process = psutil.Process()
        cpu_start = process.cpu_percent(interval=None)
        gpu_utilization = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0

        result = func(*args, **kwargs)

        elapsed_time = time.time() - start_time
        cpu_end = process.cpu_percent(interval=None)
        logger.info(
            f"{func.__name__} - Time: {elapsed_time:.4f}s | CPU: {cpu_end - cpu_start:.2f}% | GPU: {gpu_utilization} bytes"
        )
        log_performance_metrics(func.__name__, elapsed_time)
        return result
    return wrapper

# Image preprocessing
def preprocess_image(image: Image.Image) -> Image.Image:
    image = resize_with_padding(image, TARGET_SIZE)
    image = ImageOps.autocontrast(image)
    image = ImageEnhance.Contrast(image).enhance(2)
    image = ImageEnhance.Sharpness(image).enhance(1.5)
    return image

def resize_with_padding(image: Image.Image, target_size: tuple) -> Image.Image:
    old_size = image.size
    ratio = min(target_size[0] / old_size[0], target_size[1] / old_size[1])
    new_size = (int(old_size[0] * ratio), int(old_size[1] * ratio))
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    new_image = Image.new("RGB", target_size, (0, 0, 0))
    new_image.paste(image, ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2))
    return new_image

# # Face detection
# def detect_face(image: Image.Image) -> Optional[Image.Image]:
#     try:
#         faces = RetinaFace.extract_faces(np.array(image), align=True)
#         if faces:
#             return Image.fromarray(faces[0])
#         boxes, _ = mtcnn.detect(image)
#         if boxes is None:
#             return None
#         box = boxes[0]
#         return image.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
#     except Exception as e:
#         logger.error(f"Error in face detection: {e}")
#         return None


# Face detection
def detect_face(image: Image.Image) -> Optional[Image.Image]:
    """
    Detects a face in the given image using RetinaFace and MTCNN as a fallback.

    Args:
        image (Image.Image): The input image.

    Returns:
        Optional[Image.Image]: Cropped face image if a face is detected; otherwise, None.
    """
    try:
        # Convert PIL Image to NumPy array
        image_array = np.array(image)

        # Attempt face detection using RetinaFace
        detections = RetinaFace.detect(image_array)
        if detections:
            # Extract bounding box of the first face
            for key in detections:
                face_area = detections[key]["facial_area"]
                x1, y1, x2, y2 = face_area
                return image.crop((x1, y1, x2, y2))

        # Fallback to MTCNN if RetinaFace fails
        logger.warning("RetinaFace failed, falling back to MTCNN.")
        boxes, _ = mtcnn.detect(image)
        if boxes is not None:
            # Use the first detected face
            box = boxes[0]
            return image.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))

        logger.warning("No face detected.")
        return None
    except Exception as e:
        logger.error(f"Error in face detection: {e}")
        return None


# Embedding extraction
@measure_performance
def extract_embeddings(image_bytes: bytes) -> Optional[np.ndarray]:
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = preprocess_image(image)
        face = detect_face(image)
        if face is None:
            return None
        face_tensor = transforms.ToTensor()(face).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = arcface(face_tensor).cpu().numpy().flatten()
        return embedding
    except Exception as e:
        logger.error(f"Error extracting embeddings: {e}")
        return None

# Embedding comparison
def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> dict:
    cosine_sim = cosine_similarity([embedding1], [embedding2])[0][0]
    euclidean_dist = np.linalg.norm(embedding1 - embedding2)
    return {"cosine_similarity": cosine_sim, "euclidean_distance": euclidean_dist}

def compare_embeddings(stored: np.ndarray, captured: np.ndarray) -> bool:
    metrics = compute_similarity(stored, captured)
    return (
        metrics["cosine_similarity"] >= SIMILARITY_THRESHOLD and
        metrics["euclidean_distance"] <= EUCLIDEAN_THRESHOLD
    ), metrics

# CRUD Operations
@measure_performance
def create_user(db: Session, username: str, image_bytes: bytes, request: Request) -> User:
    if not username.strip():
        raise HTTPException(status_code=400, detail="Username cannot be empty.")
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="Username already exists.")
    embedding = extract_embeddings(image_bytes)
    if embedding is None:
        raise HTTPException(status_code=400, detail="No valid face detected.")
    user = User(username=username, embedding=embedding.tobytes(), created_at=datetime.utcnow())
    db.add(user)
    db.commit()
    db.refresh(user)
    save_authorized_embeddings(embedding)
    return user



# Embedding management
def save_authorized_embeddings(embedding: np.ndarray):
    global_embeddings = load_authorized_embeddings()
    if global_embeddings.size == 0:
        global_embeddings = np.array([embedding])
    else:
        global_embeddings = np.vstack([global_embeddings, embedding])
    np.save(AUTHORIZED_EMBEDDINGS_FILE, global_embeddings)

def load_authorized_embeddings() -> np.ndarray:
    try:
        return np.load(AUTHORIZED_EMBEDDINGS_FILE, allow_pickle=True)
    except Exception:
        return np.array([])

def update_global_embeddings(old_embedding: bytes, new_embedding: np.ndarray):
    embeddings = load_authorized_embeddings()
    stored_embedding = np.frombuffer(old_embedding, dtype=np.float32)
    updated_embeddings = [
        emb for emb in embeddings if not np.array_equal(emb, stored_embedding)
    ]
    updated_embeddings.append(new_embedding)
    np.save(AUTHORIZED_EMBEDDINGS_FILE, np.array(updated_embeddings))

def remove_global_embedding(old_embedding: bytes):
    embeddings = load_authorized_embeddings()
    stored_embedding = np.frombuffer(old_embedding, dtype=np.float32)
    updated_embeddings = [emb for emb in embeddings if not np.array_equal(emb, stored_embedding)]
    np.save(AUTHORIZED_EMBEDDINGS_FILE, np.array(updated_embeddings))

from fastapi import Request
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)

def track_environment(request: Request, image: Image.Image) -> dict:
    """
    Tracks environmental factors like device type, image resolution, and brightness for a given request.

    Args:
        request (Request): FastAPI Request object containing headers and metadata.
        image (Image.Image): The input image to analyze environmental factors.

    Returns:
        dict: Environmental factors including device type, resolution, and brightness.
    """
    try:
        # Determine device type from User-Agent header
        user_agent = request.headers.get("user-agent", "unknown").lower()
        device_type = "smartphone" if "mobi" in user_agent else "desktop"

        # Get image resolution
        resolution = image.size

        # Calculate average brightness
        brightness = np.mean(np.asarray(image.convert('L')))

        # Log the factors
        logger.info(f"Device Type: {device_type}")
        logger.info(f"Image Resolution: {resolution}")
        logger.info(f"Average Brightness: {brightness:.2f}")

        return {
            "device_type": device_type,
            "resolution": resolution,
            "brightness": brightness
        }
    except Exception as e:
        logger.error(f"Error tracking environment: {e}")
        return {
            "device_type": "unknown",
            "resolution": (0, 0),
            "brightness": 0.0
        }



@measure_performance
def authenticate_user(db: Session, username: str, image_bytes: bytes, request: Request) -> dict:
    """
    Authenticate the user by comparing the captured face embedding to the stored embedding for the given username.
    """
    user = db.query(User).filter(User.username == username).first()
    if not user:
        logger.warning("Authentication failed: User not found.")
        raise HTTPException(status_code=404, detail="User not found.")

    # Extract embeddings from the provided image
    stored_embedding = np.frombuffer(user.embedding, dtype=np.float32)
    captured_embedding = extract_embeddings(image_bytes)
    if captured_embedding is None:
        logger.warning("Authentication failed: Face not detected or invalid image.")
        raise HTTPException(status_code=400, detail="Face not detected or invalid image.")

    # Log environmental factors
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    env_factors = track_environment(request, image)
    logger.info(f"Environmental Factors: {json.dumps(env_factors)}")

    # Compare embeddings for the specific user
    is_match, metrics = compare_embeddings(stored_embedding, captured_embedding)
    if not is_match:
        logger.warning("Authentication failed: Embeddings do not match.")
        raise HTTPException(status_code=401, detail="User authentication failed due to mismatch.")

    # Verify no match exists with other users
    global_embeddings = load_authorized_embeddings()
    for embedding in global_embeddings:
        if np.array_equal(embedding, stored_embedding):
            continue  # Skip the current user's embedding
        if compute_similarity(embedding, captured_embedding)["cosine_similarity"] >= SIMILARITY_THRESHOLD:
            logger.warning("Authentication failed: Face matches another account.")
            raise HTTPException(status_code=401, detail="Authentication failed. Face matches another account.")

    logger.info(f"Authentication successful for user: {username}.")
    return {"message": "Authentication successful", "username": username, "env_factors": env_factors, "metrics": metrics}


@measure_performance
def update_user(db: Session, username: str, new_username: Optional[str], image_bytes: Optional[bytes], request: Request) -> User:
    """
    Update user details, ensuring strict face authentication and uniqueness of new username.
    """
    # Authenticate the user
    authenticate_user(db, username, image_bytes, request)

    # Fetch the user and validate
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    if new_username and new_username != username:
        if db.query(User).filter(User.username == new_username).first():
            raise HTTPException(status_code=400, detail="New username already exists.")
        user.username = new_username

    # Update facial embedding if new image is provided
    if image_bytes:
        new_embedding = extract_embeddings(image_bytes)
        if new_embedding is None:
            raise HTTPException(status_code=400, detail="No valid face detected.")
        user.embedding = new_embedding.tobytes()

        # Update global embeddings
        update_global_embeddings(user.embedding, new_embedding)

    # Commit changes
    user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(user)
    return user


@measure_performance
def delete_user(db: Session, username: str, image_bytes: bytes, request: Request) -> dict:
    """
    Delete a user only after strict face authentication to ensure no unauthorized access.
    """
    # Authenticate the user
    authenticate_user(db, username, image_bytes, request)

    # Fetch the user
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    # Remove embedding from global storage
    remove_global_embedding(user.embedding)

    # Delete the user from the database
    db.delete(user)
    db.commit()
    return {"message": f"User {username} deleted successfully"}













