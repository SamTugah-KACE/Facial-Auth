from fastapi import FastAPI, APIRouter, Depends, HTTPException, UploadFile, File, Form, Request
from sqlalchemy.orm import Session
from database.db_session import get_db
from models import User
import crud
from typing import Optional
from fastapi.responses import JSONResponse
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# app = FastAPI()
router = APIRouter(prefix="/users", tags=["User Management"])

# Create user route
@router.post("/create", response_model=dict)
def create_user(
    request: Request,
    username: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    try:
        image_bytes = file.file.read()
        user = crud.create_user(db=db, username=username, image_bytes=image_bytes, request=request)
        return {"message": "User created successfully", "username": user.username}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Authenticate user route
@router.post("/authenticate", response_model=dict)
def authenticate_user(
    request: Request,
    username: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    try:
        logger.info(f"Received authentication request for username: {username}")
        
        # Check file type
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a JPEG or PNG image.")
        
        # Read file and validate
        image_bytes = file.file.read()
        if not image_bytes or len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
        logger.info(f"File size: {len(image_bytes)} bytes")
        
        # Call the CRUD function for authentication
        response = crud.authenticate_user(db=db, username=username, image_bytes=image_bytes, request=request)
        logger.info("Authentication completed successfully.")
        
        return response
    except HTTPException as e:
        logger.error(f"HTTP Exception: {e.detail}")
        raise e
    except Exception as e:
        logger.exception("Unhandled exception during authentication")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# # Update user route
# @router.put("/update/{username}", response_model=dict)
# def update_user(
#     request: Request,
#     username: str,
#     new_username: Optional[str] = Form(None),
#     file: Optional[UploadFile] = File(None),
#     db: Session = Depends(get_db),
# ):
#     try:
#         image_bytes = file.file.read() if file else None
#         user = crud.update_user(
#             db=db, username=username, new_username=new_username, image_bytes=image_bytes, request=request
#         )
#         return {"message": "User updated successfully", "username": user.username}
#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# Update user route
@router.put("/update/{username}", response_model=dict)
def update_user(
    request: Request,
    username: str,
    new_username: Optional[str] = Form(None),
    update_face: bool = Form(False),  # Add bool flag to determine facial update
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
):
    try:
        if file is None:
            raise HTTPException(status_code=400, detail="File is required for authentication.")
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a JPEG or PNG image.")

        # Read image bytes
        image_bytes = file.file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Step 1: Authenticate the user
        auth_response = crud.authenticate_user(db=db, username=username, image_bytes=image_bytes, request=request)
        logger.info(f"Authentication response: {auth_response}")

        # Step 2: Update user details
        updated_user = crud.update_user(
            db=db,
            username=username,
            new_username=new_username,
            image_bytes=image_bytes if update_face else None,  # Only update embedding if `update_face` is True
            request=request,
        )

        return {
            "message": "User updated successfully",
            "username": updated_user.username,
            "updated_face": update_face,
        }
    except HTTPException as e:
        logger.error(f"HTTP Exception: {e.detail}")
        raise e
    except Exception as e:
        logger.exception("Unhandled exception during user update")
        raise HTTPException(status_code=500, detail="Internal server error")



# # Delete user route
# @router.delete("/delete/{username}", response_model=dict)
# def delete_user(username: str, db: Session = Depends(get_db)):
#     try:
#         response = crud.delete_user(db=db, username=username)
#         return response
#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# Delete user route with facial authentication
@router.delete("/delete/{username}", response_model=dict)
def delete_user(
    username: str,
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    try:
        logger.info(f"Received delete request for username: {username}")
        
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a JPEG or PNG image.")
        
        # Read image bytes
        image_bytes = file.file.read()
        if not image_bytes or len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
        # Authenticate user
        logger.info("Authenticating user before deletion...")
        # crud.authenticate_user(db=db, username=username, image_bytes=image_bytes, request=request)
        auth_response = crud.authenticate_user(db=db, username=username, image_bytes=image_bytes, request=request)

        # Handle authentication response (implicit from authenticate_user raising HTTPException on failure)
        logger.info(f"\nDelete Authentication successful for user: {auth_response['username']}.")
        
        # If authentication passes, proceed to delete
        response = crud.delete_user(db=db, username=username)
        logger.info(f"User {username} deleted successfully.")
        return response
    except HTTPException as e:
        logger.error(f"HTTP Exception during deletion: {e.detail}")
        raise e
    except Exception as e:
        logger.exception("Unhandled exception during user deletion")
        return {"detail": "Internal server error"}




# # Add router to app
# app.include_router(router)

# # Testing mechanisms (sample client request functions)
# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=8000)

























# from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
# from sqlalchemy.orm import Session
# import crud, schemas
# from database.db_session import get_db
# from typing import Optional
# import logging

# router = APIRouter(tags=["Bio-Facial Authentication System"])

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# @router.post("/users/", response_model=schemas.User)
# def create_user(username: Optional[str] = Form(None), file: Optional[UploadFile] = File(None), db: Session = Depends(get_db)):

#     logger.info(f"Received user: {username}")
#     logging.info(f"Received file: {file.filename if file else 'No file'}")
#     image_bytes = file.file.read() if file else None
#     print("image_bytes: ", image_bytes)

#     try:
#         db_user = crud.create_user(db=db, username=username, image_bytes=image_bytes)
#         return db_user
#     except HTTPException as e:
#         logging.error(f"HTTPException: {e.detail}")
#         raise e
#     except Exception as e:
#         logging.error(f"Unexpected error: {e}")
#         raise HTTPException(status_code=500, detail="An unexpected error occurred")

# @router.post("/users/authenticate/")
# def authenticate_user(username: Optional[str] = Form(None), file: Optional[UploadFile] = File(None), db: Session = Depends(get_db)):
    
#     logger.info(f"Received user: {username}")
#     logging.info(f"Received file: {file.filename if file else 'No file'}")
#     image_bytes = file.file.read() if file else None
#     #print("image_bytes: ", image_bytes)
    
#     #image_bytes = file.file.read()
#     face_embeddings = crud.extract_face_embeddings(image_bytes)
#     print("\nface_embeddings in login: ", face_embeddings)
#     if face_embeddings is None or len(face_embeddings) == 0:
#         raise HTTPException(status_code=400, detail="No face detected in the image")
    
#     captured_embedding = face_embeddings[0]
#     #print("\ni.e.captured_embedding: ", captured_embedding)
#     response = crud.authenticate(db=db, username=username, captured_embedding=captured_embedding)
#     return response

# @router.put("/users/{username}", response_model=schemas.User)
# def update_user(username: str, user: schemas.UserUpdate, file: Optional[UploadFile] = File(None), db: Session = Depends(get_db)):
#     image_bytes = file.file.read() if file else None
#     db_user = crud.update_user(db=db, username=username, user=user, image_bytes=image_bytes)
#     return db_user

# @router.delete("/users/rm/{username}", response_model=schemas.User)
# def delete_user(username: str, db: Session = Depends(get_db)):
#     db_user = crud.delete_user(db=db, username=username)
#     return db_user

# # Evaluation metrics endpoint
# @router.post("/evaluate/")
# def evaluate_model(y_true: list, y_pred: list):
#     return crud.evaluate_model(y_true, y_pred)
