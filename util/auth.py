from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta
import secrets
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import and_
from sqlalchemy.orm import Session
import string
from typing import Optional
import models
import crud
import schemas
from database.db_session import get_db


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

SECRET_KEY = secrets.token_urlsafe(32)  # Generate a random secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
COOKIE_NAME="Authorization"


#pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")


route = APIRouter(
    prefix="/auth",
    tags=["Protection"],
    responses={404: {"description": "Not found"}},
)

# Function to generate JWT token
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def hash_password(password: str) -> str:
    return pwd_context.hash(password, salt="password")

def authenticate_admin(db: Session, username: str, password: str):
    
    try:
        hashed = hash_password(password)
       
        user = db.query(models.UserAdmin).filter(
            and_(
                models.UserAdmin.username == username, 
                models.UserAdmin.h_password == hashed
                )
            ).first()
        print("user in authenticate_admin: ",user.username)
        if user is None:
          
            return None  # User not found
        else:
            print("user.h_password: ", user.h_password)
            print("verify password: ", verify_password(hashed, user.h_password))
            if not verify_password(password, user.h_password):
                return None  # Incorrect password
        return user
    except Exception as e:
        print("Error during authentication:", e)
        return None







# Function to validate JWT token
async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("username")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = crud.get_admin(db, username)
    
    if not user:
        raise credentials_exception

    return {"username": user.username,  "token_data": payload}

# Function to log in and generate access token
@route.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_admin(db, form_data.username, form_data.password)
    print("user in login_for_access: ", user)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or Password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"username": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

    
#def get_current_user_from_cookies(request:Request) -> User:
#    token = request.cookies.get(COOKIE_NAME)
#    if token:
#        user = Hasher.verify_refresh_token(token)
#    return user