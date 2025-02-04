import secrets
import os

class Settings:
    PROJECT_NAME:str = "Biometric-Based Authentication"
    PROJECT_VERSION: str = "1.0.0"

    # CORS settings
    origins = ["http://localhost:3000"]


    """
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "password")
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "localhost")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", 5432)
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "auth_sys_db")
    INSTANCE_CONNECTION_NAME: str = os.getenv("INSTANCE_CONNECTION_NAME", None)
    UNIX_SOCKET: str = os.getenv("INSTANCE_UNIX_SOCKET", '/cloudsql/')
    PROJECT_ID: str = os.getenv("PROJECT_ID")
    BUCKET_NAME: str = os.getenv("BUCKET_NAME", "auth_sys_db")
    FLYER_PATH: str = os.getenv("FLYER_PATH")
    OUTLINE_PATH: str = os.getenv("OUTLINE_PATH")
    SHOW_DOCS: str = os.getenv("SHOW_DOCS")
    

    SQLALCHEMY_DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_SERVER}:{POSTGRES_PORT}/{POSTGRES_DB}"

    """    
    MAIL_USERNAME: str = 'dev.aiti.com.gh@gmail.com'
    MAIL_PASSWORD: str = 'uefuovgtfwyfgskv'
    MAIL_FROM: str = 'dev.aiti.com.gh@gmail.com'
    MAIL_PORT: int = 587
    MAIL_SERVER: str = 'smtp.gmail.com'
    MAIL_STARTTLS: bool = True
    MAIL_SSL_TLS: bool = False
    USE_CREDENTIALS: bool = True
    VALIDATE_CERTS: bool = True




    EMAIL_CODE_DURATION_IN_MINUTES: int = 15
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 45
    REFRESH_TOKEN_DURATION_IN_MINUTES: int =  60 * 24 * 7
    PASSWORD_RESET_TOKEN_DURATION_IN_MINUTES: int = 15
    ACCOUNT_VERIFICATION_TOKEN_DURATION_IN_MINUTES: int = 15

    POOL_SIZE: int = 20
    POOL_RECYCLE: int = 3600
    POOL_TIMEOUT: int = 15
    MAX_OVERFLOW: int = 2
    CONNECT_TIMEOUT: int = 60
    connect_args = {"connect_timeout":CONNECT_TIMEOUT}

    JWT_SECRET_KEY : str = secrets.token_urlsafe(32)
    REFRESH_TOKEN_SECRET_KEY : str = secrets.token_urlsafe(32)
    ALGORITHM: str = "HS256"


    # EMAIL_CODE_DURATION_IN_MINUTES: int = os.getenv("EMAIL_CODE_DURATION_IN_MINUTES")
    # ACCESS_TOKEN_EXPIRE_MINUTES: int = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")
    # REFRESH_TOKEN_DURATION_IN_MINUTES: int = os.getenv("REFRESH_TOKEN_DURATION_IN_MINUTES")
    # PASSWORD_RESET_TOKEN_DURATION_IN_MINUTES: int = os.getenv("PASSWORD_RESET_TOKEN_DURATION_IN_MINUTES")
    # ACCOUNT_VERIFICATION_TOKEN_DURATION_IN_MINUTES: int = os.getenv("ACCOUNT_VERIFICATION_TOKEN_DURATION_IN_MINUTES")

    # POOL_SIZE: int = os.getenv("POOL_SIZE")
    # POOL_RECYCLE: int = os.getenv("POOL_RECYCLE")
    # POOL_TIMEOUT: int = os.getenv("POOL_TIMEOUT")
    # MAX_OVERFLOW: int = os.getenv("MAX_OVERFLOW")
    # CONNECT_TIMEOUT: int = os.getenv("CONNECT_TIMEOUT")
    # connect_args = {"connect_timeout":os.getenv("CONNECT_TIMEOUT")}



    # JWT_SECRET_KEY : str = os.getenv("JWT_SECRET_KEY")
    # ALGORITHM: str = os.getenv("ALGORITHM")

    """
    class Config:


        env_file = './.env'
    """

settings = Settings()