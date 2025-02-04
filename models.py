from sqlalchemy import Column, Integer, String, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid
from sqlalchemy import Column, String,  DateTime
from database.base_class import APIBase



Base = declarative_base()



class User(APIBase):
    __tablename__ = 'users'
    username = Column(String, unique=True, index=True, nullable=False)
    embedding = Column(LargeBinary, nullable=False)
  

    def __repr__(self):
        return f"<User(username='{self.username}', id={self.id})>"

'''
The embedding column stores the facial embeddings as binary data.
The __repr__ method provides a readable string representation of the user.
'''


class UserAdmin(APIBase):
    __tablename__ = "user_admin"
    username = Column(String(20), unique=True, nullable=False)
    h_password = Column(String(255), unique=True, nullable=False)
