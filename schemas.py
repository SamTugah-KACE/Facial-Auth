from pydantic import BaseModel, field_validator, UUID4

class UserBase(BaseModel):
    username: str

    @field_validator('username', mode='before')
    def validate_non_empty_and_no_string(cls, v, info):
        if not v.strip() or 'string' in v.strip():
            raise ValueError(f'{info.field_name} cannot be empty or contain the word "string"')
        return v

class UserCreate(UserBase):
    pass

class UserUpdate(UserBase):
    pass

class UserInDBBase(UserBase):
    id: UUID4

    class Config:
        orm_mode = True

class User(UserInDBBase):
    pass



class UserAdminBase(BaseModel):
    username: str


class UserAdminCreate(UserAdminBase):
    h_password: str

class UserAdminUpdate(UserAdminBase):
    pass

class UserAdmin(UserAdminBase):
    id: UUID4
    

    class Config:
        orm_mode = True