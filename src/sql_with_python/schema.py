# -*- coding: utf-8 -*-

""" table schema
"""

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Integer


Base = declarative_base()


class User(Base):

    __tablename__ = "user"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(20))
    fullname = Column(String(20))
    nickname = Column(String(20))

    def __repr__(self):
        return f"<User(name={self.name}, fullname={self.fullname}, nickname={self.nickname})>"

    def __str__(self):
        return self.__repr__()
