# -*- coding: utf-8 -*-

"""using sql alchemy
"""

from __future__ import absolute_import
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


from schema import User, Base


# create engine MySQLdb没有支持python3的版本，如果使用python3.x版本时，需要安装额外的库pymysql
engine = create_engine("mysql+pymysql://lsl:123456@localhost:3306/target",
                       encoding='latin1', echo=True)
Base.metadata.create_all(engine)

Session = sessionmaker()
Session.configure(bind=engine)
session = Session()


# if __name__ == '__main__':

ed_user = User(name="ed", fullname="ED Json", nickname="ed_nickname")

session.add(ed_user)
session.commit()


