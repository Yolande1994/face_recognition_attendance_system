from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime, Float
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from typing import Optional, List
from core.config import settings

if settings.DB_TYPE == "sqlite":
    DATABASE_URL = f"sqlite:///{settings.SQLITE_PATH}"
    connect_args = {"check_same_thread": False}
else:
    DATABASE_URL = (
        f"mysql+pymysql://{settings.MYSQL_USER}:{settings.MYSQL_PASSWORD}"
        f"@{settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DB}?charset=utf8mb4"
    )
    connect_args = {}

engine = create_engine(DATABASE_URL, echo=False, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db():
    Base.metadata.create_all(bind=engine)


class UserFace(Base):
    __tablename__ = "user_face"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(64), unique=True, index=True, nullable=False)
    user_name = Column(String(64), nullable=False)
    face_embedding = Column(Text, nullable=False)
    create_time = Column(DateTime, default=datetime.now)
    update_time = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    @staticmethod
    def embedding_to_str(embedding: list) -> str:
        return ",".join(map(str, embedding))

    @staticmethod
    def str_to_embedding(embedding_str: str) -> list:
        return list(map(float, embedding_str.split(",")))


class CheckinRecord(Base):
    __tablename__ = "checkin_record"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(64), index=True, nullable=False)
    user_name = Column(String(64), nullable=False)
    checkin_time = Column(DateTime, default=datetime.now, index=True)
    similarity = Column(Float, nullable=False)
    checkin_type = Column(String(20), default="上班")


class FaceDB:
    def __init__(self):
        self.db = SessionLocal()

    def close(self):
        self.db.close()

    def add_user_face(self, user_id: str, user_name: str, embedding: list) -> bool:
        try:
            embedding_str = UserFace.embedding_to_str(embedding)
            user = UserFace(
                user_id=user_id,
                user_name=user_name,
                face_embedding=embedding_str
            )
            self.db.add(user)
            self.db.commit()
            return True
        except Exception:
            self.db.rollback()
            return False

    def get_user_face(self, user_id: str) -> Optional[dict]:
        user = self.db.query(UserFace).filter(UserFace.user_id == user_id).first()
        if not user:
            return None
        return {
            "user_id": user.user_id,
            "user_name": user.user_name,
            "face_embedding": UserFace.str_to_embedding(user.face_embedding)
        }

    def user_exists(self, user_id: str) -> bool:
        return self.db.query(UserFace).filter(UserFace.user_id == user_id).first() is not None

    def get_all_user_faces(self) -> List[dict]:
        users = self.db.query(UserFace).all()
        return [
            {
                "user_id": u.user_id,
                "user_name": u.user_name,
                "face_embedding": UserFace.str_to_embedding(u.face_embedding)
            }
            for u in users
        ]

    def add_checkin_record(self, user_id: str, user_name: str, similarity: float,
                           checkin_type: str = "上班") -> bool:
        try:
            record = CheckinRecord(
                user_id=user_id,
                user_name=user_name,
                similarity=similarity,
                checkin_type=checkin_type
            )
            self.db.add(record)
            self.db.commit()
            return True
        except Exception:
            self.db.rollback()
            return False

    def get_checkin_records(self, limit: int = 20) -> List[dict]:
        records = self.db.query(CheckinRecord).order_by(
            CheckinRecord.checkin_time.desc()
        ).limit(limit).all()
        return [
            {
                "user_id": r.user_id,
                "user_name": r.user_name,
                "checkin_time": r.checkin_time.strftime("%Y-%m-%d %H:%M:%S"),
                "similarity": round(r.similarity, 4),
                "checkin_type": r.checkin_type
            }
            for r in records
        ]

    def get_all_users(self) -> List[dict]:
        users = self.db.query(UserFace).order_by(UserFace.create_time.desc()).all()
        return [
            {
                "user_id": u.user_id,
                "user_name": u.user_name,
                "create_time": u.create_time.strftime("%Y-%m-%d %H:%M:%S")
            }
            for u in users
        ]

    def get_user_checkin_records(self, user_id: str) -> List[dict]:
        records = self.db.query(CheckinRecord).filter(
            CheckinRecord.user_id == user_id
        ).order_by(CheckinRecord.checkin_time.desc()).all()
        return [
            {
                "id": r.id,
                "user_id": r.user_id,
                "user_name": r.user_name,
                "checkin_time": r.checkin_time.strftime("%Y-%m-%d %H:%M:%S"),
                "similarity": round(r.similarity, 4),
                "checkin_type": r.checkin_type
            }
            for r in records
        ]

    def delete_user(self, user_id: str) -> bool:
        try:
            self.db.query(CheckinRecord).filter(CheckinRecord.user_id == user_id).delete()
            user = self.db.query(UserFace).filter(UserFace.user_id == user_id).first()
            if user:
                self.db.delete(user)
                self.db.commit()
                return True
            self.db.commit()
            return False
        except Exception:
            self.db.rollback()
            return False


if __name__ == "__main__":
    init_db()