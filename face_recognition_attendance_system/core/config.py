from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # 服务配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # 人脸识别配置
    MODEL_ROOT: str = "./models"
    VERIFY_THRESHOLD: float = 0.65
    RECOGNITION_THRESHOLD: float = 0.7

    # 数据库配置
    DB_TYPE: str = "sqlite"
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = ""
    MYSQL_DB: str = "face_recognition_db"
    SQLITE_PATH: str = "./face_recognition.db"

    # 管理员配置
    ADMIN_TOKEN: str = "admin123456"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


settings = Settings()