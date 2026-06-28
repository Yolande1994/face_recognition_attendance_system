from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.exceptions import register_exception_handlers
from algorithm.face_engine import FaceRecognitionEngine
from database.db import FaceDB, init_db
from utils.image_helper import file_to_cv2
from schemas.base import BaseResponse
from schemas.response import UserInfo, VerifyResult, CheckinResult, CheckinRecord, UserListItem

face_engine: FaceRecognitionEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global face_engine
    face_engine = FaceRecognitionEngine(model_root=settings.MODEL_ROOT)
    init_db()
    yield
    face_engine.release()


app = FastAPI(
    title="人脸识别验证系统",
    version="2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

register_exception_handlers(app)


def get_db():
    db = FaceDB()
    try:
        yield db
    finally:
        db.close()


def admin_required(x_admin_token: str = Header(...)):
    if x_admin_token != settings.ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="管理员权限验证失败")


async def extract_face_feature(file: UploadFile):
    img = await file_to_cv2(file)
    result = face_engine.extract_feature(img)
    if result is None:
        raise HTTPException(status_code=400, detail="未检测到有效人脸，请正对摄像头")
    return result


# ========== 注册验证模块 ==========

@app.post("/register", summary="用户人脸注册", response_model=BaseResponse[UserInfo])
async def register(
    user_id: str = Form(...),
    user_name: str = Form(...),
    file: UploadFile = File(...),
    db: FaceDB = Depends(get_db)
):
    if db.user_exists(user_id):
        raise HTTPException(status_code=400, detail="该用户ID已注册")

    result = await extract_face_feature(file)
    success = db.add_user_face(user_id, user_name, result["embedding"])
    if not success:
        raise HTTPException(status_code=500, detail="注册失败，数据库错误")

    return BaseResponse(
        code=200,
        message="注册成功",
        data=UserInfo(user_id=user_id, user_name=user_name)
    )


@app.post("/verify", summary="用户人脸验证", response_model=BaseResponse[VerifyResult])
async def verify(
    user_id: str = Form(...),
    file: UploadFile = File(...),
    db: FaceDB = Depends(get_db)
):
    user_info = db.get_user_face(user_id)
    if not user_info:
        raise HTTPException(status_code=404, detail="用户未注册")

    result = await extract_face_feature(file)
    similarity = face_engine.compute_similarity(result["embedding"], user_info["face_embedding"])
    is_success = similarity >= settings.VERIFY_THRESHOLD

    return BaseResponse(
        code=200,
        message="验证成功" if is_success else "验证失败",
        data=VerifyResult(
            user_id=user_id,
            user_name=user_info["user_name"],
            similarity=round(similarity, 4),
            threshold=settings.VERIFY_THRESHOLD,
            is_success=is_success
        )
    )


@app.get("/health", summary="服务健康检查", response_model=BaseResponse)
async def health():
    return BaseResponse(code=200, message="服务正常运行")


# ========== 自动打卡模块 ==========

@app.post("/checkin", summary="自动打卡（1:N人脸检索）", response_model=BaseResponse[CheckinResult])
async def checkin(
    file: UploadFile = File(...),
    checkin_type: str = Form("上班"),
    db: FaceDB = Depends(get_db)
):
    result = await extract_face_feature(file)
    target_embedding = result["embedding"]

    all_users = db.get_all_user_faces()
    if not all_users:
        raise HTTPException(status_code=404, detail="系统中暂无注册用户")

    all_embeddings = [u["face_embedding"] for u in all_users]
    similarities = face_engine.compute_batch_similarity(target_embedding, all_embeddings)
    best_match = face_engine.find_best_match(similarities, all_users, threshold=settings.RECOGNITION_THRESHOLD)

    if not best_match:
        return BaseResponse(
            code=200,
            message="识别失败，未匹配到有效用户",
            data=CheckinResult(is_success=False)
        )

    success = db.add_checkin_record(
        user_id=best_match["user_id"],
        user_name=best_match["user_name"],
        similarity=best_match["similarity"],
        checkin_type=checkin_type
    )
    if not success:
        raise HTTPException(status_code=500, detail="打卡记录写入失败")

    return BaseResponse(
        code=200,
        message=f"{best_match['user_name']}，打卡成功！",
        data=CheckinResult(
            is_success=True,
            user_id=best_match["user_id"],
            user_name=best_match["user_name"],
            similarity=round(best_match["similarity"], 4),
            checkin_type=checkin_type
        )
    )


@app.get("/checkin/records", summary="查询最近打卡记录", response_model=BaseResponse)
async def get_checkin_records(limit: int = 20, db: FaceDB = Depends(get_db)):
    records = db.get_checkin_records(limit=limit)
    return BaseResponse(code=200, message="查询成功", data={"records": records})


# ========== 管理员模块 ==========

@app.get("/admin/users", summary="获取所有用户列表", dependencies=[Depends(admin_required)], response_model=BaseResponse)
async def admin_get_all_users(db: FaceDB = Depends(get_db)):
    users = db.get_all_users()
    return BaseResponse(code=200, message="查询成功", data={"users": users})


@app.get("/admin/records/{user_id}", summary="查询用户打卡记录", dependencies=[Depends(admin_required)], response_model=BaseResponse)
async def admin_get_user_records(user_id: str, db: FaceDB = Depends(get_db)):
    if not db.user_exists(user_id):
        raise HTTPException(status_code=404, detail="用户不存在")
    records = db.get_user_checkin_records(user_id)
    return BaseResponse(code=200, message="查询成功", data={"user_id": user_id, "records": records})


@app.delete("/admin/users/{user_id}", summary="删除用户", dependencies=[Depends(admin_required)], response_model=BaseResponse)
async def admin_delete_user(user_id: str, db: FaceDB = Depends(get_db)):
    success = db.delete_user(user_id)
    if success:
        return BaseResponse(code=200, message=f"用户 {user_id} 及其打卡记录已删除")
    if not db.user_exists(user_id):
        raise HTTPException(status_code=404, detail="用户不存在")
    raise HTTPException(status_code=500, detail="删除失败，数据库错误")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)