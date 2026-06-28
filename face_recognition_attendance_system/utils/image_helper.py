import cv2
import numpy as np
from fastapi import UploadFile, HTTPException


async def file_to_cv2(file: UploadFile) -> np.ndarray:
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="无效的图片文件")
        return img
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图片读取失败: {str(e)}")