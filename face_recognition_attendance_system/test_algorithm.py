# test_algorithm.py
import cv2
from algorithm.face_engine import FaceRecognitionEngine

def main():
    # 初始化人脸推理引擎
    engine = FaceRecognitionEngine()

    # 读取本地测试人脸图片
    img_path_1 = "test_images/yumeng.jpg"
    img_path_2 = "test_images/test_face.jpg"
    img1 = cv2.imread(img_path_1)
    img2 = cv2.imread(img_path_2)

    if img1 is None or img2 is None:
        print(f"[ERROR] 图片读取失败，请检查路径：{img_path_1} / {img_path_2}")
        return

    print("[INFO] 开始提取人脸特征向量")
    feat1 = engine.extract_feature(img1)
    feat2 = engine.extract_feature(img2)

    if feat1 is None or feat2 is None:
        print("[WARN] 图片中未检测到有效人脸，提取失败")
        return

    print("[SUCCESS] 人脸特征提取完成")
    print(f"特征向量维度：{len(feat1['embedding'])}")
    similarity = engine.compute_similarity(feat1["embedding"], feat2["embedding"])
    print(f"两张人脸相似度：{similarity:.4f}")

if __name__ == "__main__":
    main()