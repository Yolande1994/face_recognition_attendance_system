import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceRecognitionEngine:
    def __init__(self, model_root: str = "./models"):
        self.app = FaceAnalysis(
            name="buffalo_l",
            root=model_root,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def extract_feature(self, img_array: np.ndarray) -> dict | None:
        faces = self.app.get(img_array)
        if len(faces) == 0:
            return None

        face = faces[0]
        return {
            "bbox": face["bbox"].tolist(),
            "kps": face["kps"].tolist(),
            "embedding": face["embedding"].tolist()
        }

    @staticmethod
    def compute_similarity(embedding1: list, embedding2: list) -> float:
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        dot = np.dot(emb1, emb2)
        norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        return float(dot / norm)

    @staticmethod
    def compute_batch_similarity(target_embedding: list, all_embeddings: list) -> np.ndarray:
        if len(all_embeddings) == 0:
            return np.array([])

        target = np.array(target_embedding)
        all_embs = np.array(all_embeddings)

        dot_products = np.dot(all_embs, target)
        target_norm = np.linalg.norm(target)
        all_norms = np.linalg.norm(all_embs, axis=1)

        return dot_products / (target_norm * all_norms)

    @staticmethod
    def find_best_match(similarities: np.ndarray, all_users: list, threshold: float = 0.7) -> dict | None:
        if len(similarities) == 0:
            return None

        best_idx = np.argmax(similarities)
        best_similarity = float(similarities[best_idx])

        if best_similarity >= threshold:
            best_user = all_users[best_idx]
            return {
                "user_id": best_user["user_id"],
                "user_name": best_user["user_name"],
                "similarity": best_similarity
            }
        return None

    def release(self):
        del self.app