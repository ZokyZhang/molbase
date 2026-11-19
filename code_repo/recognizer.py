# -*- coding: utf-8 -*-
"""人脸识别模块：调用 detector 的方法完成识别流程。"""

from detector import load_image, detect_face


def recognize_person(image_path: str) -> str:
    """识别人脸身份：加载图像 -> 检测人脸 -> 返回身份标签。"""
    image = load_image(image_path)
    faces = detect_face(image)
    if not faces:
        return "unknown"
    return "person_001"


def extract_embedding(image: str) -> list[float]:
    """提取人脸特征向量（模拟）。"""
    return [0.1, 0.2, 0.3]
