# -*- coding: utf-8 -*-
"""
示例代码仓库：用于演示代码代工（AST 符号索引 + 关联方法定位）。

模拟一个简单的人脸识别系统的两个模块：
- detector.py：人脸检测
- recognizer.py：人脸识别（调用 detector 的方法）
"""


def load_image(path: str) -> str:
    """加载图像（模拟），返回图像数据句柄。"""
    return f"image:{path}"


def detect_face(image: str) -> list[dict]:
    """检测图像中的人脸，返回人脸框列表。"""
    return [{"box": [0, 0, 100, 100], "confidence": 0.98}]
