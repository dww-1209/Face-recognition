class FaceRecognitionError(Exception):
    """所有领域异常的基类。"""

    code: str = "FACE_RECOGNITION_ERROR"


class NoFaceError(FaceRecognitionError):
    """图像中未检测到人脸。"""

    code = "NO_FACE"


class MultipleFacesError(FaceRecognitionError):
    """图像中检测到多张人脸（encode_single 要求恰好一张）。"""

    code = "MULTIPLE_FACES"

    def __init__(self, count: int = 0, message: str | None = None) -> None:
        super().__init__(message or f"检出 {count} 张脸（要求 1 张）")
        self.count = count


class PersonNotFoundError(FaceRecognitionError):
    """数据库中未找到指定人员。"""

    code = "PERSON_NOT_FOUND"


class DuplicatePersonError(FaceRecognitionError):
    """尝试注册已存在的人员。"""

    code = "DUPLICATE_PERSON"


class LowConfidenceError(FaceRecognitionError):
    """识别相似度低于阈值（保留给强制匹配场景）。"""

    code = "LOW_CONFIDENCE"


class PersonHasNoTemplatesError(FaceRecognitionError):
    """人员没有任何模板向量。"""

    code = "NO_TEMPLATES"


class CameraDisconnectedError(FaceRecognitionError):
    """摄像头断开连接。"""

    code = "CAMERA_LOST"


class EncodingError(FaceRecognitionError):
    """编码异常：模型输出零向量、L2 归一化分母为 0 等模型问题。"""

    code = "ENCODING_ERROR"
