"""注册用例：从数据集目录批量注册人员到向量库。"""

import logging
from pathlib import Path

import cv2
import numpy as np

from face_recognition.domain.entities import FaceEncoding, Person, Template
from face_recognition.domain.errors import MultipleFacesError, NoFaceError
from face_recognition.domain.interfaces import FacePipeline, PersonRepository, TemplateStrategy

logger = logging.getLogger(__name__)

# 与 manual_three 策略约定的子文件夹名
SUBSET_DIRS = ["subset_0", "subset_1", "subset_2"]


class RegisterFace:
    """注册用例：遍历数据集目录，为每人编码并存入向量库。"""

    def __init__(
        self,
        pipeline: FacePipeline,
        repository: PersonRepository,
        strategy: TemplateStrategy,
    ):
        self.pipeline = pipeline
        self.repository = repository
        self.strategy = strategy

    def execute(self, dataset_dir: str | Path) -> dict:
        """
        批量注册。

        Args:
            dataset_dir: 数据集根目录，结构为 <dataset_dir>/<person_name>/*.jpg

        Returns:
            {"success": int, "skipped": int, "total_photos": int, "skipped_photos": int}
        """
        dataset_dir = Path(dataset_dir)
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"数据集目录不存在: {dataset_dir}")

        stats = {"success": 0, "skipped": 0, "total_photos": 0, "skipped_photos": 0}

        for person_dir in sorted(dataset_dir.iterdir()):
            if not person_dir.is_dir():
                continue

            person_id = person_dir.name
            try:
                person = self._register_person(person_dir, person_id)
                self.repository.add(person)
                stats["success"] += 1
                logger.info(f"注册成功: {person_id} ({person.template_count} 个模板)")
            except ValueError as e:
                stats["skipped"] += 1
                logger.error(f"跳过 {person_id}: {e}")

        logger.info(
            f"注册完成: 成功 {stats['success']} 人, "
            f"跳过 {stats['skipped']} 人, "
            f"共 {stats['total_photos']} 张照片, "
            f"跳过 {stats['skipped_photos']} 张"
        )
        return stats

    def _register_person(self, person_dir: Path, person_id: str) -> Person:
        """注册单个人。"""
        # 检测是否为 manual_three 策略且存在子文件夹
        if self.strategy.name == "manual_three" and self._has_subsets(person_dir):
            templates = self._build_manual_templates(person_dir)
        else:
            images = self._load_images(person_dir)
            encodings = self._encode_images(images, person_dir)
            if not encodings:
                raise ValueError("全部照片无法识别人脸")
            templates = self.strategy.build(encodings)

        return Person(
            person_id=person_id,
            display_name=person_id,
            templates=tuple(templates),
        )

    def _has_subsets(self, person_dir: Path) -> bool:
        return (person_dir / "subset_0").is_dir()

    def _build_manual_templates(self, person_dir: Path) -> list[Template]:
        """manual_three 专用：从子文件夹分组编码。"""
        from face_recognition.application.strategies.manual_three import (
            ManualThreeStrategy,
        )

        strategy = self.strategy
        if not isinstance(strategy, ManualThreeStrategy):
            strategy = ManualThreeStrategy()

        groups: list[list[FaceEncoding]] = []
        for subset_name in SUBSET_DIRS:
            subset_dir = person_dir / subset_name
            if subset_dir.is_dir():
                images = self._load_images(subset_dir)
                encodings = self._encode_images(images, subset_dir)
                groups.append(encodings)
            else:
                groups.append([])

        return strategy.build_from_groups(groups)

    def _load_images(self, directory: Path) -> list[np.ndarray]:
        """加载目录下所有图片（支持 jpg/png，跳过子文件夹）。"""
        images = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for img_path in sorted(directory.glob(ext)):
                img = cv2.imread(str(img_path))
                if img is not None:
                    images.append(img)
        return images

    def register_from_frames(
        self,
        person_id: str,
        display_name: str,
        frames: list[np.ndarray],
    ) -> Person:
        """从内存帧列表注册（HTTP 上传场景）。

        与 execute 的区别：
          - 输入是已解码的 ndarray 列表（不是磁盘路径）
          - 直接返回 Person 给上层做响应序列化
        """
        encodings: list[FaceEncoding] = []
        for idx, frame in enumerate(frames):
            try:
                enc = self.pipeline.encode_single(frame)
                encodings.append(enc)
            except (NoFaceError, MultipleFacesError) as e:
                logger.warning("跳过第 %d 张: %s", idx, e)

        if not encodings:
            raise ValueError(
                f"{person_id}: 上传的 {len(frames)} 张全部无法提取人脸"
            )

        templates = self.strategy.build(encodings)
        person = Person(
            person_id=person_id,
            display_name=display_name,
            templates=tuple(templates),
        )
        self.repository.add(person)
        return person

    def _encode_images(
        self, images: list[np.ndarray], source_dir: Path
    ) -> list[FaceEncoding]:
        """批量编码图片，单张失败记 warning 跳过。"""
        encodings = []
        for i, img in enumerate(images):
            try:
                enc = self.pipeline.encode_single(img)
                encodings.append(enc)
            except (NoFaceError, MultipleFacesError) as e:
                logger.warning(f"{source_dir.name}/img_{i}: {e}, 跳过")
        return encodings
