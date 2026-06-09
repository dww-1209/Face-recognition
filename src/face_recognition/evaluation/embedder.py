import logging
from pathlib import Path

import cv2
import numpy as np

# 复用 M1 的 FacePipeline Protocol——评估只是另一个调用方
from face_recognition.domain.errors import MultipleFacesError, NoFaceError
from face_recognition.domain.interfaces import FacePipeline
from face_recognition.evaluation.lfw_loader import LfwImage
from face_recognition.evaluation.types import EvalEncoding

logger = logging.getLogger(__name__)


def encode_image_paths(
    pipeline: FacePipeline,
    paths: list[Path],
    person_id: str,
) -> list[EvalEncoding]:
    """把一组图片路径批量编码成 EvalEncoding 列表，跳过无脸/读取失败的。

    所有路径默认是同一个 person_id——评估侧通常按"人/目录"分批调用。
    """
    out: list[EvalEncoding] = []
    for p in paths:
        # cv2.imread 失败时返回 None（不抛异常，注意！）
        img = cv2.imread(str(p))
        if img is None:
            logger.warning("无法读取图片，跳过：%s", p)
            continue
        # M1 约定：FacePipeline.encode_single 在检测不到脸时抛 NoFaceError；
        # 检测到多张脸时抛 MultipleFacesError(避免误把同框路人当作目标)。
        # 评估口径:两种异常都跳过这张照片(记 warning),单张失败不中断流水线——
        # 私人数据集里有合影或多人路人是常态,不能让一张多脸图把整次评估搞崩。
        try:
            face = pipeline.encode_single(img)
        except (NoFaceError, MultipleFacesError) as e:
            logger.warning("跳过 %s: %s", p, e)
            continue
        out.append(EvalEncoding(
            vector=face.vector,
            person_id=person_id,
            image_path=str(p),
        ))
    return out


def encode_lfw_images(
    pipeline: FacePipeline,
    images: list[LfwImage],
    *,
    max_skip_ratio: float = 0.10,
) -> list[EvalEncoding]:
    """把 LFW 内存图片批量编码。

    注意：sklearn fetch_lfw_people 给的是 RGB（H, W, 3）uint8，
    InsightFace.encode_single 期望的是 BGR（OpenCV 默认）——必须转通道。

    设计取舍：**单张失败安静跳过；总跳过率 > max_skip_ratio 直接抛异常**。
    LFW 是公开数据集质量稳定，零星检测失败正常（极端姿态/低分辨率），但批量失败 = 上游有 bug
    （例如忘了 RGB→BGR 转、传错色彩空间、模型加载错误等），评估指标会被静默污染。10% 是一个
    经验阈值——本科项目的 LFW 50 张样本里少于 5 张失败可接受，超过就停下来排查。
    """
    out: list[EvalEncoding] = []
    skipped = 0
    for lfw in images:
        # cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) = numpy 切片 [..., ::-1] 的等价语义
        # 选 cvtColor 更显式，读代码的人一眼看出"在做色彩空间转换"
        bgr = cv2.cvtColor(lfw.image, cv2.COLOR_RGB2BGR)
        try:
            face = pipeline.encode_single(bgr)
        except (NoFaceError, MultipleFacesError) as e:
            logger.warning("LFW 图跳过 %s: %s", lfw.person_name, e)
            skipped += 1
            continue
        out.append(EvalEncoding(
            vector=face.vector,
            person_id=lfw.person_name,
            # image_path 用伪 URI 标记来源——回溯时一眼看出是 LFW
            image_path=f"lfw://{lfw.person_name}",
        ))

    # 守门：跳过率太高说明整批有系统性问题，不能让评估"用一半样本得出结论"
    # 注意 len(images) 可能为 0（极端测试场景），用 max(...,1) 防 ZeroDivision
    skip_ratio = skipped / max(len(images), 1)
    if skip_ratio > max_skip_ratio:
        raise RuntimeError(
            f"LFW 编码跳过率过高: {skipped}/{len(images)} = {skip_ratio:.1%}, "
            f"阈值 {max_skip_ratio:.1%}。常见原因：忘了 RGB→BGR 转换；"
            f"InsightFace 模型加载失败；LFW 子集参数（n_persons）过小且都是难样本。"
        )
    return out
