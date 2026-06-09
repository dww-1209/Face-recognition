from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from face_recognition.evaluation.run_ablation import run_ablation


def _unit_vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_dataset(root: Path, n_persons: int, n_imgs_per_person: int) -> None:
    """造合成数据集:每人若干**可解码**的 jpg。

    embedder 调用 `cv2.imread()`,b"fake" 字节会让它返回 None 然后被静默跳过——
    测试虽然过但根本没走通主流程(假阴性)。这里写一张 10×10 黑图保证 imread 成功;
    像素内容无关紧要,因为下面 fake_pipeline.encode_single 是 mock 的不看像素。
    """
    import cv2

    for i in range(n_persons):
        d = root / f"person_{i:02d}"
        d.mkdir()
        for j in range(n_imgs_per_person):
            # 每张图的像素**必须不同**,否则下面 fake_encode 用 img.tobytes() 算
            # seed 时所有图都映射到同一向量,所有 train/test 配对都成 genuine,
            # 评估流水线虽然跑完但完全没意义。这里把 (person_idx, image_idx)
            # 编码到第一个像素保证唯一。
            #
            # 用 .png 而非 .jpg:JPEG 是有损压缩,10×10 图上一像素的微小差异在
            # 解码后很可能被抹平。PNG 无损,差异保得住。data_split 的 _IMG_EXTS
            # 已经包含 .png 后缀。
            img = np.zeros((10, 10, 3), dtype=np.uint8)
            img[0, 0] = (i, j, 1)
            cv2.imwrite(str(d / f"{j:03d}.png"), img)


@pytest.mark.slow
def test_run_ablation_produces_all_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """跑通整条流水线，确认 CSV / ROC.png / 多张 hist.png / summary.md 都落盘。"""
    dataset = tmp_path / "private"
    dataset.mkdir()
    _make_dataset(dataset, n_persons=5, n_imgs_per_person=10)
    out_dir = tmp_path / "reports"
    out_dir.mkdir()

    # mock pipeline：每张图返回一个稳定的向量（同人接近、异人发散）
    from face_recognition.domain.entities import FaceEncoding

    fake_pipeline = MagicMock()
    fake_pipeline.model_version = "buffalo_l"

    # 用确定性的 seed:基于内容哈希(img 的 bytes)而不是对象 id。
    # CPython 的 id() 在对象释放后会复用,跨运行不可复现,与项目"RANDOM_SEED=42 全局可复现"原则冲突。
    # hash(bytes) 是稳定的(同一进程内同 bytes 同 hash;PYTHONHASHSEED 固定后跨运行也稳定)。
    def fake_encode(img):
        seed = abs(hash(img.tobytes())) % (2**32)
        return FaceEncoding(vector=_unit_vec(seed), model_version="buffalo_l")

    fake_pipeline.encode_single.side_effect = fake_encode

    # mock LFW loader 返回 3 张陌生人，省得真下载
    from face_recognition.evaluation import lfw_loader
    fake_lfw = [
        lfw_loader.LfwImage(image=np.zeros((250, 250, 3), dtype=np.uint8), person_name=f"LFW_{i}")
        for i in range(3)
    ]
    monkeypatch.setattr(lfw_loader, "load_lfw_subset", lambda **kw: fake_lfw)

    run_ablation(
        dataset_root=dataset,
        output_dir=out_dir,
        pipeline=fake_pipeline,
        n_lfw=3,
        seed=42,
    )

    # 6 件落盘：summary.csv / roc_curves.png / 5×hist_*.png / summary.md
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "roc_curves.png").exists()
    assert (out_dir / "summary.md").exists()
    # 5 个策略各自一张直方图
    hists = list(out_dir.glob("hist_*.png"))
    assert len(hists) == 5
