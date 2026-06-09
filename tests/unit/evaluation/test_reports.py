from pathlib import Path

import numpy as np
import pytest

from face_recognition.evaluation.reports import (
    write_csv,
    plot_roc_curves,
    plot_score_histogram,
    write_markdown,
)
from face_recognition.evaluation.types import PairResult, StrategyMetrics


def _make_metrics(name: str, eer: float = 0.05) -> StrategyMetrics:
    return StrategyMetrics(
        strategy_name=name,
        eer=eer,
        eer_threshold=0.62,
        tar_at_far_1e3=0.91,
        top1_accuracy=0.96,
        top1_with_threshold=0.93,
        roc_fpr=np.array([0.0, 0.05, 1.0]),
        roc_tpr=np.array([0.0, 0.95, 1.0]),
        n_genuine=200,
        n_impostor=4500,
    )


def test_write_csv_creates_file_with_correct_columns(tmp_path: Path):
    metrics = [_make_metrics("kmeans_k3"), _make_metrics("mean_all", eer=0.07)]
    csv_path = tmp_path / "summary.csv"
    write_csv(metrics, csv_path)
    assert csv_path.exists()
    # 用 pandas 读回来核对——别手撕 CSV 字符串
    import pandas as pd
    df = pd.read_csv(csv_path)
    assert set(df.columns) >= {
        "strategy_name", "eer", "tar_at_far_1e3", "top1_accuracy", "top1_with_threshold"
    }
    assert len(df) == 2


def test_plot_roc_curves_creates_png(tmp_path: Path):
    metrics = [_make_metrics("a"), _make_metrics("b")]
    out = tmp_path / "roc.png"
    plot_roc_curves(metrics, out)
    assert out.exists()
    # PNG 文件至少应该有几 KB——空图也得有 magic header
    assert out.stat().st_size > 1000


def test_plot_score_histogram_separates_genuine_and_impostor(tmp_path: Path):
    pairs = [
        PairResult(0.9, True, "a", "a"),
        PairResult(0.85, True, "b", "b"),
        PairResult(0.3, False, "a", "b"),
        PairResult(0.2, False, "b", "a"),
    ]
    out = tmp_path / "hist.png"
    plot_score_histogram(pairs, "kmeans_k3", out)
    assert out.exists()


def test_write_markdown_includes_metrics_and_image_refs(tmp_path: Path):
    metrics = [_make_metrics("kmeans_k3", eer=0.04)]
    md = tmp_path / "summary.md"
    # 报告需要引用之前生成的 png——传相对路径
    write_markdown(
        metrics=metrics,
        output_path=md,
        roc_image="roc_curves.png",
        hist_images={"kmeans_k3": "hist_kmeans_k3.png"},
    )
    text = md.read_text()
    assert "kmeans_k3" in text
    assert "0.04" in text or "4.00%" in text  # eer 数字格式由实现决定
    assert "roc_curves.png" in text
