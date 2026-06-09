"""
CLI 入口：Typer 命令行工具。

用法：
  # 注册
  uv run python -m face_recognition.api.cli register --dataset data/lfw_subset/train --strategy kmeans_k3
  # 单图识别
  uv run python -m face_recognition.api.cli recognize data/test/alice/001.jpg
  # 列出人员 / 删除
  uv run python -m face_recognition.api.cli list
  uv run python -m face_recognition.api.cli remove alice
  # 评估
  uv run python -m face_recognition.api.cli evaluate --dataset data/lfw_subset
  # 启动 Web 服务
  uv run python -m face_recognition.api.cli serve
"""

import logging
from pathlib import Path

import cv2
import typer

from face_recognition.api.dependencies import (
    build_pipeline,
    build_recognize_use_case,
    build_register_use_case,
    build_repository,
    create_strategy,
)
from face_recognition.domain.errors import FaceRecognitionError
from face_recognition.evaluation.run_ablation import run_ablation
from face_recognition.infrastructure.config_loader import load_config

app = typer.Typer(
    name="face-recognition",
    no_args_is_help=True,
    help="基于 ArcFace 的开放集人脸识别系统 CLI",
)


@app.command()
def register(
    dataset: str = typer.Option(
        "data/lfw_subset/train",
        "--dataset",
        "-d",
        help="训练集目录（按人分文件夹存图片）",
    ),
    strategy: str = typer.Option(
        "kmeans_k3",
        "--strategy",
        "-s",
        help="模板策略：random_one / mean_all / manual_three / kmeans_k3 / all_vectors",
    ),
    db_path: str = typer.Option(
        "data/face.db",
        "--db",
        help="SQLite 数据库路径",
    ),
) -> None:
    """批量注册人员到向量库。

    数据集目录下每个子文件夹 = 一个人，文件夹名 = person_id。
    manual_three 策略会读取 <person>/subset_0..2 子文件夹。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    config = load_config()
    pipeline = build_pipeline(config)
    repository = build_repository(config)
    strategy_obj = create_strategy(strategy)
    use_case = build_register_use_case(
        config, pipeline=pipeline, repository=repository, strategy=strategy_obj
    )
    stats = use_case.execute(dataset)
    typer.echo(
        f"注册完成：成功 {stats['success']} 人 / 跳过 {stats['skipped']} 人；"
        f"处理图片 {stats['total_photos']} 张，跳过 {stats['skipped_photos']} 张"
    )


@app.command()
def recognize(image: Path = typer.Argument(..., help="待识别图片路径")) -> None:
    """识别单张图片中的人脸。"""
    logging.basicConfig(level=logging.WARNING)
    config = load_config()
    use_case = build_recognize_use_case(config)
    img = cv2.imread(str(image))
    if img is None:
        typer.echo(f"无法读取图片：{image}", err=True)
        raise typer.Exit(code=1)
    try:
        result = use_case.execute_single(img)
    except FaceRecognitionError as e:
        typer.echo(f"识别失败 [{e.code}]：{e}", err=True)
        raise typer.Exit(code=2)
    if result.person_id is None:
        typer.echo(
            f"未知人员（最高相似度 {result.similarity:.4f} < 阈值 {result.threshold}）"
        )
    else:
        typer.echo(
            f"识别为：{result.person_id}（相似度 {result.similarity:.4f}）"
        )


@app.command(name="list")
def list_persons(
    db_path: str = typer.Option("data/face.db", "--db", help="SQLite 数据库路径"),
) -> None:
    """列出库内所有人员。"""
    logging.basicConfig(level=logging.WARNING)
    repo = build_repository(load_config())
    persons = repo.list_all()
    if not persons:
        typer.echo("（库为空）")
        return
    typer.echo(f"共 {len(persons)} 人：")
    for p in persons:
        typer.echo(f"  {p.person_id}  ({p.template_count} 模板)  — {p.display_name}")


@app.command()
def remove(
    person_id: str = typer.Argument(..., help="要删除的人员 ID"),
    db_path: str = typer.Option("data/face.db", "--db", help="SQLite 数据库路径"),
) -> None:
    """从库中删除人员。"""
    logging.basicConfig(level=logging.WARNING)
    repo = build_repository(load_config())
    try:
        repo.remove(person_id)
        typer.echo(f"已删除：{person_id}")
    except FaceRecognitionError as e:
        typer.echo(f"删除失败 [{e.code}]：{e}", err=True)
        raise typer.Exit(code=2)


@app.command()
def evaluate(
    dataset: str = typer.Option(
        "data/lfw_subset",
        "--dataset",
        "-d",
        help="数据集根目录（按人分文件夹，由 data_split 按人 80/20 切分）",
    ),
    output: str = typer.Option("reports", "--output", "-o", help="报告输出目录"),
    n_lfw: int = typer.Option(50, "--n-lfw", "-n", help="LFW 库外陌生人数"),
) -> None:
    """运行 5 策略消融实验。输出 ROC 曲线、EER、TAR@FAR 到 reports/。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    config = load_config()
    pipeline = build_pipeline(config)
    all_metrics = run_ablation(
        dataset_root=Path(dataset),
        output_dir=Path(output),
        pipeline=pipeline,
        n_lfw=n_lfw,
        seed=config.evaluation.random_seed,
    )
    typer.echo("\n===== 5 策略消融结果 =====")
    for m in all_metrics:
        typer.echo(
            f"{m.strategy_name:15s}  EER={m.eer:.4f}  "
            f"TAR@FAR=1e-3={m.tar_at_far_1e3:.4f}  "
            f"({m.n_genuine} Gen / {m.n_impostor} Imp)"
        )


@app.command()
def serve(
    config_path: str = typer.Option(
        "config.yaml", "--config", "-c", help="配置文件路径"
    ),
) -> None:
    """启动 FastAPI 实时识别 Web 服务。"""
    import uvicorn

    cfg = load_config(config_path)
    uvicorn.run(
        "face_recognition.api.server:app",
        host=cfg.api.host,
        port=cfg.api.port,
        reload=False,
    )


if __name__ == "__main__":
    app()
