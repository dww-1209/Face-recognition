"""REST API：人员增删查。"""

import logging
import tempfile
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import JSONResponse

from face_recognition.api.dependencies import (
    get_pipeline,
    get_repository,
    get_recognizer,
    get_template_matrix,
)
from face_recognition.application.register_face import RegisterFace
from face_recognition.application.strategies.mean_all import MeanAllStrategy
from face_recognition.domain.entities import Person
from face_recognition.domain.errors import MultipleFacesError, NoFaceError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/persons", tags=["persons"])


@router.get("")
async def list_persons():
    """列出所有库内人员。"""
    repo = get_repository()
    persons = repo.list_all()
    return [
        {
            "person_id": p.person_id,
            "display_name": p.display_name,
            "template_count": p.template_count,
        }
        for p in persons
    ]


@router.post("")
async def register_person(
    person_id: str = Form(...),
    display_name: str = Form(...),
    strategy: str = Form("kmeans_k3"),
    images: list[UploadFile] = File(...),
):
    """注册新人：上传多张照片 → 编码 → 生成模板 → 入库。"""
    if not images:
        return JSONResponse(status_code=422, content={"detail": "至少上传 1 张照片"})

    pipeline = get_pipeline()
    repo = get_repository()

    temp_dir = Path(tempfile.mkdtemp(prefix="face_reg_"))

    try:
        # 保存上传文件到临时目录
        image_arrays = []
        for i, upload in enumerate(images):
            contents = await upload.read()
            temp_path = temp_dir / f"{i:04d}_{upload.filename or 'img.jpg'}"
            temp_path.write_bytes(contents)
            img = cv2.imread(str(temp_path))
            if img is not None:
                image_arrays.append(img)

        if not image_arrays:
            return JSONResponse(status_code=422, content={"detail": "无法读取任何有效图片"})

        # 使用 RegisterFace.register_from_frames（内存路径）
        from face_recognition.api.dependencies import create_strategy

        strat = create_strategy(strategy)
        register = RegisterFace(pipeline, repo, strat)
        person = register.register_from_frames(person_id, display_name, image_arrays)

        # 刷新识别缓存
        recognizer = get_recognizer()
        recognizer.refresh_cache()
        get_template_matrix().reload()

        return {
            "person_id": person.person_id,
            "display_name": person.display_name,
            "template_count": person.template_count,
        }

    except ValueError as e:
        return JSONResponse(status_code=422, content={"detail": str(e)})

    finally:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


@router.delete("/{person_id}")
async def delete_person(person_id: str):
    """删除人员。"""
    repo = get_repository()
    repo.remove(person_id)
    recognizer = get_recognizer()
    recognizer.refresh_cache()
    get_template_matrix().reload()
    return JSONResponse(status_code=204, content=None)


@router.get("/{person_id}/templates")
async def get_person_templates(person_id: str):
    """查看某人的模板详情。"""
    repo = get_repository()
    person = repo.get(person_id)
    if person is None:
        return JSONResponse(status_code=404, content={"detail": f"人员不存在: {person_id}"})
    return {
        "person_id": person.person_id,
        "display_name": person.display_name,
        "template_count": person.template_count,
        "templates": [
            {"source": t.source, "created_at": t.created_at.isoformat()}
            for t in person.templates
        ],
    }
