# M4 实时识别 Web 服务实施计划（FastAPI + WebSocket + IoU 跟踪）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 起一个 FastAPI 应用：后端读摄像头、每帧检测、IoU 跟踪、识别按需触发，再通过 WebSocket 把"JPEG 帧 + 识别结果 JSON"推给浏览器；同时提供 REST 端点支持人员动态增删。

**Architecture:** 三个并发实体 = 采集线程（持续 cap.read）+ 识别工作循环（FastAPI startup 启的后台 task，从最新帧拉一帧跑检测+识别）+ WebSocket 处理器（async 把最新帧 + 元数据推给所有连接的浏览器）。模板矩阵在启动时一次性从 SQLite 加载到内存，注册/删除时通过 `app.state` 暴露的"重载函数"热更新。

**Tech Stack:** FastAPI（HTTP + WebSocket）、uvicorn、OpenCV `VideoCapture`、threading（采集）、asyncio（推流）、numpy（矩阵乘法检索）、starlette StaticFiles（托管前端）、pytest + httpx + TestClient。

---

## 任务清单（14 个）

> 教材风格：首次出现的 API/装饰器/方法详细解释，第二次起从简。M1+M2 已经讲过的（`@dataclass(frozen=True)`、`np.linalg.norm`、`pytest fixture`、`MagicMock.side_effect`、`@dataclass`、`Protocol`、`cv2.imread`、`np.argmax`、`@app.command()` 等）一律不再重复。

| # | Task | 类型 |
| --- | --- | --- |
| 0 | M1 接口预扩展（domain 加 DetectedFace + Pipeline.detect_and_encode + RegisterFace.register_from_frames） | TDD |
| 1 | IoU 跟踪器（`infrastructure/iou_tracker.py`） | TDD |
| 2 | OpenCV 摄像头封装（`infrastructure/camera_capture.py`） | TDD（mock） |
| 3 | 模板矩阵服务（`application/template_matrix.py`） | TDD |
| 4 | 帧渲染器（`infrastructure/frame_renderer.py`：画框/写名/编码 JPEG） | TDD |
| 5 | RecognizeFrame 用例（`application/recognize_frame.py`：跟踪 + 按需识别） | TDD |
| 6 | FastAPI 应用骨架（`api/server.py`：lifespan + 静态托管） | TDD |
| 7 | REST: GET /api/persons 列表 | TDD |
| 8 | REST: POST /api/persons 注册（multipart 上传 ≥1 张图） | TDD |
| 9 | REST: DELETE /api/persons/{id} | TDD |
| 10 | REST: GET /api/persons/{id}/templates | TDD |
| 11 | WebSocket /ws/stream（推 JPEG + JSON 元数据） | TDD |
| 12 | 全局异常处理器（FaceRecognitionError → HTTP 状态码 + JSON） | TDD |
| 13 | 端到端冒烟测试（启服务 + httpx 调几个 REST） | 集成 |

---

### Task 0: M1 接口预扩展（实时场景需要的额外能力）

**Files:**
- Modify: `src/face_recognition/domain/entities.py`（新增 `DetectedFace`）
- Modify: `src/face_recognition/domain/interfaces.py`（`FacePipeline` 增加 `detect_and_encode`）
- Modify: `src/face_recognition/infrastructure/insightface_pipeline.py`（实现新方法）
- Modify: `src/face_recognition/application/register_face.py`（新增 `register_from_frames`）
- Test: 在已有的 unit 测试文件里追加用例

**为什么需要这个 Task**：M1 的接口是为"CLI + 单图"设计的——`FacePipeline.encode(image) -> list[FaceEncoding]` 把检测出的 bbox 信息丢了，`RegisterFace` 也只接受文件夹路径。实时识别和 Web 注册要求：
1. 同时拿到 **bbox + encoding**（因为要在画面上画框）
2. 接受 **内存中的 frame 列表**（HTTP 上传图片解码到内存即可，不必落盘）

不在 M1 里加是因为 M1 写完时还不知道这两个需求；现在补——这是清洁架构里的标准做法（domain 层需要时可演化，但要保持接口最小）。

- [ ] **Step 1: 写 `DetectedFace` 实体的失败测试**

```python
# tests/unit/test_entities.py 追加
import numpy as np
from face_recognition.domain.entities import DetectedFace, FaceEncoding


def test_detected_face_carries_bbox_and_encoding():
    enc = FaceEncoding(vector=np.ones(512, dtype=np.float32) / np.sqrt(512), model_version="buffalo_l")
    df = DetectedFace(bbox=(10, 20, 100, 200), encoding=enc)
    assert df.bbox == (10, 20, 100, 200)
    assert df.encoding is enc
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/unit/test_entities.py::test_detected_face_carries_bbox_and_encoding -v
```

- [ ] **Step 3: 在 `entities.py` 加 `DetectedFace`**

```python
# domain/entities.py 追加
@dataclass(frozen=True)
class DetectedFace:
    """一张人脸在某帧中的位置 + 编码。

    实时场景使用：bbox 用于画框，encoding 用于查询模板矩阵。
    M1 单图注册用 FaceEncoding 足够，不必关心 bbox。
    """
    # bbox 形式：(x1, y1, x2, y2) 整数像素坐标，左上 + 右下
    # 用 tuple 而非 list：四个值一旦定下就不变，配合 frozen 保证不可变
    bbox: tuple[int, int, int, int]
    encoding: FaceEncoding
```

- [ ] **Step 4: 在 `interfaces.py` 给 `FacePipeline` 加方法**

```python
# domain/interfaces.py 修改 FacePipeline
class FacePipeline(Protocol):
    def encode(self, image: np.ndarray) -> list[FaceEncoding]: ...
    def encode_single(self, image: np.ndarray) -> FaceEncoding: ...
    # 新增：实时场景用，返回带 bbox 的版本
    # 为什么不用 encode 替代：encode 已被 M1 大量使用且语义清晰（"只编码"），
    # 加新方法比改老方法更安全（开闭原则）。
    def detect_and_encode(self, image: np.ndarray) -> list["DetectedFace"]: ...
```

- [ ] **Step 5: 在 `insightface_pipeline.py` 实现**

```python
# infrastructure/insightface_pipeline.py 追加方法
from face_recognition.domain.entities import DetectedFace

class InsightFacePipeline:
    # ...M1 已有的 encode / encode_single...

    def detect_and_encode(self, image: np.ndarray) -> list[DetectedFace]:
        """同 encode，但保留 InsightFace 的 bbox（int4 像素坐标）。"""
        # FaceAnalysis.get(img) 返回 list[Face]，每个 Face 有：
        #   - bbox: np.ndarray shape=(4,) float32 [x1, y1, x2, y2]
        #   - normed_embedding: 已 L2 归一化的 (512,) float32
        faces = self._app.get(image)
        out: list[DetectedFace] = []
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int).tolist()
            enc = FaceEncoding(
                vector=self._normalize(f.normed_embedding),
                model_version=self._model_pack,
            )
            out.append(DetectedFace(bbox=(x1, y1, x2, y2), encoding=enc))
        return out
```

- [ ] **Step 6: 在 `register_face.py` 加 `register_from_frames`**

```python
# application/register_face.py RegisterFace 类追加方法
class RegisterFace:
    # ...M1 已有的 __init__ / execute_for_person / execute...

    def register_from_frames(
        self,
        person_id: str,
        display_name: str,
        frames: list[np.ndarray],
    ) -> Person:
        """从内存帧列表注册（HTTP 上传场景）。

        与 execute_for_person 的区别：
          - 输入是已解码的 ndarray 列表（不是磁盘路径）
          - 直接返回 Person 给上层做响应序列化（不只是计数）
        """
        encodings: list[FaceEncoding] = []
        for idx, frame in enumerate(frames):
            try:
                enc = self._pipeline.encode_single(frame)
                encodings.append(enc)
            except FaceRecognitionError as e:
                logger.warning("跳过第 %d 张: %s", idx, e)

        if not encodings:
            raise PersonHasNoTemplatesError(
                f"{person_id}: 上传的 {len(frames)} 张全部无法提取人脸"
            )

        templates = self._strategy.build(encodings)
        person = Person(
            person_id=person_id,
            display_name=display_name,
            templates=tuple(templates),
        )
        self._repo.add(person)
        return person
```

- [ ] **Step 7: 跑测试 + commit**

```bash
uv run pytest tests/unit/test_entities.py -v
git add src/face_recognition/domain src/face_recognition/infrastructure/insightface_pipeline.py src/face_recognition/application/register_face.py tests/unit/test_entities.py
git commit -m "feat(domain): 加 DetectedFace + Pipeline.detect_and_encode + RegisterFace.register_from_frames（M4 准备）"
```

---

### Task 1: IoU 跟踪器

**Files:**
- Create: `src/face_recognition/infrastructure/iou_tracker.py`
- Test: `tests/unit/infrastructure/test_iou_tracker.py`

为什么需要"跟踪器"？实时识别每帧都跑 ArcFace 编码会浪费算力（同一张脸在镜头前 5 秒 × 30 FPS = 150 帧，识别 1 次就够）。跟踪器解决两个问题：

1. **同人帧间复用身份**：第一帧识别为"alice"后，后续帧只要 IoU 匹配上同一个 box，就不再跑识别，直接沿用 alice
2. **新人触发识别**：检测到一个**新框**（之前没匹配过的）就标记"需要识别"

**IoU**（Intersection over Union）= 两个矩形的"交集面积 / 并集面积"，0~1，1 表示完全重合，0 表示不相交。两帧间同一人脸的 IoU 通常 > 0.5。

- [ ] **Step 1: 写失败的测试 `tests/unit/infrastructure/test_iou_tracker.py`**

```python
from face_recognition.infrastructure.iou_tracker import IoUTracker, Track


def test_iou_overlap_box_is_assigned_same_track_id():
    """两帧 IoU 高的框 → 同一 track_id。"""
    # 第 1 帧：1 张脸在 (10, 10, 110, 110)（左上 + 右下坐标，100×100）
    tracker = IoUTracker(iou_threshold=0.5, max_missing_frames=15)
    tracks_f1 = tracker.update([(10, 10, 110, 110)])
    assert len(tracks_f1) == 1
    tid_f1 = tracks_f1[0].track_id

    # 第 2 帧：脸略偏移到 (15, 15, 115, 115) —— IoU 应该 > 0.8
    tracks_f2 = tracker.update([(15, 15, 115, 115)])
    assert len(tracks_f2) == 1
    # 关键断言：track_id 跨帧保持
    assert tracks_f2[0].track_id == tid_f1


def test_disjoint_box_gets_new_track_id():
    """完全不重叠的框 → 新 track_id。"""
    tracker = IoUTracker(iou_threshold=0.5)
    [t1] = tracker.update([(0, 0, 100, 100)])
    [t2] = tracker.update([(500, 500, 600, 600)])
    assert t1.track_id != t2.track_id


def test_new_track_marked_needs_recognition():
    """新出现的 track 应该 needs_recognition=True，提示上层去跑识别。"""
    tracker = IoUTracker(iou_threshold=0.5)
    [t] = tracker.update([(0, 0, 100, 100)])
    assert t.needs_recognition is True


def test_recognized_track_keeps_identity_across_frames():
    """识别完后调用 set_identity；后续帧应继承 identity 且 needs_recognition=False。"""
    tracker = IoUTracker(iou_threshold=0.5)
    [t1] = tracker.update([(10, 10, 110, 110)])
    tracker.set_identity(t1.track_id, person_id="alice", similarity=0.82)

    [t2] = tracker.update([(12, 12, 112, 112)])
    assert t2.track_id == t1.track_id
    assert t2.identity == "alice"
    assert t2.similarity == 0.82
    assert t2.needs_recognition is False


def test_track_disappears_after_max_missing_frames():
    """连续 N 帧不出现 → track 被清理；之后同样位置的框拿到新 track_id。"""
    tracker = IoUTracker(iou_threshold=0.5, max_missing_frames=2)
    [t1] = tracker.update([(0, 0, 100, 100)])
    # 连续 3 帧没人脸（超过 max_missing_frames=2）
    tracker.update([])
    tracker.update([])
    tracker.update([])
    [t2] = tracker.update([(0, 0, 100, 100)])
    # 同一位置但 track_id 应该不同（旧 track 已被清理）
    assert t2.track_id != t1.track_id
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/unit/infrastructure/test_iou_tracker.py -v
```

- [ ] **Step 3: 实现 `src/face_recognition/infrastructure/iou_tracker.py`**

```python
# IoU 跟踪器：实时识别中"识别按需触发"的关键基础设施。
# 设计取舍：用最简单的 greedy IoU 匹配（每个新框找 IoU 最大的已有 track），
# 不引入卡尔曼滤波 / 匈牙利算法——35 人小场景里 greedy 完全够用。
from dataclasses import dataclass, field

# 矩形坐标的别名：(x1, y1, x2, y2) = 左上 x、左上 y、右下 x、右下 y。
# 所有坐标都是像素整数。InsightFace 检测出的 bbox 也是这种格式（float 但我们 round 到 int）。
BBox = tuple[int, int, int, int]


# 用 dataclass 而不是 frozen=True：track 是**有状态**的——身份要在跑识别后回填，
# 帧计数 missing_frames 也要每帧加。frozen 跟"会变的对象"不兼容。
# 默认 eq=True 让两个 Track 实例可以用 == 比较（按字段值），方便测试断言
@dataclass
class Track:
    """一个被持续跟踪的人脸。"""
    track_id: int
    bbox: BBox
    # identity / similarity / needs_recognition 三件套：
    #   - 新 track 默认 needs_recognition=True，identity=None
    #   - 上层跑完识别后调 tracker.set_identity 把它们更新
    identity: str | None = None
    similarity: float = 0.0
    needs_recognition: bool = True
    # 用来超时清理：每帧 +1，被匹配上重置为 0；超过 max_missing_frames 就被删
    missing_frames: int = 0


def _iou(box_a: BBox, box_b: BBox) -> float:
    """计算两个矩形的 IoU。
    几何意义：
      - 交集面积：x 方向重叠长度 × y 方向重叠长度（任一方向 ≤ 0 则面积 0）
      - 并集面积：A 面积 + B 面积 − 交集
      - IoU = 交 / 并，范围 [0, 1]
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    # max(ax1, bx1) = 交集左边界；min(ax2, bx2) = 交集右边界
    # 用 max(..., 0) 把"无重叠"（负值）夹到 0
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


@dataclass
class IoUTracker:
    """每帧调 update(boxes) 喂入新检测结果，返回同步后的 Track 列表。"""
    iou_threshold: float = 0.5
    max_missing_frames: int = 15
    # 内部状态：active 字典 = {track_id: Track}
    # field(default_factory=dict)：M1 已讲过——可变默认值的正确写法
    _tracks: dict[int, Track] = field(default_factory=dict)
    _next_id: int = 0

    def update(self, detected_boxes: list[BBox]) -> list[Track]:
        """喂入当前帧检测到的所有 bbox，返回更新后的 active tracks。

        步骤：
          1. 每个旧 track 标记 missing += 1（先假设都没出现）
          2. 每个新 box 找 IoU 最大的旧 track；若 ≥ 阈值则匹配成功（更新 bbox + missing=0）
          3. 没匹配上的 box 创建新 track（needs_recognition=True）
          4. missing > max_missing_frames 的 track 删除
        """
        # 步骤 1：所有 track 先 +1，等下匹配上的会被重置回 0
        for t in self._tracks.values():
            t.missing_frames += 1

        # 步骤 2：贪心匹配
        unmatched_boxes: list[BBox] = []
        for box in detected_boxes:
            best_id: int | None = None
            best_iou = self.iou_threshold  # 必须 ≥ 阈值才考虑匹配
            for tid, track in self._tracks.items():
                # 已经被本帧匹配过的 track（missing=0）跳过——不允许两个 box 抢同一个 track
                if track.missing_frames == 0:
                    continue
                score = _iou(track.bbox, box)
                if score >= best_iou:
                    best_iou = score
                    best_id = tid
            if best_id is not None:
                # 匹配成功：更新 bbox，重置 missing；identity 不动
                t = self._tracks[best_id]
                t.bbox = box
                t.missing_frames = 0
            else:
                unmatched_boxes.append(box)

        # 步骤 3：未匹配的新建
        for box in unmatched_boxes:
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = Track(track_id=tid, bbox=box)

        # 步骤 4：清理过期 track
        # list(...keys()) 拷贝 keys 避免遍历时改字典抛 RuntimeError
        for tid in list(self._tracks.keys()):
            if self._tracks[tid].missing_frames > self.max_missing_frames:
                del self._tracks[tid]

        # 只返回本帧实际出现的 track（missing=0）；已经"消失但尚未超时"的不渲染
        return [t for t in self._tracks.values() if t.missing_frames == 0]

    def set_identity(self, track_id: int, person_id: str | None, similarity: float) -> None:
        """识别完成后回填身份。person_id=None 表示"识别过但低于阈值"——下次不再重复识别。"""
        if track_id not in self._tracks:
            return  # track 已经被清理（罕见但要防）；安静忽略
        t = self._tracks[track_id]
        t.identity = person_id
        t.similarity = similarity
        t.needs_recognition = False
```

- [ ] **Step 4: 跑测试确认通过**

```bash
uv run pytest tests/unit/infrastructure/test_iou_tracker.py -v
```

预期：5 passed

- [ ] **Step 5: commit**

```bash
git add src/face_recognition/infrastructure/iou_tracker.py tests/unit/infrastructure/test_iou_tracker.py
git commit -m "feat(infra): 加 IoU 跟踪器（greedy 匹配 + 识别按需）"
```

---

### Task 2: OpenCV 摄像头封装

**Files:**
- Create: `src/face_recognition/infrastructure/camera_capture.py`
- Test: `tests/unit/infrastructure/test_camera_capture.py`

为什么要"封装" `cv2.VideoCapture`？两点：

1. **统一错误**：cv2 打不开摄像头时返回 `not isOpened()`，不抛异常——我们包一层抛 `CameraDisconnectedError`，让上层只管 try/except 领域异常
2. **方便测试**：测试时注入 mock 的 `VideoCapture`，不依赖真摄像头硬件

- [ ] **Step 1: 写失败的测试**

```python
from unittest.mock import MagicMock

import numpy as np
import pytest

from face_recognition.domain.errors import CameraDisconnectedError
from face_recognition.infrastructure.camera_capture import CameraCapture


def test_camera_open_failure_raises_domain_error():
    """cv2.VideoCapture 打不开（isOpened()=False）→ 抛 CameraDisconnectedError。"""
    fake_cv2_cap = MagicMock()
    fake_cv2_cap.isOpened.return_value = False
    # cap_factory 是依赖注入点：测试传 mock 工厂，生产传 cv2.VideoCapture
    with pytest.raises(CameraDisconnectedError):
        CameraCapture(device_index=0, resolution=(640, 480), cap_factory=lambda i: fake_cv2_cap)


def test_read_returns_frame_when_ok():
    """read() 返回 (True, frame) 时正常出帧。"""
    fake_cv2_cap = MagicMock()
    fake_cv2_cap.isOpened.return_value = True
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    fake_cv2_cap.read.return_value = (True, fake_frame)

    cam = CameraCapture(device_index=0, resolution=(640, 480), cap_factory=lambda i: fake_cv2_cap)
    frame = cam.read()
    # 形状校验，确认我们没乱处理 frame
    assert frame.shape == (480, 640, 3)


def test_read_returns_false_raises_domain_error():
    """read() 返回 (False, None) 表示采集失败 → 抛 CameraDisconnectedError。"""
    fake_cv2_cap = MagicMock()
    fake_cv2_cap.isOpened.return_value = True
    fake_cv2_cap.read.return_value = (False, None)

    cam = CameraCapture(device_index=0, resolution=(640, 480), cap_factory=lambda i: fake_cv2_cap)
    with pytest.raises(CameraDisconnectedError):
        cam.read()


def test_release_calls_cv2_release():
    """release() 应该转发到 cv2 cap.release，避免摄像头资源泄漏。"""
    fake_cv2_cap = MagicMock()
    fake_cv2_cap.isOpened.return_value = True
    cam = CameraCapture(device_index=0, resolution=(640, 480), cap_factory=lambda i: fake_cv2_cap)
    cam.release()
    fake_cv2_cap.release.assert_called_once()
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/unit/infrastructure/test_camera_capture.py -v
```

- [ ] **Step 3: 实现 `src/face_recognition/infrastructure/camera_capture.py`**

```python
from collections.abc import Callable

import cv2
import numpy as np

from face_recognition.domain.errors import CameraDisconnectedError

# Callable[[int], 任意]：M1 中已用过此类型注解
# 这里 cap_factory 接收 device_index，返回一个有 isOpened/read/release 方法的对象（duck typing）
# 默认指向 cv2.VideoCapture；测试时换成返回 MagicMock 的 lambda
_CapFactory = Callable[[int], "cv2.VideoCapture"]


class CameraCapture:
    """OpenCV VideoCapture 的薄封装，把 cv2 错误码翻译成领域异常。"""

    def __init__(
        self,
        device_index: int,
        resolution: tuple[int, int],
        cap_factory: _CapFactory = cv2.VideoCapture,
    ) -> None:
        # 懒得把 cv2.VideoCapture 写死——cap_factory 让测试可以注入 mock
        self._cap = cap_factory(device_index)
        if not self._cap.isOpened():
            raise CameraDisconnectedError(f"摄像头 {device_index} 无法打开")
        # cv2 的 set 设置不一定生效（取决于驱动），但常见 USB 摄像头都支持。
        # CAP_PROP_FRAME_WIDTH/HEIGHT 是常量整数（在 cv2 命名空间下），和 ffmpeg 的属性一一对应
        w, h = resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    def read(self) -> np.ndarray:
        """读取一帧。失败抛 CameraDisconnectedError。"""
        # cv2 的 read 返回 (ret: bool, frame: np.ndarray | None)。
        # 失败原因可能是摄像头被拔、被其他程序占用、驱动崩溃。
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise CameraDisconnectedError("摄像头读取失败")
        return frame

    def release(self) -> None:
        """释放摄像头资源。FastAPI 关闭时调用。"""
        self._cap.release()
```

- [ ] **Step 4: 跑测试确认通过**

```bash
uv run pytest tests/unit/infrastructure/test_camera_capture.py -v
```

预期：4 passed

- [ ] **Step 5: commit**

```bash
git add src/face_recognition/infrastructure/camera_capture.py tests/unit/infrastructure/test_camera_capture.py
git commit -m "feat(infra): 加 CameraCapture（cv2 错误 → CameraDisconnectedError）"
```

---

### Task 3: 模板矩阵服务（内存检索 + 热更新）

**Files:**
- Create: `src/face_recognition/application/template_matrix.py`
- Test: `tests/unit/application/test_template_matrix.py`

为什么要单独抽一个 `TemplateMatrixService`？识别用的是"内存里的 (M, 512) 矩阵 + person_id 列表"，不是每次 SQL 查询。把"加载 / 重载 / 矩阵乘检索"封装在一个对象里：

- 服务启动时 `load()` 一次性把 SQLite 拉进来
- 注册/删人后调 `reload()` 热更新，不用重启服务
- `query(vector) -> (best_person_id, similarity)` 是单次识别的核心入口

- [ ] **Step 1: 写失败的测试**

```python
from datetime import datetime
from unittest.mock import MagicMock

import numpy as np

from face_recognition.application.template_matrix import TemplateMatrixService
from face_recognition.domain.entities import FaceEncoding, Person, Template


def _unit_vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_person(pid: str, n_templates: int, seed: int) -> Person:
    """造一个有 n_templates 个模板的 Person。"""
    templates = tuple(
        Template(
            encoding=FaceEncoding(vector=_unit_vec(seed + i), model_version="buffalo_l"),
            source=f"tpl_{i}",
            created_at=datetime(2026, 5, 21),
        )
        for i in range(n_templates)
    )
    return Person(person_id=pid, display_name=pid, templates=templates)


def test_load_builds_matrix_and_pid_list():
    """load 后 matrix.shape = (总模板数, 512)，pid_list 长度一致。"""
    repo = MagicMock()
    repo.list_all.return_value = [
        _make_person("alice", 3, seed=0),    # 3 个模板
        _make_person("bob", 1, seed=10),     # 1 个模板
    ]
    svc = TemplateMatrixService(repository=repo)
    svc.load()
    assert svc.matrix.shape == (4, 512)
    assert svc.pid_list == ["alice", "alice", "alice", "bob"]


def test_query_returns_best_matching_person():
    """query 应返回相似度最高的 person_id 和分数。"""
    repo = MagicMock()
    alice = _make_person("alice", 1, seed=0)
    bob = _make_person("bob", 1, seed=100)
    repo.list_all.return_value = [alice, bob]

    svc = TemplateMatrixService(repository=repo)
    svc.load()

    # query alice 自己的模板向量 → 应该匹配到 alice，相似度 = 1.0
    pid, sim = svc.query(alice.templates[0].encoding.vector)
    assert pid == "alice"
    assert sim > 0.99


def test_reload_picks_up_repository_changes():
    """reload 后矩阵应反映 repository 的新状态。"""
    repo = MagicMock()
    repo.list_all.return_value = [_make_person("alice", 1, seed=0)]

    svc = TemplateMatrixService(repository=repo)
    svc.load()
    assert svc.matrix.shape == (1, 512)

    # 模拟新增 bob
    repo.list_all.return_value = [_make_person("alice", 1, seed=0), _make_person("bob", 2, seed=10)]
    svc.reload()
    assert svc.matrix.shape == (3, 512)
    assert "bob" in svc.pid_list


def test_query_on_empty_matrix_returns_none():
    """库里一个人都没有时 query 应返回 (None, 0.0) 而非崩溃。"""
    repo = MagicMock()
    repo.list_all.return_value = []
    svc = TemplateMatrixService(repository=repo)
    svc.load()
    pid, sim = svc.query(_unit_vec(0))
    assert pid is None
    assert sim == 0.0
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/unit/application/test_template_matrix.py -v
```

- [ ] **Step 3: 实现 `src/face_recognition/application/template_matrix.py`**

```python
import numpy as np

from face_recognition.domain.interfaces import PersonRepository


class TemplateMatrixService:
    """把整个库的模板装进 (M, 512) 矩阵 + 长度 M 的 person_id 列表，
    支持 O(M) 矩阵乘法检索。M 在 35 人 × 平均 3 模板 ≈ 100 量级，毫秒返回。

    使用流程：
      svc = TemplateMatrixService(repo)
      svc.load()                           # 启动时一次
      pid, sim = svc.query(face_vector)    # 每次识别
      svc.reload()                         # POST/DELETE persons 后调
    """

    def __init__(self, repository: PersonRepository) -> None:
        self._repo = repository
        # 显式声明类型 + 初始化为空——load 之前 query 也能跑（返回空）
        self.matrix: np.ndarray = np.zeros((0, 512), dtype=np.float32)
        self.pid_list: list[str] = []

    def load(self) -> None:
        """从 repository 一次性加载所有人的所有模板到内存。"""
        persons = self._repo.list_all()
        # 列表推导展开"多人 × 多模板" → 一个长 list
        # 每个 Template 的 encoding.vector 已经是 L2 归一化的（domain 的不变量）
        rows: list[np.ndarray] = []
        pids: list[str] = []
        for p in persons:
            for tpl in p.templates:
                rows.append(tpl.encoding.vector)
                pids.append(p.person_id)
        if rows:
            # np.stack 把多个 (512,) 堆成 (M, 512)
            self.matrix = np.stack(rows).astype(np.float32)
        else:
            self.matrix = np.zeros((0, 512), dtype=np.float32)
        self.pid_list = pids

    def reload(self) -> None:
        """注册/删除人员后调用。等价 load——别名只是语义清晰。"""
        self.load()

    def query(self, vector: np.ndarray) -> tuple[str | None, float]:
        """单次识别：矩阵乘 → argmax → 返回 (person_id, similarity)。

        库为空时返回 (None, 0.0)；否则返回最相似那条模板对应的 person_id 和分数。
        阈值判断由调用方（RecognizeFrame 用例）做——这层只负责"找最像的"。
        """
        if self.matrix.shape[0] == 0:
            return None, 0.0
        # 矩阵乘 (M, 512) @ (512,) = (M,) 相似度向量；vector 必须已归一化
        scores = self.matrix @ vector
        best_idx = int(np.argmax(scores))
        return self.pid_list[best_idx], float(scores[best_idx])
```

- [ ] **Step 4: 跑测试确认通过**

```bash
uv run pytest tests/unit/application/test_template_matrix.py -v
```

预期：4 passed

- [ ] **Step 5: commit**

```bash
git add src/face_recognition/application/template_matrix.py tests/unit/application/test_template_matrix.py
git commit -m "feat(app): 加 TemplateMatrixService（内存矩阵 + 热重载 + 矩阵乘检索）"
```

---

### Task 4: 帧渲染器（画框 + 写名 + 编码 JPEG）

**Files:**
- Create: `src/face_recognition/infrastructure/frame_renderer.py`
- Test: `tests/unit/infrastructure/test_frame_renderer.py`

把"在 BGR 帧上画绿色矩形 + 写姓名"和"把帧编码成 JPEG bytes"这两件事抽出来——
WebSocket 处理器拿到的是"原始帧 + tracks"，要变成可推送的 JPEG bytes，所以这是必经之路。

- [ ] **Step 1: 写失败的测试**

```python
import numpy as np

from face_recognition.infrastructure.frame_renderer import (
    encode_jpeg,
    render_tracks,
)
from face_recognition.infrastructure.iou_tracker import Track


def _blank_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_render_tracks_draws_on_frame():
    """画框后帧不再全黑——至少画出来的位置有非零像素。"""
    frame = _blank_frame()
    tracks = [Track(track_id=0, bbox=(100, 100, 300, 300), identity="alice", similarity=0.85)]
    out = render_tracks(frame, tracks)
    # 矩形线条上应该有非零像素（OpenCV 默认线条颜色是绿色 (0, 255, 0)）
    # 我们检查框的左上角 1×1 区域 ——
    assert np.any(out[100:101, 100:101] > 0)
    # 输入不被修改：render_tracks 返回新数组，不污染原 frame
    assert np.all(frame == 0)


def test_render_tracks_handles_unknown_identity():
    """identity=None（未识别）不报错，应该写"未知"或类似标签。"""
    frame = _blank_frame()
    tracks = [Track(track_id=0, bbox=(50, 50, 200, 200), identity=None)]
    out = render_tracks(frame, tracks)
    # 不崩溃就是过；具体文字内容不强求（避免 OpenCV 字体测试脆性）
    assert out.shape == frame.shape


def test_encode_jpeg_returns_valid_bytes():
    """encode_jpeg 返回的 bytes 应该以 JPEG magic header 开头。"""
    frame = _blank_frame()
    data = encode_jpeg(frame, quality=80)
    # JPEG 文件头：FF D8 FF（任何编码器都遵循）
    assert data[:3] == b"\xff\xd8\xff"
    assert len(data) > 100  # 全黑图也得有几百字节
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/unit/infrastructure/test_frame_renderer.py -v
```

- [ ] **Step 3: 实现 `src/face_recognition/infrastructure/frame_renderer.py`**

```python
import cv2
import numpy as np

from face_recognition.infrastructure.iou_tracker import Track


def render_tracks(frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
    """在帧上画框 + 写姓名（或"未知"）。返回新帧，不修改原始 frame。"""
    # frame.copy() 拿到独立副本，避免污染采集线程持有的原帧
    # （多线程下原帧可能被其他消费者同时读，写它会引发竞态）
    out = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = t.bbox
        # cv2.rectangle(img, pt1, pt2, color, thickness)
        # color 用 BGR 三元组：(0, 255, 0) = 纯绿
        # 已识别人员画绿色，未识别画红色——一眼区分
        color = (0, 255, 0) if t.identity is not None else (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness=2)

        # 标签文字：identity (similarity)，None 时写"未知"
        label = f"{t.identity} ({t.similarity:.2f})" if t.identity else "未知"
        # cv2.putText(img, text, org, fontFace, fontScale, color, thickness)
        # FONT_HERSHEY_SIMPLEX 是常见无衬线字体；fontScale=0.6 大致 14px 字号
        # 文字位置 (x1, y1 - 8)：框上方 8 像素留白，避免压框线
        cv2.putText(out, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness=2)
    return out


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    """把帧编码成 JPEG bytes，供 WebSocket 推送。

    quality 80 是带宽/画质平衡点：1280×720 全黑帧约 5KB，普通画面约 50~80KB。
    quality 越高文件越大，过 95 收益递减。
    """
    # cv2.imencode 接收 ".jpg" 扩展名 → 用 JPEG 编码器；返回 (success, ndarray of bytes)
    # IMWRITE_JPEG_QUALITY 是 0~100 整数：quality 参数的 cv2 常量名
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        # 极少触发——通常是输入 frame 形状非法（如全空数组）
        raise RuntimeError("JPEG 编码失败")
    # buf 是 (N, 1) uint8 ndarray；.tobytes() 转成 Python bytes 喂给 WebSocket
    return buf.tobytes()
```

- [ ] **Step 4: 跑测试确认通过**

```bash
uv run pytest tests/unit/infrastructure/test_frame_renderer.py -v
```

预期：3 passed

- [ ] **Step 5: commit**

```bash
git add src/face_recognition/infrastructure/frame_renderer.py tests/unit/infrastructure/test_frame_renderer.py
git commit -m "feat(infra): 加 frame_renderer（画框 + 写名 + JPEG 编码）"
```

---

### Task 5: RecognizeFrame 用例（编排：检测→跟踪→按需识别）

**Files:**
- Create: `src/face_recognition/application/recognize_frame.py`
- Test: `tests/unit/application/test_recognize_frame.py`

**职责**：把单帧画面变成"已识别的 tracks 列表"。这是实时识别的"大脑"，串起 4 个组件：`FacePipeline`（检测 + 编码）、`IoUTracker`（跟踪）、`TemplateMatrixService`（查询）、阈值判定。

**为什么单独建一个用例文件**：M1 已有 `RecognizeFace`（用例：单张图 → 单个识别结果），但实时场景"多脸 + 跟踪 + 节流"逻辑完全不同——不要硬塞进 `RecognizeFace`，新建一个用例更清晰，符合 SRP（单一职责）。

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/application/test_recognize_frame.py
"""测 RecognizeFrame 用例的编排逻辑。所有外部依赖全 mock。"""
from unittest.mock import MagicMock
import numpy as np
import pytest

from face_recognition.application.recognize_frame import RecognizeFrame
from face_recognition.domain.entities import FaceEncoding
from face_recognition.infrastructure.iou_tracker import IoUTracker, Track


from face_recognition.domain.entities import DetectedFace, FaceEncoding


def _fake_face(bbox, vec_value=0.1):
    """造一个 DetectedFace（M1 Task 0 新增的 domain 实体），
    模拟 pipeline.detect_and_encode 的返回项。"""
    v = np.full(512, vec_value, dtype=np.float32)
    enc = FaceEncoding(vector=v / np.linalg.norm(v), model_version="buffalo_l")
    return DetectedFace(bbox=bbox, encoding=enc)


def test_first_frame_detects_and_recognizes():
    """第一帧：tracker 是空的 → 所有 detection 都是新 track → 全部触发识别。"""
    # mock 三个依赖
    pipeline = MagicMock()
    pipeline.detect_and_encode.return_value = [
        _fake_face((10, 10, 50, 50), vec_value=0.1),
    ]
    tracker = IoUTracker(iou_threshold=0.5, max_missing_frames=15)
    matrix = MagicMock()
    matrix.query.return_value = ("alice", 0.85)

    use_case = RecognizeFrame(pipeline=pipeline, tracker=tracker,
                              template_matrix=matrix, threshold=0.45)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracks = use_case.process_frame(frame)

    assert len(tracks) == 1
    assert tracks[0].identity == "alice"
    assert tracks[0].similarity == pytest.approx(0.85)
    assert tracks[0].needs_recognition is False  # 识别完成后置 False
    matrix.query.assert_called_once()


def test_below_threshold_marks_unknown():
    """相似度低于 threshold → identity 设为 None（未知人）。"""
    pipeline = MagicMock()
    pipeline.detect_and_encode.return_value = [
        _fake_face((10, 10, 50, 50)),
    ]
    tracker = IoUTracker()
    matrix = MagicMock()
    matrix.query.return_value = ("bob", 0.30)  # 远低于阈值 0.45

    use_case = RecognizeFrame(pipeline=pipeline, tracker=tracker,
                              template_matrix=matrix, threshold=0.45)

    tracks = use_case.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))

    assert tracks[0].identity is None
    assert tracks[0].similarity == pytest.approx(0.30)


def test_existing_track_skips_recognition():
    """已识别过的 track（needs_recognition=False）下一帧不再触发 query。"""
    pipeline = MagicMock()
    # 两帧都返回同一位置的脸 → IoU 高 → 同一 track
    pipeline.detect_and_encode.return_value = [_fake_face((10, 10, 50, 50))]
    tracker = IoUTracker(iou_threshold=0.5)
    matrix = MagicMock()
    matrix.query.return_value = ("alice", 0.85)

    use_case = RecognizeFrame(pipeline=pipeline, tracker=tracker,
                              template_matrix=matrix, threshold=0.45)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    use_case.process_frame(frame)  # 第一帧：识别一次
    use_case.process_frame(frame)  # 第二帧：同一 track，不再识别

    assert matrix.query.call_count == 1


def test_no_faces_returns_empty():
    """画面没人 → 返回空列表，不报错。"""
    pipeline = MagicMock()
    pipeline.detect_and_encode.return_value = []
    tracker = IoUTracker()
    matrix = MagicMock()

    use_case = RecognizeFrame(pipeline=pipeline, tracker=tracker,
                              template_matrix=matrix, threshold=0.45)

    tracks = use_case.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))

    assert tracks == []
    matrix.query.assert_not_called()
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/unit/application/test_recognize_frame.py -v
```

预期：4 个测试全部 FAIL（ImportError，因为模块还没建）

- [ ] **Step 3: 写实现**

```python
# src/face_recognition/application/recognize_frame.py
"""RecognizeFrame 用例：单帧画面 → 已识别的 tracks 列表。

为什么这里不用 try/except 包 NoFaceError：
- 实时场景下"画面没人"是常态，不是异常——pipeline.detect_and_encode 返回空列表即可
- M1 的 RecognizeFace 用例（单张图）抛 NoFaceError 是合理的（用户明确给了一张照片，没脸是错误）
- 这里语义不同，所以不复用，新建一个用例
"""
from dataclasses import dataclass

import numpy as np

from face_recognition.domain.interfaces import FacePipeline
from face_recognition.infrastructure.iou_tracker import IoUTracker, Track
from face_recognition.application.template_matrix import TemplateMatrixService


@dataclass
class RecognizeFrame:
    """实时识别用例。

    依赖通过 dataclass 字段注入（而非 __init__ 手写赋值）——
    这是 Python 3.10+ 的常用模式，省掉重复模板代码。

    使用方式（在 api/server.py 的识别线程里）：
        use_case = RecognizeFrame(pipeline=..., tracker=..., template_matrix=..., threshold=0.45)
        while True:
            frame = capture.read()
            tracks = use_case.process_frame(frame)
            # tracks 喂给 frame_renderer 画框 + 推到 WebSocket
    """
    pipeline: FacePipeline
    tracker: IoUTracker
    template_matrix: TemplateMatrixService
    threshold: float

    def process_frame(self, frame: np.ndarray) -> list[Track]:
        """处理一帧画面，返回更新后的所有 tracks。

        流程：
            1. pipeline 检测 + 编码（一站式）→ 拿到 list[DetectedFace]
            2. 提取 bbox → 喂给 tracker.update → 拿到 tracks（含 needs_recognition 标记）
            3. 对每个 needs_recognition=True 的 track，找到对应 face、查 matrix、判阈值
            4. tracker.set_identity 写回结果

        为什么"按需识别"而不是每帧都识别：
            ResNet100 编码一次 ~10ms，如果每帧都识别每个脸，3 个脸的画面 30fps
            就是 3×10×30=900ms/秒，CPU 直接打满。跟踪命中后不再重复识别 = 省 90% 计算。
        """
        # 一站式调用 InsightFace：检测 + 5 关键点对齐 + 512 维向量编码
        # detect_and_encode 是 M1 Task 0 加的方法，返回 list[DetectedFace]
        # 每个 DetectedFace 有 .bbox（int4 元组）、.encoding（FaceEncoding，已 L2 归一化）
        faces = self.pipeline.detect_and_encode(frame)

        # 没人就早返回，避免后续无意义的循环
        if not faces:
            # 注意：还是要让 tracker 走一遍 update（喂空 list），
            # 否则原先存在的 track 不会增加 missing_frames，永远不被清理
            return self.tracker.update([])

        # 把 DetectedFace.bbox 拍扁成 tracker 要的 list[BBox]
        # bbox 是 (x1, y1, x2, y2) int 元组，与 IoUTracker 的 BBox 类型一致
        bboxes = [face.bbox for face in faces]

        # tracker 一次性返回所有更新后的 tracks（含已存在和新建的）
        tracks = self.tracker.update(bboxes)

        # bbox → face 的反查表（根据 IoU 关联，但这里偷懒用顺序：
        # tracker.update 的实现保证返回顺序与传入 bboxes 一一对应——若实现变了这里会出 bug）
        # 更稳妥的做法：再算一次 IoU 关联。但 35 人小项目下不必过度防御。
        bbox_to_face = dict(zip(bboxes, faces))

        for track in tracks:
            if not track.needs_recognition:
                continue  # 已识别过的 track 跳过

            face = bbox_to_face.get(track.bbox)
            if face is None:
                continue  # 防御：理论上不会发生

            # 查询模板矩阵：返回 (best_pid, max_similarity)
            # face.encoding.vector 是已 L2 归一化的 (512,) float32
            person_id, similarity = self.template_matrix.query(face.encoding.vector)

            # 阈值判定：低于阈值 → 标为未知（identity=None）但仍记录 similarity 便于调试
            if similarity < self.threshold:
                self.tracker.set_identity(track.track_id, None, similarity)
            else:
                self.tracker.set_identity(track.track_id, person_id, similarity)

        # 重新拿一次 tracks（identity 已被 set_identity 写回）
        # 复用 tracker._tracks 的当前快照，避免再算一遍 IoU
        return list(self.tracker._tracks.values())
```

- [ ] **Step 4: 跑测试确认通过**

```bash
uv run pytest tests/unit/application/test_recognize_frame.py -v
```

预期：4 passed

- [ ] **Step 5: commit**

```bash
git add src/face_recognition/application/recognize_frame.py tests/unit/application/test_recognize_frame.py
git commit -m "feat(app): 加 RecognizeFrame 用例（实时多脸识别编排）"
```

---

### Task 6: FastAPI 应用骨架（lifespan + StaticFiles + 依赖装配）

**Files:**
- Create: `src/face_recognition/api/server.py`
- Modify: `src/face_recognition/api/dependencies.py`（增加实时识别相关依赖）
- Test: `tests/integration/api/test_server_smoke.py`

**职责**：起一个 FastAPI 应用，挂上 StaticFiles（前端单文件 HTML），配置 lifespan（启动时加载 pipeline + 模板矩阵，关闭时释放摄像头）。这一步只搭骨架，REST 和 WebSocket 路由在 Task 7-11 添加。

**关键概念解释**：

- **FastAPI**：Python 3 的现代 Web 框架，基于 Starlette + Pydantic。自带 OpenAPI 文档、依赖注入、异步支持。比 Flask 更适合需要 WebSocket / 类型校验的项目。
- **lifespan**（应用生命周期）：FastAPI 0.93+ 推荐的启动/关闭钩子。比老的 `@app.on_event("startup")` 更类型友好。它是一个 async generator：`yield` 之前是启动逻辑，之后是关闭逻辑。
- **StaticFiles**：把磁盘目录挂成 HTTP 静态资源。这里把前端 HTML 挂在 `/` 根路径下。
- **依赖注入（FastAPI Depends）**：`Depends(get_xxx)` 让路由函数自动拿到对象实例，避免全局变量。

- [ ] **Step 1: 写冒烟测试**

```python
# tests/integration/api/test_server_smoke.py
"""冒烟测试：app 能起来 + 静态首页能访问。

不测具体业务逻辑——那是后续 task 的事。
这里只验证 FastAPI 应用能正确装配（lifespan 不抛错、路由注册成功）。
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """TestClient 是 FastAPI 自带的测试客户端，基于 httpx，
    会自动触发 lifespan startup/shutdown，模拟真实启动流程。"""
    from face_recognition.api.server import app
    with TestClient(app) as c:
        yield c


def test_app_starts(client):
    """app 启动 + 关闭都不抛错 = 装配链路 OK。"""
    # TestClient 进入 with 块时已经跑过 lifespan startup，没炸就算过
    assert client.app is not None


def test_index_html_served(client):
    """根路径 / 应该返回 index.html。"""
    response = client.get("/")
    assert response.status_code == 200
    # StaticFiles 默认返回 text/html
    assert "text/html" in response.headers["content-type"]
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/integration/api/test_server_smoke.py -v
```

预期：FAIL（ImportError 或 app 不存在）

- [ ] **Step 3: 改 dependencies.py，加实时识别相关装配**

```python
# src/face_recognition/api/dependencies.py
"""依赖装配中心。

CLI 和 Server 共享同一份装配逻辑——这是清洁架构里"装配点"的体现：
具体类（infrastructure 层）在这里注入到用例（application 层），
其他模块完全不知道"用的是哪个 Repository / Pipeline 实现"。

M4 在这里**追加** Server 用的 lru_cache 单例 + 实时识别用例工厂。
M1 的 build_pipeline / build_repository / build_strategy / build_register_use_case /
build_recognize_use_case 保留给 CLI 用——CLI 是"短命进程"每次新建即可，无需缓存。

为什么用模块级单例（@lru_cache）：
    - InsightFace 模型加载 ~5s，每个 HTTP 请求重加载会卡死
    - 模型本身是无状态的，多线程共享读没问题（buffalo_l 已实测）
"""
from functools import lru_cache

from face_recognition.infrastructure.config_loader import AppConfig, load_config
from face_recognition.infrastructure.insightface_pipeline import InsightFacePipeline
from face_recognition.infrastructure.sqlite_repository import SqliteRepository
from face_recognition.application.template_matrix import TemplateMatrixService
from face_recognition.infrastructure.iou_tracker import IoUTracker
from face_recognition.application.recognize_frame import RecognizeFrame
from face_recognition.domain.interfaces import FacePipeline, PersonRepository

# M1 已有的工厂函数：保留不动（CLI 仍然用）
# build_pipeline, build_repository, build_strategy,
# build_register_use_case, build_recognize_use_case
# ↓ 以下是 M4 新增的 Server 单例


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """加载 config.yaml 一次，后续调用直接返回缓存。

    @lru_cache(maxsize=1) 是 Python 标准库的"无参函数级单例"惯用法。
    """
    from pathlib import Path
    return load_config(Path("config.yaml"))


@lru_cache(maxsize=1)
def get_pipeline() -> FacePipeline:
    """加载 InsightFace 模型一次。@lru_cache 保证全应用单例。

    构造参数对齐 M1：model_pack（不是 model_name）+ ctx_id（GPU/CPU 选择）+ det_size。
    """
    cfg = get_config()
    return InsightFacePipeline(
        model_pack=cfg.model.pack,
        ctx_id=cfg.model.ctx_id,
        det_size=cfg.model.det_size,
    )


@lru_cache(maxsize=1)
def get_repository() -> PersonRepository:
    """SQLite 仓储单例。

    M1 的 SqliteRepository 接受 db_path: Path | str，从 cfg.data.sqlite_path 取。
    """
    cfg = get_config()
    return SqliteRepository(cfg.data.sqlite_path)


@lru_cache(maxsize=1)
def get_template_matrix() -> TemplateMatrixService:
    """模板矩阵服务单例。注意：构造时只是"持有 repo 引用"，
    真正加载矩阵在 lifespan 启动里调 .load()。"""
    return TemplateMatrixService(repository=get_repository())


def build_recognize_frame_use_case() -> RecognizeFrame:
    """每次启动一个识别会话时调一次（不是每帧）。

    返回新 IoUTracker（每个会话独立的跟踪状态），但共享 pipeline + matrix（无状态）。

    为什么 tracker 不能 lru_cache：
        多个客户端同时连 WebSocket 会用同一个 tracker，状态串号。
        每个连接独立 tracker 才正确。
    """
    cfg = get_config()
    return RecognizeFrame(
        pipeline=get_pipeline(),
        tracker=IoUTracker(
            iou_threshold=cfg.realtime.iou_threshold,
            max_missing_frames=cfg.realtime.track_max_missing_frames,
        ),
        template_matrix=get_template_matrix(),
        threshold=cfg.recognition.threshold,
    )
```

> 注：M1 的 `config.yaml` 已含 `recognition.threshold` + `realtime.{iou_threshold, track_max_missing_frames, detect_every_n_frames, recognize_on_new_track}` + `camera.{device_index, resolution, fps}`。**M4 需要新增** `realtime.jpeg_quality: int = 80`（用于 frame_renderer）——同步在 `RealtimeConfig` BaseModel 加 `jpeg_quality: int = Field(default=80, ge=1, le=100)`，并在 config.yaml 的 realtime 段加一行。

- [ ] **Step 4: 写 server.py 骨架**

```python
# src/face_recognition/api/server.py
"""FastAPI 应用入口。

启动方式：
    uv run uvicorn face_recognition.api.server:app --host 0.0.0.0 --port 8000

uvicorn 是 ASGI 服务器（异步版的 wsgi），FastAPI 必须用 ASGI 服务器（不能用 gunicorn 默认配置）。
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from face_recognition.api.dependencies import (
    get_config,
    get_pipeline,
    get_template_matrix,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期钩子。

    @asynccontextmanager 是 Python 3.7+ 的装饰器，
    把一个 async generator 转换为 async with 可用的上下文管理器。

    yield 之前 = 启动；yield 之后 = 关闭。
    """
    # ----- 启动逻辑 -----
    logger.info("正在加载配置...")
    cfg = get_config()
    logger.info("配置加载完成 (db=%s)", cfg.data.sqlite_path)

    logger.info("正在加载 InsightFace 模型（首次启动 ~5s）...")
    get_pipeline()  # 触发 lru_cache，提前加载
    logger.info("模型加载完成")

    logger.info("正在加载模板矩阵...")
    matrix = get_template_matrix()
    matrix.load()
    logger.info("模板矩阵加载完成")

    yield  # ↑ 启动 / ↓ 关闭

    # ----- 关闭逻辑 -----
    logger.info("应用关闭中...")
    # 没有必须释放的全局资源——SQLite 连接由 repository 自管
    # 摄像头由 WebSocket 连接的 finally 释放


# 创建 FastAPI 应用
# title / version 会出现在 /docs（自动生成的 OpenAPI 文档）
app = FastAPI(
    title="人脸识别系统",
    version="0.1.0",
    lifespan=lifespan,
)


# 挂载静态文件：把 src/face_recognition/api/static/ 挂到 /
# html=True 让 / 自动返回 index.html（无需写 @app.get("/") 路由）
# 注意 mount 必须在所有 @app.<method> 路由之后调用——否则会"吞掉"后注册的路由
# 所以这一行放在文件最末（Task 7-11 加完路由后再 mount）。
# 这里先写在最末作为占位：
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
```

> ⚠️ **重要顺序约束**：StaticFiles 的 `app.mount("/", ...)` 必须放在所有 `@app.get/post/...` 路由之后。后续 Task 7-12 会在 mount 之前插入路由。Task 6 暂时只挂 StaticFiles + lifespan，没有任何 API 路由。

- [ ] **Step 5: 准备一个最小占位的 index.html**

```bash
mkdir -p src/face_recognition/api/static
cat > src/face_recognition/api/static/index.html << 'HTML_EOF'
<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>人脸识别系统</title></head>
<body><h1>占位首页（M5 会替换为完整前端）</h1></body>
</html>
HTML_EOF
```

- [ ] **Step 6: 跑测试确认通过**

```bash
uv run pytest tests/integration/api/test_server_smoke.py -v
```

预期：2 passed（首次跑会加载 InsightFace，~5-10s）

- [ ] **Step 7: 手动启动验证**

```bash
uv run uvicorn face_recognition.api.server:app --port 8000
# 浏览器访问 http://localhost:8000/ 应该看到占位 H1
# 访问 http://localhost:8000/docs 应该看到 Swagger UI（暂时空的）
```

- [ ] **Step 8: commit**

```bash
git add src/face_recognition/api/server.py src/face_recognition/api/dependencies.py src/face_recognition/api/static/index.html
git commit -m "feat(api): FastAPI 骨架（lifespan + StaticFiles + 依赖装配）"
```

---

### Task 7: REST GET /api/persons（列出所有人）

**Files:**
- Create: `src/face_recognition/api/routes_persons.py`
- Modify: `src/face_recognition/api/server.py`（注册路由）
- Test: `tests/integration/api/test_routes_persons.py`

**职责**：返回库内所有人的元信息（ID、姓名、模板数量、注册时间）。前端"人员管理"页面用。

**Pydantic 解释**：FastAPI 的输入/输出格式定义都用 Pydantic 模型（继承 `BaseModel`）。它会自动：
- 校验请求体类型（错了返 422）
- 序列化响应为 JSON
- 生成 OpenAPI 文档里的 schema

- [ ] **Step 1: 写失败测试**

```python
# tests/integration/api/test_routes_persons.py
"""REST 路由集成测试：用真 SQLite + 假数据。"""
from datetime import datetime
import pytest
from fastapi.testclient import TestClient

from face_recognition.api.server import app
from face_recognition.api.dependencies import get_repository
from face_recognition.domain.entities import Person, Template, FaceEncoding
import numpy as np


@pytest.fixture
def client_with_data(tmp_path, monkeypatch):
    """构造一个临时 SQLite + 预置 2 个人。

    monkeypatch 是 pytest 内置的 fixture：
        在测试期间临时替换属性 / 环境变量，测试结束自动还原。
    这里我们替换 dependencies.get_repository 的 lru_cache，
    让它返回一个临时仓储而非默认的生产 db。
    """
    from face_recognition.infrastructure.sqlite_repository import SqliteRepository
    repo = SqliteRepository(tmp_path / "test.db")

    def _enc(seed: int) -> FaceEncoding:
        v = np.ones(512, dtype=np.float32) / np.sqrt(512)
        return FaceEncoding(vector=v, model_version="buffalo_l")

    # M1 的 Person 实体字段：person_id / display_name / templates
    # M1 的 Template：encoding / source / created_at（一条 Template = 一个向量）
    # 多向量策略（kmeans_k3 / all_vectors）= 多条 Template 而非"一条 Template 多向量"
    repo.add(Person(
        person_id="alice",
        display_name="Alice",
        templates=(Template(
            encoding=_enc(0),
            source="random_one",
            created_at=datetime(2026, 5, 21, 10, 0),
        ),),
    ))
    repo.add(Person(
        person_id="bob",
        display_name="Bob",
        templates=tuple(
            Template(
                encoding=_enc(i + 1),
                source=f"kmeans_centroid_{i}",
                created_at=datetime(2026, 5, 21, 11, 0),
            )
            for i in range(3)
        ),
    ))

    # FastAPI 依赖覆盖：app.dependency_overrides 是官方推荐的测试覆盖方式
    app.dependency_overrides[get_repository] = lambda: repo
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_list_persons_empty(tmp_path):
    from face_recognition.infrastructure.sqlite_repository import SqliteRepository
    repo = SqliteRepository(tmp_path / "empty.db")
    app.dependency_overrides[get_repository] = lambda: repo
    with TestClient(app) as c:
        response = c.get("/api/persons")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == []


def test_list_persons_returns_two(client_with_data):
    response = client_with_data.get("/api/persons")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    ids = {p["person_id"] for p in data}
    assert ids == {"alice", "bob"}
    # 校验每条 schema
    alice = next(p for p in data if p["person_id"] == "alice")
    assert alice["display_name"] == "Alice"
    assert alice["template_count"] == 1
    bob = next(p for p in data if p["person_id"] == "bob")
    assert bob["template_count"] == 3
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/integration/api/test_routes_persons.py -v
```

预期：FAIL（路由不存在 → 404）

- [ ] **Step 3: 写路由实现**

```python
# src/face_recognition/api/routes_persons.py
"""人员管理 REST 路由。

为什么用 APIRouter 而不是直接 @app.get：
    APIRouter 是 FastAPI 提供的"路由分组"机制，能：
        - 把同主题路由集中到一个文件
        - 设置共同前缀（这里全部以 /api/persons 开头）
        - 设置共同 tags（OpenAPI 文档分组）
"""
from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from face_recognition.api.dependencies import get_repository
from face_recognition.domain.interfaces import PersonRepository

router = APIRouter(prefix="/api/persons", tags=["persons"])


class PersonResponse(BaseModel):
    """对外暴露的人员摘要。

    为什么不直接返回 domain 的 Person 实体：
        - Person.templates 里有 512 维向量，不该塞进 HTTP 响应（太大、无意义）
        - domain 实体应该和 HTTP 协议解耦——这是清洁架构的核心思想

    BaseModel 是 Pydantic 提供的基类，FastAPI 会自动把它序列化为 JSON。

    字段名对齐 M1 domain：person_id / display_name。
    template_count 用 len(p.templates)（M1 的 Template 是单向量结构,多向量策略 → 多条 Template）。
    earliest_created_at 取自第一条 Template 的 created_at——M1 Person 没有"注册时间"字段。
    """
    person_id: str
    display_name: str
    template_count: int
    earliest_created_at: datetime | None = None


@router.get("", response_model=list[PersonResponse])
def list_persons(repository: PersonRepository = Depends(get_repository)) -> list[PersonResponse]:
    """列出库内所有人。

    response_model=list[PersonResponse] 让 FastAPI 知道：
        - 自动把返回值校验/序列化为这个 schema
        - 在 /docs 里展示这个 schema
    """
    persons = repository.list_all()
    return [
        PersonResponse(
            person_id=p.person_id,
            display_name=p.display_name,
            template_count=len(p.templates),
            earliest_created_at=min((t.created_at for t in p.templates), default=None),
        )
        for p in persons
    ]
```

- [ ] **Step 4: 在 server.py 注册路由（mount StaticFiles 之前）**

```python
# server.py 修改：在 app = FastAPI(...) 之后、app.mount(...) 之前加：
from face_recognition.api.routes_persons import router as persons_router

app.include_router(persons_router)

# 然后才是 app.mount("/", StaticFiles(...))
```

- [ ] **Step 5: 跑测试确认通过**

```bash
uv run pytest tests/integration/api/test_routes_persons.py -v
```

预期：2 passed

- [ ] **Step 6: commit**

```bash
git add src/face_recognition/api/routes_persons.py src/face_recognition/api/server.py tests/integration/api/test_routes_persons.py
git commit -m "feat(api): GET /api/persons 列出所有人"
```

---

### Task 8: REST POST /api/persons（注册新人，多图上传）

**Files:**
- Modify: `src/face_recognition/api/routes_persons.py`
- Test: 追加到 `tests/integration/api/test_routes_persons.py`

**职责**：接受表单（姓名、策略名、≥1 张图片），调用 M1 已有的 `RegisterFace` 用例。

**关键概念**：

- **multipart/form-data**：HTTP 上传文件的标准协议。FastAPI 用 `UploadFile` + `Form` 接收。
- **UploadFile**：FastAPI 提供的文件包装，有 `.filename`、`.content_type`、`.file`（SpooledTemporaryFile，小文件放内存大文件落盘）、`.read()`（async）。比直接 bytes 节省内存。
- **HTTPException**：FastAPI 的异常类，抛出后会被自动转成对应 HTTP 状态码 + JSON 错误体。

- [ ] **Step 1: 写失败测试**

```python
# 追加到 test_routes_persons.py
import io
import numpy as np
from PIL import Image


def _make_jpeg_bytes(seed: int) -> bytes:
    """造一张随机噪点 JPEG（不是真人脸——用于测试上传链路本身，
    真正的"识别成功"由 M1 集成测试覆盖）。

    PIL（Pillow）的 Image.save 接受 file-like 对象，
    BytesIO 是内存中的"假文件"，写完用 .getvalue() 拿 bytes。
    """
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_register_person_no_files_returns_400(client_with_data):
    """没传文件 → 422（Pydantic 校验失败）或 400。"""
    response = client_with_data.post(
        "/api/persons",
        data={"person_id": "charlie", "display_name": "Charlie", "strategy": "random_one"},
    )
    assert response.status_code in (400, 422)


def test_register_person_unknown_strategy_returns_400(client_with_data):
    """策略名拼错 → 400。"""
    files = [("images", ("a.jpg", _make_jpeg_bytes(1), "image/jpeg"))]
    response = client_with_data.post(
        "/api/persons",
        data={"person_id": "charlie", "display_name": "Charlie", "strategy": "totally_wrong"},
        files=files,
    )
    assert response.status_code == 400
    assert "strategy" in response.json()["detail"].lower()


# 注：真正的 happy path 测试需要 InsightFace 模型 + GPU，
# 放进 @pytest.mark.gpu 标记里，CI 上跳过，本地手动跑。
@pytest.mark.gpu
def test_register_person_happy_path(client_with_data):
    """真实流程冒烟：上传 1 张 InsightFace sample 图，注册成功。

    @pytest.mark.gpu 是自定义标记（在 pyproject.toml 的 pytest 配置里声明），
    `pytest -m gpu` 才跑这个测试，默认 `pytest` 跳过。
    """
    import insightface
    from pathlib import Path
    sample_dir = Path(insightface.__file__).parent / "data" / "images"
    sample_jpg = sample_dir / "t1.jpg"  # InsightFace 内置示例图，含人脸
    files = [("images", ("t1.jpg", sample_jpg.read_bytes(), "image/jpeg"))]
    response = client_with_data.post(
        "/api/persons",
        data={"person_id": "testuser", "display_name": "TestUser", "strategy": "random_one"},
        files=files,
    )
    assert response.status_code == 201
    body = response.json()
    assert body["person_id"] == "testuser"
    assert body["display_name"] == "TestUser"
    assert body["template_count"] >= 1
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/integration/api/test_routes_persons.py::test_register_person_no_files_returns_400 -v
```

预期：FAIL（路由不存在）

- [ ] **Step 3: 写路由实现**

```python
# 追加到 routes_persons.py
import logging
from typing import Annotated

from fastapi import File, Form, HTTPException, UploadFile
import numpy as np
import cv2

import cv2
from face_recognition.application.register_face import RegisterFace
from face_recognition.api.dependencies import (
    get_pipeline,
    get_template_matrix,
    get_config,
)
# build_strategy 是 M1 已有的工厂函数：name → 策略实例
from face_recognition.api.dependencies import build_strategy
from face_recognition.domain.errors import (
    NoFaceError,
    MultipleFacesError,
    PersonHasNoTemplatesError,
    DuplicatePersonError,
)

logger = logging.getLogger(__name__)

# M1 config_loader.StrategyName 已用 Literal 圈定白名单：random_one / mean_all /
# manual_three / kmeans_k3 / all_vectors。这里复用同一个常量集做 HTTP 层校验。
_VALID_STRATEGIES = {"random_one", "mean_all", "manual_three", "kmeans_k3", "all_vectors"}


@router.post("", response_model=PersonResponse, status_code=201)
async def register_person(
    person_id: Annotated[str, Form(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_\-]+$")],
    display_name: Annotated[str, Form(min_length=1, max_length=64)],
    strategy: Annotated[str, Form()],
    images: Annotated[list[UploadFile], File()],
    pipeline: FacePipeline = Depends(get_pipeline),
    repository: PersonRepository = Depends(get_repository),
    template_matrix: TemplateMatrixService = Depends(get_template_matrix),
    cfg: AppConfig = Depends(get_config),
) -> PersonResponse:
    """注册新人。

    Annotated[T, Form(...)] 是 FastAPI 推荐的"参数 + 校验元数据"写法，
    比老式 `name: str = Form(...)` 更清晰、类型工具更友好。

    multipart/form-data 同时含表单字段 + 文件 → 文本字段用 Form，images 用 File。

    person_id 必须是 [a-zA-Z0-9_-]+：因为它会作为 URL 路径参数（DELETE / templates 端点）。
    """
    # ----- 1. 参数校验 -----
    if strategy not in _VALID_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"未知 strategy: {strategy}; 可选: {sorted(_VALID_STRATEGIES)}",
        )
    if not images:
        raise HTTPException(status_code=400, detail="至少上传一张图片")

    # ----- 2. 把 UploadFile 解码为 np.ndarray（BGR）-----
    frames: list[np.ndarray] = []
    for upload in images:
        data = await upload.read()  # async 读全部 bytes
        # np.frombuffer + cv2.imdecode：从内存 bytes 解码图像。
        # frombuffer 不拷贝数据（视图）；imdecode 自动识别 JPEG/PNG，返回 BGR 通道顺序。
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(
                status_code=400,
                detail=f"图片解码失败: {upload.filename}",
            )
        frames.append(frame)

    # ----- 3. 构造用例并调用 -----
    # build_strategy 来自 M1 dependencies.py：name + seed → TemplateStrategy 实例
    strategy_obj = build_strategy(strategy, seed=cfg.evaluation.random_seed)
    # RegisterFace 的 image_loader 这里用不到（我们直接传内存帧给 register_from_frames）,
    # 用 lambda 占位即可。
    use_case = RegisterFace(
        pipeline=pipeline,
        repository=repository,
        strategy=strategy_obj,
        image_loader=lambda _p: (_ for _ in ()).throw(NotImplementedError("HTTP 路径不走文件加载")),
    )
    try:
        # register_from_frames 是 M4 Task 0 给 M1 RegisterFace 加的方法：
        # 接受内存中的 frame 列表 + 显式 person_id/display_name，返回 Person
        person = use_case.register_from_frames(
            person_id=person_id,
            display_name=display_name,
            frames=frames,
        )
    except (NoFaceError, MultipleFacesError, PersonHasNoTemplatesError) as e:
        # 这几个 domain 异常都会被 Task 12 的全局 handler 捕获并转为 4xx，
        # 这里其实可以不写 try/except——但显式 catch 让本端点意图更清楚。
        raise HTTPException(status_code=400, detail=str(e))
    except DuplicatePersonError as e:
        raise HTTPException(status_code=409, detail=str(e))

    # ----- 4. 触发模板矩阵 reload，让新人立即生效 -----
    # 不 reload 的话，正在跑的 WebSocket 识别线程不会知道新人加了
    template_matrix.reload()
    logger.info("注册成功: person_id=%s display_name=%s strategy=%s",
                person.person_id, person.display_name, strategy)

    # ----- 5. 返回响应 -----
    return PersonResponse(
        person_id=person.person_id,
        display_name=person.display_name,
        template_count=len(person.templates),
        earliest_created_at=min((t.created_at for t in person.templates), default=None),
    )
```

- [ ] **Step 4: 跑测试确认通过（非 GPU 部分）**

```bash
uv run pytest tests/integration/api/test_routes_persons.py -v -m "not gpu"
```

预期：4 passed（list 2 个 + 校验 2 个），1 个 skipped（GPU 标记）

- [ ] **Step 5: 手动 GPU 冒烟测试**

```bash
uv run pytest tests/integration/api/test_routes_persons.py::test_register_person_happy_path -v -m gpu
```

预期：1 passed

- [ ] **Step 6: commit**

```bash
git add src/face_recognition/api/routes_persons.py tests/integration/api/test_routes_persons.py
git commit -m "feat(api): POST /api/persons 注册新人（multipart 上传）"
```

---

### Task 9: REST DELETE /api/persons/{person_id}（删除人员）

**Files:**
- Modify: `src/face_recognition/api/routes_persons.py`
- Test: 追加到 `tests/integration/api/test_routes_persons.py`

- [ ] **Step 1: 写失败测试**

```python
def test_delete_person_success(client_with_data):
    response = client_with_data.delete("/api/persons/alice")
    assert response.status_code == 204  # 204 No Content：删除成功的标准状态码

    # 再 list 应该只剩 1 个
    listed = client_with_data.get("/api/persons").json()
    assert len(listed) == 1
    assert listed[0]["person_id"] == "bob"


def test_delete_person_not_found(client_with_data):
    response = client_with_data.delete("/api/persons/nonexistent")
    assert response.status_code == 404
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/integration/api/test_routes_persons.py::test_delete_person_success -v
```

预期：FAIL（路由不存在）

- [ ] **Step 3: 写实现**

```python
# 追加到 routes_persons.py
from fastapi import status


@router.delete("/{person_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_person(
    person_id: str,
    repository: PersonRepository = Depends(get_repository),
    template_matrix: TemplateMatrixService = Depends(get_template_matrix),
) -> None:
    """删除指定 ID 的人。

    路径参数 person_id 通过 URL 路由模板捕获。FastAPI 会自动当字符串处理。

    返回 204 No Content：HTTP 标准约定"删除成功无响应体"。

    M1 的 PersonRepository Protocol 提供 get(person_id) -> Person | None 和
    remove(person_id) -> None（不是 get_by_id / delete）。
    """
    if repository.get(person_id) is None:
        raise HTTPException(status_code=404, detail=f"人员不存在: {person_id}")

    repository.remove(person_id)
    template_matrix.reload()  # 立即生效
```

- [ ] **Step 4: 跑测试确认通过**

```bash
uv run pytest tests/integration/api/test_routes_persons.py -v -m "not gpu"
```

预期：6 passed

- [ ] **Step 5: commit**

```bash
git add src/face_recognition/api/routes_persons.py tests/integration/api/test_routes_persons.py
git commit -m "feat(api): DELETE /api/persons/{id} 删除人员"
```

---

### Task 10: REST GET /api/persons/{person_id}/templates（查看模板元信息）

**Files:**
- Modify: `src/face_recognition/api/routes_persons.py`
- Test: 追加到 `tests/integration/api/test_routes_persons.py`

**职责**：返回某个人的模板信息（策略、向量数量），不返回向量本身（向量是 512×float32 数组，对前端无意义）。

- [ ] **Step 1: 写失败测试**

```python
def test_get_templates_success(client_with_data):
    response = client_with_data.get("/api/persons/alice/templates")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    # M1 Template 是单向量结构,source 是其字符串标识
    assert data[0]["source"] == "random_one"


def test_get_templates_for_kmeans(client_with_data):
    """bob 是 kmeans_k3 → 3 条 Template,source 各为 kmeans_centroid_0/1/2。"""
    data = client_with_data.get("/api/persons/bob/templates").json()
    assert len(data) == 3
    sources = {t["source"] for t in data}
    assert sources == {"kmeans_centroid_0", "kmeans_centroid_1", "kmeans_centroid_2"}


def test_get_templates_not_found(client_with_data):
    response = client_with_data.get("/api/persons/ghost/templates")
    assert response.status_code == 404
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/integration/api/test_routes_persons.py::test_get_templates_success -v
```

预期：FAIL（404，路由不存在）

- [ ] **Step 3: 写实现**

```python
# 追加到 routes_persons.py
class TemplateResponse(BaseModel):
    """模板摘要——不含向量本身，前端用不到。

    对齐 M1 的 Template 实体（encoding/source/created_at），不暴露 encoding 内容。
    """
    source: str
    created_at: datetime


@router.get("/{person_id}/templates", response_model=list[TemplateResponse])
def get_templates(
    person_id: str,
    repository: PersonRepository = Depends(get_repository),
) -> list[TemplateResponse]:
    person = repository.get(person_id)
    if person is None:
        raise HTTPException(status_code=404, detail=f"人员不存在: {person_id}")
    return [
        TemplateResponse(source=t.source, created_at=t.created_at)
        for t in person.templates
    ]
```

- [ ] **Step 4: 跑测试确认通过**

```bash
uv run pytest tests/integration/api/test_routes_persons.py -v -m "not gpu"
```

预期：9 passed

- [ ] **Step 5: commit**

```bash
git add src/face_recognition/api/routes_persons.py tests/integration/api/test_routes_persons.py
git commit -m "feat(api): GET /api/persons/{id}/templates 查看模板"
```

---

### Task 11: WebSocket /ws/stream（识别线程 + JPEG 二进制 + JSON 元数据）

**Files:**
- Create: `src/face_recognition/api/routes_stream.py`
- Modify: `src/face_recognition/api/server.py`（注册路由）
- Test: `tests/integration/api/test_routes_stream.py`

**职责**：客户端连上 `/ws/stream` 后，服务端启动摄像头 + 识别线程，把每一帧的两条消息推给前端：
1. **二进制消息**：JPEG 编码的渲染帧（已画框 + 写名）
2. **文本消息（JSON）**：本帧的 tracks 元数据（track_id / identity / similarity / bbox）

**为什么"两条消息"而不是"一条消息含图 + 元数据"**：
- WebSocket 协议原生支持二进制 + 文本两种帧类型
- 浏览器 `WebSocket` API 区分 `Blob` 和 `string`，前端写起来直接 `if (typeof event.data === 'string')` 分支
- 单条消息同时塞图 + JSON 需要 base64 编码 → 体积膨胀 33% + 解析麻烦

**关键概念**：

- **WebSocket**：在单条 TCP 连接上做"全双工"通信的协议。HTTP 单向请求/响应，WebSocket 双向、长连接。
- **FastAPI WebSocket**：用 `@app.websocket("/path")` 装饰器，路由函数收 `WebSocket` 对象，async 调用 `accept()`、`send_bytes()`、`send_text()`、`receive_text()`、`close()`。
- **threading.Thread**：Python 标准库，CPU 密集型用 multiprocessing，I/O 密集型用 threading。这里"摄像头读帧 + 识别"是混合密集，但 GIL 在 numpy / OpenCV / InsightFace 内部会释放（C 扩展），threading 完全够用。
- **queue.Queue**：线程安全的 FIFO 队列。识别线程把"已识别帧 + tracks"丢进队列，async 主任务从队列取出来推送 WebSocket。
- **threading.Event**：跨线程的布尔信号（"该停了吗？"）。WebSocket 断开时 set，识别线程 while not event.is_set 退出。

**架构**：

```
┌─────────────────┐       ┌──────────────────┐       ┌──────────────┐
│ Capture+Recog   │ queue │ async ws main    │ ws    │ Browser      │
│ Thread          │ ─────>│ task             │ ────> │ <video>      │
│ (1 个，背景跑)  │       │ (asyncio)        │       │ + Canvas     │
└─────────────────┘       └──────────────────┘       └──────────────┘
   ↑ shared:
   - CameraCapture
   - RecognizeFrame use case
   - frame_renderer
```

> 简化决策：spec 里说"1 capture 线程 + 1 recognition 线程"，本实现把它们合并为一个线程（capture.read → process_frame → encode_jpeg → put queue）。在 35 人小项目下足够，能避免线程间帧数据传递的额外队列。如果未来要解耦再拆。

- [ ] **Step 1: 写测试**

WebSocket 真实测试需要真摄像头，难自动化。这里写一个"路由能 accept 连接"的最小冒烟测试，真正端到端测试在 Task 13。

```python
# tests/integration/api/test_routes_stream.py
"""WebSocket 路由冒烟测试。

完整流程（识别 + 推流）由 Task 13 端到端测试 + 手动浏览器验证覆盖。
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

from face_recognition.api.server import app


def test_ws_accepts_connection(monkeypatch):
    """确认 WebSocket 路由能接受连接。

    用 monkeypatch mock CameraCapture 避免真开摄像头：
    测试环境通常没有摄像头权限，必须 mock。
    """
    fake_cap = MagicMock()
    fake_cap.read.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    fake_cap.release.return_value = None

    monkeypatch.setattr(
        "face_recognition.api.routes_stream.CameraCapture",
        lambda *a, **kw: fake_cap,
    )

    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as ws:
            # 收一条二进制 + 一条文本就算路由通了
            data = ws.receive_bytes()
            assert isinstance(data, bytes)
            assert len(data) > 0  # JPEG 不为空
            meta = ws.receive_json()
            assert "tracks" in meta
            assert "frame_id" in meta
```

- [ ] **Step 2: 写实现**

```python
# src/face_recognition/api/routes_stream.py
"""WebSocket 实时识别推流。

线程模型：
    每个 WebSocket 连接独占：
        - 1 个 CameraCapture（独占摄像头——同一时刻只能 1 个连接）
        - 1 个识别线程（capture.read → process_frame → renderer → queue）
        - 1 个 async 主任务（从 queue 取帧 → ws.send_bytes/send_text）
    断开时 stop_event.set，线程自杀，capture.release。

为什么用线程而不是 asyncio：
    cv2 + InsightFace 都是同步阻塞 API，强行用 asyncio 要包 run_in_executor 反而更乱。
    用一个独立线程同步循环最简单。FastAPI 主进程仍然是 asyncio，两者通过 queue.Queue 通信。
"""
import asyncio
import json
import logging
import queue
import threading
from dataclasses import asdict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from face_recognition.api.dependencies import (
    build_recognize_frame_use_case,
    get_config,
)
from face_recognition.infrastructure.camera_capture import CameraCapture, CameraDisconnectedError
from face_recognition.infrastructure.frame_renderer import render_tracks, encode_jpeg

logger = logging.getLogger(__name__)
router = APIRouter()


def _recognition_loop(
    capture: CameraCapture,
    use_case,
    out_queue: queue.Queue,
    stop_event: threading.Event,
    jpeg_quality: int,
) -> None:
    """识别线程主循环。

    每帧产出：(jpeg_bytes, metadata_dict)
    丢进 queue 给 async 任务；满了就丢老的（防止前端慢导致延迟堆积）。
    """
    frame_id = 0
    while not stop_event.is_set():
        try:
            frame = capture.read()
        except CameraDisconnectedError as e:
            logger.warning("摄像头断开: %s", e)
            # 把错误塞进 queue 让 ws 任务知道
            out_queue.put(("error", str(e)))
            break

        tracks = use_case.process_frame(frame)
        rendered = render_tracks(frame, tracks)
        jpeg = encode_jpeg(rendered, quality=jpeg_quality)

        meta = {
            "frame_id": frame_id,
            "tracks": [
                {
                    "track_id": t.track_id,
                    "identity": t.identity,
                    "similarity": round(t.similarity, 4),
                    "bbox": list(t.bbox),
                }
                for t in tracks
            ],
        }
        frame_id += 1

        # 队列满 = 前端跟不上 → 丢弃当前帧（保实时不保完整）
        try:
            out_queue.put_nowait(("frame", (jpeg, meta)))
        except queue.Full:
            try:
                out_queue.get_nowait()  # 丢一个最老的
            except queue.Empty:
                pass
            try:
                out_queue.put_nowait(("frame", (jpeg, meta)))
            except queue.Full:
                pass  # 还是放不下就放弃这帧

    capture.release()
    logger.info("识别线程退出")


@router.websocket("/ws/stream")
async def stream_recognition(websocket: WebSocket) -> None:
    """实时识别 WebSocket。

    协议：服务端单向推送
        - 二进制帧：JPEG 编码的渲染图
        - 文本帧：JSON {"frame_id": int, "tracks": [...]}
        每帧两条消息成对发送：先二进制后文本。

    客户端：连上即开始接收，断开即停止。
    """
    await websocket.accept()
    cfg = get_config()

    capture = None
    stop_event = threading.Event()
    thread = None
    try:
        capture = CameraCapture(
            device_index=cfg.camera.device_index,
            resolution=tuple(cfg.camera.resolution),
        )
        use_case = build_recognize_frame_use_case()

        # maxsize=2：缓冲两帧足够；过大会引入延迟（用户看到的是"过去的画面"）
        out_queue: queue.Queue = queue.Queue(maxsize=2)

        thread = threading.Thread(
            target=_recognition_loop,
            args=(capture, use_case, out_queue, stop_event, cfg.realtime.jpeg_quality),
            daemon=True,  # 主进程退出时线程自动清理
        )
        thread.start()

        loop = asyncio.get_event_loop()
        while True:
            # queue.get 是阻塞调用 → 在 executor 里跑避免阻塞 asyncio
            # run_in_executor(None, ...) = 用默认线程池跑同步函数
            kind, payload = await loop.run_in_executor(None, out_queue.get)
            if kind == "error":
                await websocket.send_text(json.dumps({"error": payload}))
                break
            jpeg, meta = payload
            await websocket.send_bytes(jpeg)
            await websocket.send_text(json.dumps(meta))

    except WebSocketDisconnect:
        logger.info("WebSocket 客户端断开")
    except Exception:
        logger.exception("WebSocket 推流异常")
    finally:
        stop_event.set()
        if thread is not None:
            thread.join(timeout=2.0)
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
```

- [ ] **Step 3: 在 server.py 注册路由（mount StaticFiles 之前）**

```python
# server.py 修改：
from face_recognition.api.routes_stream import router as stream_router

app.include_router(stream_router)
# 然后才是 app.mount(...)
```

- [ ] **Step 4: 跑冒烟测试**

```bash
uv run pytest tests/integration/api/test_routes_stream.py -v
```

预期：1 passed

- [ ] **Step 5: 手动验证**

```bash
uv run uvicorn face_recognition.api.server:app --port 8000
# 浏览器开发者工具 Console 跑：
#   const ws = new WebSocket("ws://localhost:8000/ws/stream");
#   ws.binaryType = "blob";
#   ws.onmessage = e => console.log(typeof e.data, e.data);
# 应该看到交替的 Blob / string 消息流
```

- [ ] **Step 6: commit**

```bash
git add src/face_recognition/api/routes_stream.py src/face_recognition/api/server.py tests/integration/api/test_routes_stream.py
git commit -m "feat(api): WebSocket /ws/stream 实时识别推流"
```

---

### Task 12: 全局异常处理（domain 异常 → HTTP 状态码）

**Files:**
- Modify: `src/face_recognition/api/server.py`
- Test: 新增 `tests/integration/api/test_error_handling.py`

**职责**：把 domain 层的异常（`NoFaceError`、`MultipleFacesError`、`PersonNotFoundError` 等）映射到对应 HTTP 状态码 + 友好 JSON 错误体。避免在每个路由里 try/except。

**为什么用全局异常处理**：
- DRY：同一类异常处理写一次
- 路由代码更干净，专注业务编排
- FastAPI 的 `@app.exception_handler` 装饰器原生支持

- [ ] **Step 1: 写失败测试**

```python
# tests/integration/api/test_error_handling.py
"""验证 domain 异常被正确转换为 HTTP 错误响应。"""
from fastapi.testclient import TestClient

from face_recognition.api.server import app
from face_recognition.domain.errors import NoFaceError


def test_no_face_error_returns_400(monkeypatch):
    """当用例抛 NoFaceError 时,应返回 400。

    注：这里我们走"假装路由内部抛异常"的方式来验证 handler 注册了。
    实际触发链路在 Task 8 的 register endpoint 里。
    """
    @app.get("/__test__/no_face")
    def _raise():
        raise NoFaceError("没检测到人脸")

    with TestClient(app) as client:
        response = client.get("/__test__/no_face")

    assert response.status_code == 400
    assert response.json()["detail"] == "没检测到人脸"
```

> 注：上面 `@app.get("/__test__/no_face")` 是测试时动态加的——这种"测试夹具路由"略 hack 但简单。也可以在 `routes_persons.py` 里删掉局部 `try/except NoFaceError` 后用真路由验证，看个人偏好。

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/integration/api/test_error_handling.py -v
```

预期：FAIL（默认 FastAPI 把未捕获异常转为 500）

- [ ] **Step 3: 写实现**

```python
# server.py 增加：
from fastapi import Request
from fastapi.responses import JSONResponse

from face_recognition.domain.errors import (
    FaceRecognitionError,
    NoFaceError,
    MultipleFacesError,
    PersonNotFoundError,
)


# 异常类 → HTTP 状态码的映射
# 每个 domain 异常都对应一个明确的"客户端错误"状态码
_ERROR_STATUS_MAP = {
    NoFaceError: 400,
    MultipleFacesError: 400,
    PersonNotFoundError: 404,
}


@app.exception_handler(FaceRecognitionError)
async def domain_error_handler(request: Request, exc: FaceRecognitionError) -> JSONResponse:
    """所有 domain 异常的统一处理。

    @app.exception_handler(ExceptionClass) 是 FastAPI 注册自定义异常 handler 的方式;
    路由里 raise 这个异常类型(或子类)就会自动调到这里。
    """
    status_code = _ERROR_STATUS_MAP.get(type(exc), 500)
    logger.warning("domain 异常: %s (path=%s)", exc, request.url.path)
    return JSONResponse(
        status_code=status_code,
        content={"detail": str(exc)},
    )
```

> 这之后 Task 8 的 `routes_persons.py` 里的 `try/except NoFaceError` 可以删掉了——交给全局 handler。但**现在不要急着删**：M1 的实施情况未知，先确认 M1 的 errors.py 真的有 `FaceRecognitionError` 基类再删。

- [ ] **Step 4: 跑测试确认通过**

```bash
uv run pytest tests/integration/api/test_error_handling.py -v
```

预期：1 passed

- [ ] **Step 5: commit**

```bash
git add src/face_recognition/api/server.py tests/integration/api/test_error_handling.py
git commit -m "feat(api): 全局 domain 异常 → HTTP 错误响应映射"
```

---

### Task 13: 端到端冒烟测试 + 启动文档

**Files:**
- Create: `tests/integration/api/test_e2e_smoke.py`
- Modify: `README.md`（增加"启动 Web 服务"段落）

**职责**：跑一次完整链路验证：启动 server → 注册一个人 → 列表能查到 → 删除 → 列表清空。WebSocket 推流由于需要真摄像头不在自动测试覆盖。

- [ ] **Step 1: 写端到端测试**

```python
# tests/integration/api/test_e2e_smoke.py
"""端到端冒烟测试：模拟用户在网页上完整走一遍 REST 流程。

需要 GPU + InsightFace 模型;打 @pytest.mark.gpu。
"""
from pathlib import Path
import pytest
import insightface
from fastapi.testclient import TestClient

from face_recognition.api.server import app
from face_recognition.api.dependencies import get_repository, get_template_matrix
from face_recognition.infrastructure.sqlite_repository import SqliteRepository


@pytest.mark.gpu
def test_register_list_delete_e2e(tmp_path):
    """完整生命周期:注册 → 列表 → 模板查询 → 删除 → 列表为空。"""
    repo = SqliteRepository(tmp_path / "e2e.db")
    app.dependency_overrides[get_repository] = lambda: repo

    sample_jpg = Path(insightface.__file__).parent / "data" / "images" / "t1.jpg"

    with TestClient(app) as client:
        # 1. 初始为空
        assert client.get("/api/persons").json() == []

        # 2. 注册
        with sample_jpg.open("rb") as f:
            response = client.post(
                "/api/persons",
                data={"person_id": "e2e", "display_name": "E2E", "strategy": "random_one"},
                files={"images": ("t1.jpg", f, "image/jpeg")},
            )
        assert response.status_code == 201
        person_id = response.json()["person_id"]
        assert person_id == "e2e"

        # 3. 列表有 1 个
        listed = client.get("/api/persons").json()
        assert len(listed) == 1

        # 4. 模板信息可查
        templates = client.get(f"/api/persons/{person_id}/templates").json()
        assert len(templates) == 1
        assert "source" in templates[0]
        assert "created_at" in templates[0]

        # 5. 删除
        assert client.delete(f"/api/persons/{person_id}").status_code == 204

        # 6. 再列表为空
        assert client.get("/api/persons").json() == []

    app.dependency_overrides.clear()
```

- [ ] **Step 2: 跑测试**

```bash
uv run pytest tests/integration/api/test_e2e_smoke.py -v -m gpu
```

预期：1 passed

- [ ] **Step 3: 在 README.md 加一段"M4 启动指南"**

```markdown
## 启动 Web 服务（M4）

确保 `data/face.db` 已有注册数据（或先用 CLI 注册）：

\`\`\`bash
uv run python -m face_recognition.api.cli register \
    --strategy kmeans_k3 --dataset data/private_dataset/
\`\`\`

启动 FastAPI:

\`\`\`bash
uv run uvicorn face_recognition.api.server:app --host 0.0.0.0 --port 8000
\`\`\`

打开浏览器访问 `http://localhost:8000/`（M5 会替换为完整前端，当前是占位 H1）。
开发者文档: `http://localhost:8000/docs`（自动生成的 OpenAPI 页面）。
```

- [ ] **Step 4: commit**

```bash
git add tests/integration/api/test_e2e_smoke.py README.md
git commit -m "test(api): M4 端到端冒烟测试 + README 启动说明"
```

- [ ] **Step 5: 打 M4 完成 tag**

```bash
git tag m4-realtime-web
git push origin main --tags
```

---

## M4 完成标准

- ✅ `iou_tracker.py`、`camera_capture.py`、`template_matrix.py`、`frame_renderer.py` 全部带单元测试
- ✅ `RecognizeFrame` 用例覆盖"按需识别"逻辑
- ✅ FastAPI 起得来，`/docs` 可访问
- ✅ REST CRUD 4 个端点（GET 列表、POST 注册、DELETE、GET 模板）全部有集成测试
- ✅ WebSocket `/ws/stream` 至少能接受连接 + 推一帧
- ✅ 端到端测试 `test_register_list_delete_e2e` 通过
- ✅ `git tag m4-realtime-web` 已打

下一步：M5 单文件前端（HTML + JS）连接 REST + WebSocket，画 Canvas 框 + 显示视频。
