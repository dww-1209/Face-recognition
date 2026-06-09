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
        color = (0, 255, 0) if t.person_id is not None else (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness=2)

        # 标签文字：identity (similarity)，None 时写"未知"
        label = f"{t.person_id} ({t.similarity:.2f})" if t.person_id else "未知"
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
