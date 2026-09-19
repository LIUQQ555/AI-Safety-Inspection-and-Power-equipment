"""FastAPI 后端入口。

启动::

    .venv\\Scripts\\python -m uvicorn backend.main:app --reload --port 8000

然后浏览器打开 http://127.0.0.1:8000

接口一览::

    GET    /                         前端页面
    GET    /api/health               系统状态（各分支后端、模型、知识库）
    POST   /api/inspect              执行一次巡检（多模态文件上传）
    GET    /api/inspections          巡检历史列表
    GET    /api/inspections/{id}     单次巡检完整结果
    GET    /api/inspections/{id}/report   导出 Markdown 报告
    DELETE /api/inspections/{id}     删除记录
    GET    /api/stats                风险统计
    POST   /api/knowledge/rebuild    重建知识库索引
    GET    /api/providers            大模型提供方可用状态
"""

from __future__ import annotations

import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from backend.config import PROJECT_ROOT, get_config
from backend.core.schemas import to_jsonable
from backend.llm.factory import describe_providers
from backend.services.inspection_service import InspectionService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("powerai")

config = get_config()
FRONTEND_DIR = PROJECT_ROOT / "frontend"
UPLOAD_DIR = config.ensure_dir("app", "upload_dir")

ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ALLOWED_TABLE_SUFFIXES = {".csv", ".xlsx", ".xls", ".json"}

# 全局服务实例。检测器初始化（YOLO 权重加载）开销大，只做一次。
service: Optional[InspectionService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    logger.info("正在初始化巡检服务……")
    service = InspectionService(config)

    # 知识库索引在启动时构建。失败不影响服务启动——只是报告中「依据标准」为空。
    try:
        count = service.knowledge_base.build()
        if count:
            logger.info("知识库就绪，共 %d 个文本块", count)
        else:
            logger.warning(
                "知识库为空。请把电力标准/规程文档放入 %s 后调用 "
                "POST /api/knowledge/rebuild 重建索引。",
                service.knowledge_base.knowledge_dir,
            )
    except Exception as exc:
        logger.error("知识库初始化失败：%s", exc)

    health = service.health()
    logger.info("可见光检测后端：%s", health["visible_detector"]["backend"])
    logger.info("时序模型：%s", "已加载" if health["timeseries_model"]["loaded"] else "未训练（使用统计判据）")
    logger.info("大模型提供方：%s", health["llm"]["name"])
    logger.info("服务已就绪 → http://127.0.0.1:8000")

    yield

    logger.info("服务已停止")


app = FastAPI(
    title=config.app.name,
    version=str(config.app.version),
    description="基于多模态AI的电力设备智能安全检测与风险评估系统",
    lifespan=lifespan,
)

# 允许本地前端页面直接调用（开发时前端可能运行在其它端口）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_service() -> InspectionService:
    if service is None:  # pragma: no cover - lifespan 未执行时
        raise HTTPException(status_code=503, detail="服务尚未初始化完成")
    return service


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
async def _save_upload(upload: UploadFile, subdir: str = "") -> Path:
    """保存上传文件到 data/uploads，返回落盘路径。"""
    suffix = Path(upload.filename or "").suffix.lower()
    target_dir = UPLOAD_DIR / subdir if subdir else UPLOAD_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    # 用上传文件原名会导致中文/重名问题，这里保留原扩展名、生成唯一名，
    # 同时在响应中回传原始文件名以便追溯。
    import uuid
    target = target_dir / f"{uuid.uuid4().hex[:12]}{suffix}"
    with target.open("wb") as fh:
        shutil.copyfileobj(upload.file, fh)
    return target


def _validate_suffix(path: Path, allowed: set[str], kind: str) -> None:
    if path.suffix.lower() not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"{kind}格式不支持：{path.suffix}。支持：{sorted(allowed)}",
        )


# ---------------------------------------------------------------------------
# 接口
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health() -> Dict[str, Any]:
    """系统状态。"""
    svc = get_service()
    return {
        "status": "ok",
        "app": {"name": config.app.name, "version": str(config.app.version)},
        **svc.health(),
    }


@app.get("/api/providers")
async def providers() -> Dict[str, Any]:
    """可配置的大模型提供方及当前可用性。"""
    svc = get_service()
    return {
        "current": svc.provider.name,
        "available": describe_providers(config),
    }


@app.post("/api/inspect")
async def inspect(
    visible_image: Optional[UploadFile] = File(None, description="可见光巡检图像"),
    thermal_image: Optional[UploadFile] = File(None, description="红外热像图"),
    timeseries_csv: Optional[UploadFile] = File(None, description="电压/电流/温度时序 CSV"),
    device_name: str = Form(""),
    location: str = Form(""),
    operator: str = Form(""),
    three_phase_temps: str = Form("", description="三相温度实测值，逗号分隔，如 '62.1,64.3,71.8'"),
) -> JSONResponse:
    """执行一次多模态巡检。"""
    svc = get_service()

    if visible_image is None and thermal_image is None and timeseries_csv is None:
        raise HTTPException(status_code=400, detail="请至少上传一种数据")

    visible_path = thermal_path = csv_path = None

    if visible_image is not None and visible_image.filename:
        visible_path = await _save_upload(visible_image, "visible")
        _validate_suffix(visible_path, ALLOWED_IMAGE_SUFFIXES, "可见光图像")

    if thermal_image is not None and thermal_image.filename:
        thermal_path = await _save_upload(thermal_image, "infrared")
        _validate_suffix(thermal_path, ALLOWED_IMAGE_SUFFIXES, "红外图像")

    if timeseries_csv is not None and timeseries_csv.filename:
        csv_path = await _save_upload(timeseries_csv, "timeseries")
        _validate_suffix(csv_path, ALLOWED_TABLE_SUFFIXES, "时序数据")

    phases: Optional[List[float]] = None
    if three_phase_temps.strip():
        try:
            phases = [float(x) for x in three_phase_temps.replace("，", ",").split(",") if x.strip()]
            if len(phases) != 3:
                raise ValueError("需要 3 个数值")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"三相温度格式错误：{exc}") from exc

    try:
        report = svc.inspect(
            visible_image=visible_path,
            thermal_image=thermal_path,
            timeseries_csv=csv_path,
            device_name=device_name.strip(),
            location=location.strip(),
            operator=operator.strip(),
            three_phase_temps=phases,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("巡检执行失败")
        raise HTTPException(status_code=500, detail=f"巡检执行失败：{exc}") from exc

    return JSONResponse(content=to_jsonable(report))


@app.get("/api/inspections")
async def list_inspections(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    risk_level: Optional[str] = Query(None, description="normal|attention|abnormal|critical"),
) -> Dict[str, Any]:
    """巡检历史列表。"""
    svc = get_service()
    items = svc.database.list_inspections(limit=limit, offset=offset, risk_level=risk_level)
    return {"total": svc.database.count(), "count": len(items), "items": items}


@app.get("/api/inspections/{inspection_id}")
async def get_inspection(inspection_id: str) -> JSONResponse:
    """单次巡检的完整结果。"""
    svc = get_service()
    report = svc.database.get_report(inspection_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"巡检记录不存在：{inspection_id}")
    return JSONResponse(content=report)


@app.get("/api/inspections/{inspection_id}/report")
async def export_report(inspection_id: str) -> PlainTextResponse:
    """导出 Markdown 报告。"""
    svc = get_service()
    markdown = svc.database.get_markdown(inspection_id)
    if markdown is None:
        raise HTTPException(status_code=404, detail=f"巡检记录不存在：{inspection_id}")
    return PlainTextResponse(
        content=markdown,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{inspection_id}.md"'},
    )


@app.delete("/api/inspections/{inspection_id}")
async def delete_inspection(inspection_id: str) -> Dict[str, Any]:
    """删除巡检记录。"""
    svc = get_service()
    if not svc.database.delete_report(inspection_id):
        raise HTTPException(status_code=404, detail=f"巡检记录不存在：{inspection_id}")
    return {"deleted": inspection_id}


@app.get("/api/stats")
async def stats() -> Dict[str, Any]:
    """风险统计。"""
    svc = get_service()
    return svc.database.stats()


@app.post("/api/knowledge/rebuild")
async def rebuild_knowledge() -> Dict[str, Any]:
    """重建知识库索引（knowledge/ 目录内容变化后调用）。"""
    svc = get_service()
    try:
        count = svc.rebuild_knowledge_base(force=True)
    except Exception as exc:
        logger.exception("知识库重建失败")
        raise HTTPException(status_code=500, detail=f"知识库重建失败：{exc}") from exc
    return {"chunks": count, **svc.knowledge_base.stats()}


# ---------------------------------------------------------------------------
# 静态资源
# ---------------------------------------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# 报告插图（标注图）通过 /files 暴露
annotated_dir = config.resolve("app", "report_dir") / "annotated"
annotated_dir.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=str(annotated_dir)), name="files")


@app.get("/")
async def index() -> Any:
    """前端首页。"""
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse(
            status_code=404,
            content={"detail": "未找到前端页面 frontend/index.html。"
                               "可直接访问 /docs 使用交互式 API 文档。"},
        )
    return FileResponse(str(index_file))


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
