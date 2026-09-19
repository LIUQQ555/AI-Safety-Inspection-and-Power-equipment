"""SQLite 持久化。

对应技术方案第二十三节「数据库」：保存检测时间、设备编号、图像、
传感器数据、检测结果、风险等级、处理结果。

设计：摘要字段独立成列（便于按风险等级、时间范围做 SQL 查询与统计），
完整报告以 JSON 存在 ``payload`` 列（便于前端一次性还原全部明细，
也避免为每种模态单独建表带来的 schema 迁移负担）。

使用标准库 ``sqlite3``，无需额外依赖。SQLite 在并发写入上限制较多，
本项目为单机部署、写入频率低（每次巡检一条），不构成瓶颈；
若后续需要多实例部署，可把本模块替换为 PostgreSQL —— 接口保持不变。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from backend.core.schemas import InspectionReport, to_jsonable

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    inspection_id   TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    device_type     TEXT,
    device_name     TEXT,
    location        TEXT,
    operator        TEXT,
    risk_score      REAL,
    risk_level      TEXT,
    risk_level_name TEXT,
    has_visible     INTEGER DEFAULT 0,
    has_infrared    INTEGER DEFAULT 0,
    has_timeseries  INTEGER DEFAULT 0,
    payload         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_inspections_created  ON inspections(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_inspections_level    ON inspections(risk_level);
"""


class Database:
    """巡检记录存储。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    # ------------------------------------------------------------------
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.path), timeout=15.0)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _init_schema(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    def save_report(self, report: InspectionReport) -> None:
        """写入或覆盖一条巡检记录。"""
        payload = json.dumps(to_jsonable(report), ensure_ascii=False)
        fusion = report.fusion

        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO inspections (
                    inspection_id, created_at, device_type, device_name, location, operator,
                    risk_score, risk_level, risk_level_name,
                    has_visible, has_infrared, has_timeseries, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.inspection_id,
                    report.created_at,
                    report.device_type,
                    report.device_name,
                    report.location,
                    report.operator,
                    float(fusion.risk_score),
                    fusion.risk_level.value,
                    fusion.risk_level_name,
                    int(report.visible is not None),
                    int(report.infrared is not None),
                    int(report.timeseries is not None),
                    payload,
                ),
            )
        logger.info("已保存巡检记录 %s（风险 %.1f / %s）",
                    report.inspection_id, fusion.risk_score, fusion.risk_level_name)

    # ------------------------------------------------------------------
    def get_report(self, inspection_id: str) -> Optional[Dict[str, Any]]:
        """按编号读取完整报告。"""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM inspections WHERE inspection_id = ?",
                (inspection_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["payload"])
        except json.JSONDecodeError as exc:
            logger.error("报告 JSON 解析失败 %s：%s", inspection_id, exc)
            return None

    def get_markdown(self, inspection_id: str) -> Optional[str]:
        report = self.get_report(inspection_id)
        return report.get("report_markdown") if report else None

    def list_inspections(
        self,
        limit: int = 50,
        offset: int = 0,
        risk_level: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """列出巡检记录摘要（不含完整 payload，减小响应体）。"""
        sql = """
            SELECT inspection_id, created_at, device_type, device_name, location, operator,
                   risk_score, risk_level, risk_level_name,
                   has_visible, has_infrared, has_timeseries
            FROM inspections
        """
        params: List[Any] = []
        if risk_level:
            sql += " WHERE risk_level = ?"
            params.append(risk_level)
        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([int(limit), int(offset)])

        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()

        return [
            {
                "inspection_id": row["inspection_id"],
                "created_at": row["created_at"],
                "device_type": row["device_type"],
                "device_name": row["device_name"],
                "location": row["location"],
                "operator": row["operator"],
                "risk_score": row["risk_score"],
                "risk_level": row["risk_level"],
                "risk_level_name": row["risk_level_name"],
                "modalities": [
                    name for name, flag in (
                        ("visible", row["has_visible"]),
                        ("infrared", row["has_infrared"]),
                        ("electrical", row["has_timeseries"]),
                    ) if flag
                ],
            }
            for row in rows
        ]

    def delete_report(self, inspection_id: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM inspections WHERE inspection_id = ?", (inspection_id,)
            )
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """风险统计，供前端仪表盘使用。"""
        with self._connect() as connection:
            total = connection.execute("SELECT COUNT(*) AS n FROM inspections").fetchone()["n"]

            by_level = {
                row["risk_level"]: row["n"]
                for row in connection.execute(
                    "SELECT risk_level, COUNT(*) AS n FROM inspections GROUP BY risk_level"
                ).fetchall()
            }

            row = connection.execute(
                "SELECT AVG(risk_score) AS avg_score, MAX(risk_score) AS max_score FROM inspections"
            ).fetchone()

            since = (datetime.now() - timedelta(days=7)).isoformat(timespec="seconds")
            recent = connection.execute(
                "SELECT COUNT(*) AS n FROM inspections WHERE created_at >= ?", (since,)
            ).fetchone()["n"]

            modality_rows = connection.execute(
                """SELECT SUM(has_visible) AS v, SUM(has_infrared) AS ir,
                          SUM(has_timeseries) AS ts FROM inspections"""
            ).fetchone()

        return {
            "total": int(total or 0),
            "last_7_days": int(recent or 0),
            "avg_risk_score": round(float(row["avg_score"]), 2) if row["avg_score"] is not None else 0.0,
            "max_risk_score": round(float(row["max_score"]), 2) if row["max_score"] is not None else 0.0,
            "by_level": {
                "normal": int(by_level.get("normal", 0)),
                "attention": int(by_level.get("attention", 0)),
                "abnormal": int(by_level.get("abnormal", 0)),
                "critical": int(by_level.get("critical", 0)),
            },
            "modalities": {
                "visible": int(modality_rows["v"] or 0),
                "infrared": int(modality_rows["ir"] or 0),
                "electrical": int(modality_rows["ts"] or 0),
            },
        }

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) AS n FROM inspections").fetchone()["n"])


def create_database(config) -> Database:
    return Database(config.resolve("app", "db_path"))


__all__ = ["Database", "create_database"]
