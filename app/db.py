"""SQLite 存储层。

版本化规则（核心）：
- 每次成功提交样本或目标修改（PUT/DELETE）都会令全局 epoch +1；
- epoch 变更会删除全部旧作业记录，并置位取消事件，使在途计算永久失效
  （即使把数据改回原值也不会恢复旧结果，必须重新计算）；
- 结果只在作业成功完成时随作业原子写入，不存在“半份结果”；
- 进程重启后所有未完成作业标记为 interrupted，不开放任何部分结果。

删组杰克刀运行（jk_runs）：
- 依附于某个成功校准作业，与其同属一个 epoch；
- epoch 变更时运行不删除：在途运行被标记为 invalidated（查询返回 409），
  已完成的运行成为历史报告，同样因 epoch 失配而返回 409，绝不提供过期估计；
- 进程重启后未完成的运行标记为 interrupted。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    epoch INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS samples (
    person_id TEXT PRIMARY KEY,
    region TEXT NOT NULL,
    age_group TEXT NOT NULL,
    d INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS targets (
    dimension TEXT NOT NULL,
    code TEXT NOT NULL,
    population INTEGER NOT NULL,
    PRIMARY KEY (dimension, code)
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    epoch INTEGER NOT NULL,
    status TEXT NOT NULL,            -- queued|running|succeeded|infeasible|failed|interrupted
    submitted_at TEXT NOT NULL,
    finished_at TEXT,
    input_snapshot TEXT NOT NULL,    -- 提交时刻完整输入快照（JSON）
    result TEXT,                     -- 仅终态写入（完整权重+依据，或冲突集，或错误）
    error TEXT
);
CREATE TABLE IF NOT EXISTS jk_runs (
    jk_id TEXT PRIMARY KEY,
    epoch INTEGER NOT NULL,
    status TEXT NOT NULL,            -- queued|running|succeeded|failed|invalidated|interrupted
    submitted_at TEXT NOT NULL,
    finished_at TEXT,
    source_job_id TEXT NOT NULL,     -- 源校准作业号
    request TEXT NOT NULL,           -- 快照、权重、用电量与分组（JSON，提交时刻固化）
    result TEXT,                     -- 仅成功终态写入（均值/各轮均值/标准误）
    failure TEXT,                    -- 仅失败终态写入（组号+冲突目标或数值原因）
    error TEXT
);
"""


class StaleEpochError(RuntimeError):
    """写入时全局 epoch 已变化：源作业在提交瞬间失效。"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._write_lock = threading.RLock()
        self._cancel = threading.Event()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        cur = self._conn.execute("SELECT epoch FROM meta WHERE id=1")
        row = cur.fetchone()
        if row is None:
            self._conn.execute("INSERT INTO meta(id, epoch) VALUES (1, 1)")
            self._conn.commit()
            self.epoch = 1
        else:
            self.epoch = int(row["epoch"])
        # 重启：未完成作业一律中断，不允许读到半成品
        self._conn.execute(
            "UPDATE jobs SET status='interrupted', finished_at=?, "
            "error='process restarted' WHERE status IN ('queued','running')",
            (_now(),),
        )
        self._conn.execute(
            "UPDATE jk_runs SET status='interrupted', finished_at=?, "
            "error='process restarted' WHERE status IN ('queued','running')",
            (_now(),),
        )
        self._conn.commit()

    def close(self) -> None:
        with self._write_lock:
            self._conn.close()

    # ------------------------------------------------------------------ 读取
    def get_epoch(self) -> int:
        with self._write_lock:
            return self.epoch

    def list_samples(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT person_id, region, age_group, d FROM samples ORDER BY rowid"
        ).fetchall()
        return [dict(r) for r in rows]

    def list_targets(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT dimension, code, population FROM targets ORDER BY rowid"
        ).fetchall()
        return [dict(r) for r in rows]

    def snapshot(self) -> dict:
        with self._write_lock:
            return {
                "epoch": self.epoch,
                "samples": self.list_samples(),
                "targets": self.list_targets(),
            }

    def get_job(self, job_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def latest_succeeded_job(self) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE status='succeeded' "
            "ORDER BY finished_at DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def list_jobs(self, limit: int = 50) -> list[dict]:
        rows = self._conn.execute(
            "SELECT job_id, epoch, status, submitted_at, finished_at, error "
            "FROM jobs ORDER BY submitted_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ 写入
    def _bump_epoch_locked(self) -> int:
        """调用方必须持有 _write_lock。"""
        self.epoch += 1
        self._conn.execute("UPDATE meta SET epoch=? WHERE id=1", (self.epoch,))
        # 旧作业（含在途）永久失效
        self._conn.execute("DELETE FROM jobs")
        # 在途杰克刀运行标记失效（保留行，查询返回 409）；已完成运行保留为
        # 历史报告，但因 epoch 失配同样不再提供估计
        self._conn.execute(
            "UPDATE jk_runs SET status='invalidated', finished_at=?, "
            "error='源校准作业已因输入变更而失效，剩余轮次已跳过' "
            "WHERE status IN ('queued','running')",
            (_now(),),
        )
        self._cancel.set()
        # 唤醒等待者后立即重建事件，供下一轮作业使用
        self._cancel = threading.Event()
        return self.epoch

    def replace_samples(self, samples: list[dict]) -> int:
        with self._write_lock:
            try:
                self._conn.execute("DELETE FROM samples")
                self._conn.executemany(
                    "INSERT INTO samples(person_id, region, age_group, d) "
                    "VALUES (:person_id, :region, :age_group, :d)",
                    samples,
                )
                epoch = self._bump_epoch_locked()
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            return epoch

    def replace_targets(self, targets: list[dict]) -> int:
        with self._write_lock:
            try:
                self._conn.execute("DELETE FROM targets")
                self._conn.executemany(
                    "INSERT INTO targets(dimension, code, population) "
                    "VALUES (:dimension, :code, :population)",
                    targets,
                )
                epoch = self._bump_epoch_locked()
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            return epoch

    # ------------------------------------------------------------------ 作业
    def current_cancel_event(self) -> threading.Event:
        with self._write_lock:
            return self._cancel

    def create_job(self, snapshot: dict) -> dict:
        with self._write_lock:
            job_id = uuid.uuid4().hex
            now = _now()
            epoch = self.epoch
            self._conn.execute(
                "INSERT INTO jobs(job_id, epoch, status, submitted_at, input_snapshot) "
                "VALUES (?,?,?,?,?)",
                (job_id, epoch, "queued", now, json.dumps(snapshot, ensure_ascii=False)),
            )
            self._conn.commit()
            return self.get_job(job_id)  # type: ignore[return-value]

    def mark_running(self, job_id: str) -> None:
        with self._write_lock:
            self._conn.execute(
                "UPDATE jobs SET status='running' WHERE job_id=? AND status='queued'",
                (job_id,),
            )
            self._conn.commit()

    def finish_job(self, job_id: str, status: str, result: dict | None,
                   error: str | None) -> None:
        with self._write_lock:
            self._conn.execute(
                "UPDATE jobs SET status=?, finished_at=?, result=?, error=? "
                "WHERE job_id=?",
                (status, _now(),
                 json.dumps(result, ensure_ascii=False) if result is not None else None,
                 error, job_id),
            )
            self._conn.commit()

    # ------------------------------------------------------------------ 杰克刀
    def create_jk_run(self, payload: dict, expected_epoch: int) -> dict:
        """创建杰克刀运行；expected_epoch 与当前 epoch 不一致时抛 StaleEpochError。"""
        with self._write_lock:
            if self.epoch != expected_epoch:
                raise StaleEpochError(
                    f"epoch 已变为 {self.epoch}（提交时为 {expected_epoch}）"
                )
            jk_id = uuid.uuid4().hex
            self._conn.execute(
                "INSERT INTO jk_runs(jk_id, epoch, status, submitted_at, "
                "source_job_id, request) VALUES (?,?,?,?,?,?)",
                (jk_id, self.epoch, "queued", _now(),
                 payload["source_job_id"],
                 json.dumps(payload, ensure_ascii=False)),
            )
            self._conn.commit()
            row = self.get_jk_run(jk_id)
            assert row is not None
            return row

    def get_jk_run(self, jk_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM jk_runs WHERE jk_id=?", (jk_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_jk_runs(self, limit: int = 50) -> list[dict]:
        rows = self._conn.execute(
            "SELECT jk_id, epoch, status, source_job_id, submitted_at, "
            "finished_at, error FROM jk_runs "
            "ORDER BY submitted_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_jk_running(self, jk_id: str) -> None:
        with self._write_lock:
            self._conn.execute(
                "UPDATE jk_runs SET status='running' "
                "WHERE jk_id=? AND status='queued'",
                (jk_id,),
            )
            self._conn.commit()

    def finish_jk_run(self, jk_id: str, status: str,
                      result: dict | None, failure: dict | None,
                      error: str | None,
                      only_if_epoch: int | None = None) -> bool:
        """写入杰克刀终态。

        仅在运行仍处于非终态时写入；only_if_epoch 给出时还要求全局 epoch
        未变（成功/失败结果只在源作业仍有效时落库）。返回是否写入。
        """
        with self._write_lock:
            if only_if_epoch is not None and self.epoch != only_if_epoch:
                return False
            cur = self._conn.execute(
                "UPDATE jk_runs SET status=?, finished_at=?, result=?, "
                "failure=?, error=? WHERE jk_id=? "
                "AND status IN ('queued','running')",
                (status, _now(),
                 json.dumps(result, ensure_ascii=False) if result is not None else None,
                 json.dumps(failure, ensure_ascii=False) if failure is not None else None,
                 error, jk_id),
            )
            self._conn.commit()
            return cur.rowcount > 0
