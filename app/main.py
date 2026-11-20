"""FastAPI 应用：样本/目标更正、后台权重计算、结果取用、删组杰克刀估计。"""

from __future__ import annotations

import json
import math
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .db import Database, StaleEpochError
from .jobs import JobManager
from . import jackknife
from .schemas import (
    JackknifeFailure,
    JackknifePayload,
    JackknifeRecordIn,
    JackknifeResult,
    JackknifeRoundItem,
    JobResult,
    SamplesPayload,
    TargetsPayload,
    TargetTotalsItem,
    WeightItem,
)

DB_PATH = os.environ.get("WEIGHTS_DB", "/data/weights.db")


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    app.state.db = Database(DB_PATH)
    app.state.jobs = JobManager(app.state.db)
    yield
    app.state.jobs.shutdown()
    app.state.db.close()


app = FastAPI(
    title="回访问卷样本权重校正 API",
    version="1.1.0",
    description=(
        "根据地区/年龄段人口总量，对至多 200 人的基础权重 d 做二次规划校正，"
        "满足 0.5d ≤ w ≤ 2d 且目标人群权重合计等于人口总量，"
        "最小化 Σ(w−d)²/d。任何样本/目标修改都会令旧结果与在途计算永久失效。"
        "删组杰克刀接口基于成功校准作业，对家庭月用电量均值给出"
        "全样本/各删组轮加权均值与删组杰克刀标准误。"
    ),
    lifespan=lifespan,
)


def db() -> Database:
    return app.state.db


def jobs() -> JobManager:
    return app.state.jobs


def _sanitize_nonfinite(obj):
    """错误详情中的 NaN/Inf 无法 JSON 序列化，替换为字符串。"""
    if isinstance(obj, float) and not math.isfinite(obj):
        return repr(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_nonfinite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_nonfinite(v) for v in obj]
    return obj


@app.exception_handler(RequestValidationError)
async def handle_request_validation(request, exc):
    # 与默认处理器一致，仅额外清洗非有限浮点（如 kwh=NaN 的输入回显）
    return JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(_sanitize_nonfinite(exc.errors()))},
    )


# ---------------------------------------------------------------- 输入更正
@app.put("/samples", status_code=200, tags=["输入数据"],
         summary="整体替换样本（至多 200 人；使所有旧结果与在途计算失效）")
def put_samples(payload: SamplesPayload):
    samples = [s.model_dump() for s in payload.samples]
    epoch = db().replace_samples(samples)
    return {"epoch": epoch, "count": len(samples),
            "message": "样本已更新，旧结果与在途计算已永久失效，须重新计算"}


@app.get("/samples", tags=["输入数据"], summary="查看当前样本")
def get_samples():
    snap = db().snapshot()
    return {"epoch": snap["epoch"], "samples": snap["samples"]}


@app.put("/targets", status_code=200, tags=["输入数据"],
         summary="整体替换人口目标（至多 12 项；使所有旧结果与在途计算失效）")
def put_targets(payload: TargetsPayload):
    targets = [t.model_dump() for t in payload.targets]
    epoch = db().replace_targets(targets)
    return {"epoch": epoch, "count": len(targets),
            "message": "目标已更新，旧结果与在途计算已永久失效，须重新计算"}


@app.get("/targets", tags=["输入数据"], summary="查看当前人口目标")
def get_targets():
    snap = db().snapshot()
    return {"epoch": snap["epoch"], "targets": snap["targets"]}


# ---------------------------------------------------------------- 计算/结果
@app.post("/calculate", status_code=202, tags=["计算"],
          summary="提交后台权重计算（不阻塞，返回 job_id）")
def calculate():
    snap = db().snapshot()
    if not snap["samples"]:
        raise HTTPException(422, "当前没有样本，无法计算：请先 PUT /samples")
    if not snap["targets"]:
        raise HTTPException(422, "当前没有目标，无法计算：请先 PUT /targets")
    job = jobs().submit()
    return JSONResponse(
        status_code=202,
        content={"job_id": job["job_id"], "epoch": job["epoch"],
                 "status": job["status"],
                 "poll": f"/jobs/{job['job_id']}"},
    )


def _serialize_job(row: dict) -> dict:
    result = json.loads(row["result"]) if row["result"] else None
    weights = totals = None
    objective = l1 = lmax = None
    iis = None
    if row["status"] == "succeeded" and result:
        weights = [WeightItem(**x).model_dump() for x in result["weights"]]
        totals = [TargetTotalsItem(**x).model_dump() for x in result["target_totals"]]
        objective = result["objective"]
        l1 = result["residual_abs_sum"]
        lmax = result["residual_max_abs"]
    elif row["status"] == "infeasible" and result:
        iis = result["infeasible_targets"]

    out = JobResult(
        job_id=row["job_id"],
        epoch=row["epoch"],
        status=row["status"],
        submitted_at=row["submitted_at"],
        finished_at=row["finished_at"],
        input_snapshot=json.loads(row["input_snapshot"]),
        error=row["error"],
        weights=weights,
        objective=objective,
        residual_abs_sum=l1,
        residual_max_abs=lmax,
        target_totals=totals,
        infeasible_targets=iis,
    )
    return out.model_dump(exclude_none=False)


@app.get("/jobs", tags=["计算"], summary="列出最近作业")
def list_jobs():
    return {"epoch": db().get_epoch(), "jobs": db().list_jobs()}


@app.get("/jobs/{job_id}", tags=["计算"],
         summary="查询作业状态与完整结果（仅成功时返回权重，失败/中断无半份结果）")
def get_job(job_id: str):
    row = db().get_job(job_id)
    if row is None:
        raise HTTPException(404, "作业不存在或已因输入变更而永久失效")
    return _serialize_job(row)


@app.get("/result", tags=["计算"],
         summary="取用与当前输入一致的完整权重；无可用完整结果时返回 409")
def get_result():
    row = db().latest_succeeded_job()
    if row is None:
        raise HTTPException(
            409,
            "当前没有与现有输入一致的完整权重结果（尚未计算、计算中、冲突、"
            "数值失败或进程重启均不开放半份结果）",
        )
    data = _serialize_job(dict(row))
    # 一致性核对：作业快照必须与当前输入逐字段一致
    current = db().snapshot()
    snap = data["input_snapshot"]
    if snap.get("samples") != current["samples"] or \
            snap.get("targets") != current["targets"] or \
            snap.get("epoch") != current["epoch"]:
        # 理论上不会发生（epoch 变更即清除作业），双保险
        raise HTTPException(409, "结果与当前输入不一致，已拒绝取用，请重新计算")
    return data


@app.get("/health", tags=["系统"], summary="健康检查")
def health():
    return {"status": "ok", "epoch": db().get_epoch()}


# ---------------------------------------------------------------- 删组杰克刀
@app.post("/jackknife", status_code=202, tags=["删组杰克刀"],
          summary="提交删组杰克刀估计（后台执行，返回 jk_id）")
def post_jackknife(payload: JackknifePayload):
    job = db().get_job(payload.job_id)
    if job is None:
        raise HTTPException(
            409,
            "源校准作业不存在或已因输入变更而失效：请重新 POST /calculate "
            "并在成功后再次提交",
        )
    if job["status"] != "succeeded":
        raise HTTPException(
            409,
            f"源校准作业未成功（当前状态 {job['status']}），"
            "只有成功完成的校准作业才能用于杰克刀估计",
        )
    records = [r.model_dump() for r in payload.records]
    try:
        run = jobs().submit_jackknife(job, records)
    except jackknife.JackknifeError as exc:
        raise HTTPException(422, str(exc))
    except StaleEpochError:
        raise HTTPException(
            409, "提交瞬间输入被修改，源作业已失效：请重新计算后再次提交"
        )
    return JSONResponse(
        status_code=202,
        content={"jk_id": run["jk_id"], "epoch": run["epoch"],
                 "status": run["status"],
                 "source_job_id": job["job_id"],
                 "poll": f"/jackknife/{run['jk_id']}"},
    )


def _serialize_jk(row: dict) -> dict:
    req = json.loads(row["request"])
    result = json.loads(row["result"]) if row["result"] else None
    failure = json.loads(row["failure"]) if row["failure"] else None
    out = JackknifeResult(
        jk_id=row["jk_id"],
        epoch=row["epoch"],
        status=row["status"],
        source_job_id=req["source_job_id"],
        submitted_at=row["submitted_at"],
        finished_at=row["finished_at"],
        records=[JackknifeRecordIn(**r) for r in req["records"]],
        group_count=len(req["labels"]),
        group_size=len(req["samples"]) // len(req["labels"]),
        groups=req["labels"],
        error=row["error"],
        mean_full=result["mean_full"] if result else None,
        rounds=[JackknifeRoundItem(**r) for r in result["rounds"]] if result else None,
        jk_variance=result["jk_variance"] if result else None,
        jk_se=result["jk_se"] if result else None,
        failure=JackknifeFailure(**failure) if failure else None,
    )
    return out.model_dump(exclude_none=False)


@app.get("/jackknife/{jk_id}", tags=["删组杰克刀"],
         summary="查询杰克刀运行状态与完整估计（源作业失效返回 409）")
def get_jackknife(jk_id: str):
    row = db().get_jk_run(jk_id)
    if row is None:
        raise HTTPException(404, "杰克刀运行不存在")
    if row["epoch"] != db().get_epoch() or row["status"] == "invalidated":
        raise HTTPException(
            409,
            "源校准作业已因输入变更而失效，本次杰克刀估计不再可取用："
            "请重新 POST /calculate 并重新提交杰克刀请求",
        )
    return _serialize_jk(row)


@app.get("/jackknife", tags=["删组杰克刀"], summary="列出最近杰克刀运行")
def list_jackknife():
    return {"epoch": db().get_epoch(), "runs": db().list_jk_runs()}
