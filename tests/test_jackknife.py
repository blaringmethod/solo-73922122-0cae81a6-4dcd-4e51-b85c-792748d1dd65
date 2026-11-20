"""删组杰克刀：纯计算单元测试 + API 端到端（校验、失效语义、轮失败、在途失效）。"""

import importlib
import json
import math
import time

import pytest

from app import jackknife, solver


# ---------------------------------------------------------------- 纯函数：organize
def rec(pid, kwh, group):
    return {"person_id": pid, "kwh": kwh, "group": group}


def test_organize_aligns_to_snapshot_order():
    records = [rec("B", 20, 2), rec("A", 10, 1), rec("C", 30, 2), rec("D", 40, 1)]
    y, groups, labels = jackknife.organize(records, ["A", "B", "C", "D"])
    assert y == [10.0, 20.0, 30.0, 40.0]
    assert groups == [1, 2, 2, 1]
    assert labels == [1, 2]


def test_organize_rejects_duplicate_person():
    with pytest.raises(jackknife.JackknifeError, match="重复"):
        jackknife.organize([rec("A", 1, 1), rec("A", 2, 2)], ["A", "B"])


def test_organize_rejects_missing_and_extra():
    with pytest.raises(jackknife.JackknifeError, match="缺少"):
        jackknife.organize([rec("A", 1, 1)], ["A", "B"])
    with pytest.raises(jackknife.JackknifeError, match="不在作业快照"):
        jackknife.organize([rec("A", 1, 1), rec("B", 1, 2), rec("X", 1, 2)],
                           ["A", "B"])


def test_organize_rejects_bad_group_count():
    # 只有 1 个组
    with pytest.raises(jackknife.JackknifeError, match="2～8"):
        jackknife.organize([rec("A", 1, 5), rec("B", 1, 5)], ["A", "B"])
    # 9 个组（超出 2～8）
    ids = [f"P{i}" for i in range(9)]
    with pytest.raises(jackknife.JackknifeError, match="2～8"):
        jackknife.organize([rec(p, 1, i) for i, p in enumerate(ids)], ids)


def test_organize_rejects_unequal_groups():
    records = [rec("A", 1, 1), rec("B", 1, 1), rec("C", 1, 2)]
    with pytest.raises(jackknife.JackknifeError, match="等大"):
        jackknife.organize(records, ["A", "B", "C"])


# ---------------------------------------------------------------- 纯函数：compute
DET_SAMPLES = [
    {"person_id": "P1", "region": "ALL", "age_group": "a", "d": 1},
    {"person_id": "P2", "region": "ALL", "age_group": "a", "d": 1},
    {"person_id": "P3", "region": "ALL", "age_group": "a", "d": 1},
    {"person_id": "P4", "region": "ALL", "age_group": "a", "d": 1},
]
DET_TARGETS = [{"dimension": "region", "code": "ALL", "population": 4}]


def test_compute_deterministic_hand_values():
    """手算对照：w=1，y=[10,20,30,40]，组 [1,1,2,2]。

    全样本均值 25；删组1后 w'=2,2 均值 35；删组2后均值 15；
    方差 (1/2)(10²+10²)=100，标准误 10。
    """
    y, groups, labels = jackknife.organize(
        [rec("P1", 10, 1), rec("P2", 20, 1), rec("P3", 30, 2), rec("P4", 40, 2)],
        ["P1", "P2", "P3", "P4"],
    )
    out = jackknife.compute(DET_SAMPLES, DET_TARGETS, [1, 1, 1, 1], y, groups)
    assert out["status"] == "succeeded"
    assert out["group_count"] == 2 and out["group_size"] == 2
    assert out["mean_full"] == pytest.approx(25.0)
    means = {r["dropped_group"]: r["mean"] for r in out["rounds"]}
    assert means[1] == pytest.approx(35.0)
    assert means[2] == pytest.approx(15.0)
    assert out["jk_variance"] == pytest.approx(100.0)
    assert out["jk_se"] == pytest.approx(10.0)


def test_compute_scales_base_weights_by_factor():
    """删组后剩余样本基础权重须乘以 G/(G-1)，界限随动。"""
    # 目标合计 = 4：删组后 d' = 2*1 = 2（G=2），两人 w'=2 时合计恰为 4
    out = jackknife.compute(
        DET_SAMPLES, DET_TARGETS, [1, 1, 1, 1],
        [1.0, 1.0, 1.0, 1.0], [1, 1, 2, 2],
    )
    assert out["status"] == "succeeded"
    for r in out["rounds"]:
        assert r["weight_sum"] == pytest.approx(4.0)  # 2 人 × d'=2


def test_compute_infeasible_round_reports_group_and_targets():
    samples = [
        {"person_id": "P1", "region": "R1", "age_group": "a", "d": 2},
        {"person_id": "P2", "region": "R1", "age_group": "a", "d": 2},
        {"person_id": "P3", "region": "R2", "age_group": "a", "d": 2},
        {"person_id": "P4", "region": "R2", "age_group": "a", "d": 2},
    ]
    targets = [
        {"dimension": "region", "code": "R1", "population": 8},
        {"dimension": "region", "code": "R2", "population": 8},
    ]
    # 删去组 1（P1,P2）后 R1 无样本，人口 8 不可行
    out = jackknife.compute(samples, targets, [4, 4, 4, 4],
                            [1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2])
    assert out["status"] == "failed"
    f = out["failure"]
    assert f["kind"] == "infeasible"
    assert f["dropped_group"] == 1 and f["round"] == 1
    assert {(t["dimension"], t["code"]) for t in f["infeasible_targets"]} == {
        ("region", "R1")}


def test_compute_numeric_failure(monkeypatch):
    def boom(samples, targets):
        return solver.QPResult(status="failed", reason="simulated")

    monkeypatch.setattr(solver, "solve_qp", boom)
    out = jackknife.compute(DET_SAMPLES, DET_TARGETS, [1, 1, 1, 1],
                            [1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2])
    assert out["status"] == "failed"
    assert out["failure"]["kind"] == "numeric"
    assert "simulated" in out["failure"]["reason"]


def test_compute_solver_error_is_numeric_failure(monkeypatch):
    def boom(samples, targets):
        raise solver.SolverError("lp crashed")

    monkeypatch.setattr(solver, "solve_qp", boom)
    out = jackknife.compute(DET_SAMPLES, DET_TARGETS, [1, 1, 1, 1],
                            [1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2])
    assert out["status"] == "failed"
    assert out["failure"]["kind"] == "numeric"
    assert "lp crashed" in out["failure"]["reason"]


def test_compute_cancel_skips_remaining_rounds(monkeypatch):
    calls = {"n": 0}
    orig = solver.solve_qp
    state = {"cancel": False}

    def solve_then_cancel(samples, targets):
        calls["n"] += 1
        res = orig(samples, targets)
        state["cancel"] = True  # 第一轮求解收尾后才置位
        return res

    monkeypatch.setattr(solver, "solve_qp", solve_then_cancel)
    out = jackknife.compute(DET_SAMPLES, DET_TARGETS, [1, 1, 1, 1],
                            [1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2],
                            cancelled=lambda: state["cancel"])
    assert out["status"] == "invalidated"
    assert out["completed_rounds"] == 1
    assert out["total_rounds"] == 2
    assert calls["n"] == 1  # 剩余轮次确实被跳过


# ---------------------------------------------------------------- API 端到端
@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("WEIGHTS_DB", str(tmp_path / "test.db"))
    import app.main as main
    importlib.reload(main)
    from fastapi.testclient import TestClient
    with TestClient(main.app) as c:
        yield c, main
    main.app.state.jobs.shutdown()


FEAS_SAMPLES = [
    {"person_id": "P1", "region": "E", "age_group": "y", "d": 2},
    {"person_id": "P2", "region": "E", "age_group": "y", "d": 2},
    {"person_id": "P3", "region": "E", "age_group": "m", "d": 2},
    {"person_id": "P4", "region": "E", "age_group": "m", "d": 2},
    {"person_id": "P5", "region": "W", "age_group": "y", "d": 2},
    {"person_id": "P6", "region": "W", "age_group": "y", "d": 2},
    {"person_id": "P7", "region": "W", "age_group": "m", "d": 2},
    {"person_id": "P8", "region": "W", "age_group": "m", "d": 2},
]
# 目标 = 各单元 d 合计：全样本 w=d 恰可行；每个单元 2 人且分属不同组，
# 任意删一组后各单元仍有人、各轮重校准均可行
FEAS_TARGETS = [
    {"dimension": "region", "code": "E", "population": 8},
    {"dimension": "region", "code": "W", "population": 8},
    {"dimension": "age_group", "code": "y", "population": 8},
    {"dimension": "age_group", "code": "m", "population": 8},
]
JK_KWH = {"P1": 120, "P2": 250, "P3": 90, "P4": 310,
          "P5": 180, "P6": 220, "P7": 140, "P8": 265}
JK_GROUPS = {"P1": 1, "P3": 1, "P2": 2, "P4": 2,
             "P5": 3, "P7": 3, "P6": 4, "P8": 4}  # 4 组 × 2 人，同单元不同组


def poll(c, job_id, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = c.get(f"/jobs/{job_id}")
        if r.status_code == 404:
            return r.json()
        body = r.json()
        if body["status"] not in ("queued", "running"):
            return body
        time.sleep(0.05)
    raise TimeoutError


def poll_jk(c, jk_id, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = c.get(f"/jackknife/{jk_id}")
        if r.status_code != 200:
            return r.status_code, r.json()
        body = r.json()
        if body["status"] not in ("queued", "running"):
            return r.status_code, body
        time.sleep(0.05)
    raise TimeoutError


def setup_succeeded_job(c):
    assert c.put("/samples", json={"samples": FEAS_SAMPLES}).status_code == 200
    assert c.put("/targets", json={"targets": FEAS_TARGETS}).status_code == 200
    job = poll(c, c.post("/calculate").json()["job_id"])
    assert job["status"] == "succeeded"
    return job


def jk_records(groups=None, kwh=None, ids=None):
    groups = groups or JK_GROUPS
    kwh = kwh or JK_KWH
    ids = ids or list(JK_KWH)
    return [{"person_id": p, "kwh": kwh[p], "group": groups[p]} for p in ids]


def test_jackknife_full_flow_independent_recomputation(client):
    c, _ = client
    job = setup_succeeded_job(c)
    r = c.post("/jackknife", json={"job_id": job["job_id"],
                                   "records": jk_records()})
    assert r.status_code == 202, r.json()
    jk_id = r.json()["jk_id"]
    assert r.json()["poll"] == f"/jackknife/{jk_id}"
    code, body = poll_jk(c, jk_id)
    assert code == 200
    assert body["status"] == "succeeded"
    assert body["source_job_id"] == job["job_id"]
    assert body["group_count"] == 4 and body["group_size"] == 2
    assert body["groups"] == [1, 2, 3, 4]

    # ---- 独立重算对照 ----
    w = {x["person_id"]: x["w"] for x in job["weights"]}
    n = len(FEAS_SAMPLES)
    g_count = 4
    mean_full = sum(w[p] * JK_KWH[p] for p in JK_KWH) / sum(w.values())
    assert body["mean_full"] == pytest.approx(mean_full, abs=1e-9)

    round_means = {}
    for g in (1, 2, 3, 4):
        keep = [s for s in FEAS_SAMPLES if JK_GROUPS[s["person_id"]] != g]
        sub = [{**s, "d": s["d"] * g_count / (g_count - 1)} for s in keep]
        res = solver.solve_qp(sub, FEAS_TARGETS)
        assert res.status == "solved"
        mg = sum(wi * JK_KWH[s["person_id"]]
                 for wi, s in zip(res.weights, keep)) / sum(res.weights)
        round_means[g] = mg
    assert len(body["rounds"]) == 4
    for item in body["rounds"]:
        assert item["mean"] == pytest.approx(round_means[item["dropped_group"]],
                                             abs=1e-9)
        assert item["kept"] == n - 2
    var = (g_count - 1) / g_count * sum(
        (m - mean_full) ** 2 for m in round_means.values())
    assert body["jk_variance"] == pytest.approx(var, abs=1e-12)
    assert body["jk_se"] == pytest.approx(math.sqrt(var), abs=1e-12)
    assert body["failure"] is None
    # 记录回显，便于审计
    assert len(body["records"]) == n


def test_jackknife_deterministic_hand_values_via_api(client):
    c, _ = client
    assert c.put("/samples", json={"samples": DET_SAMPLES}).status_code == 200
    assert c.put("/targets", json={"targets": DET_TARGETS}).status_code == 200
    job = poll(c, c.post("/calculate").json()["job_id"])
    assert job["status"] == "succeeded"
    records = [rec("P1", 10, 1), rec("P2", 20, 1), rec("P3", 30, 2), rec("P4", 40, 2)]
    r = c.post("/jackknife", json={"job_id": job["job_id"], "records": records})
    assert r.status_code == 202
    code, body = poll_jk(c, r.json()["jk_id"])
    assert code == 200 and body["status"] == "succeeded"
    assert body["mean_full"] == pytest.approx(25.0)
    means = {x["dropped_group"]: x["mean"] for x in body["rounds"]}
    assert means[1] == pytest.approx(35.0)
    assert means[2] == pytest.approx(15.0)
    assert body["jk_se"] == pytest.approx(10.0)


def test_jackknife_source_job_must_be_valid_and_succeeded(client):
    c, _ = client
    setup_succeeded_job(c)
    # 不存在的作业号 -> 409
    r = c.post("/jackknife", json={"job_id": "0" * 32, "records": jk_records()})
    assert r.status_code == 409
    # 未成功的作业（冲突）-> 409
    assert c.put("/samples", json={"samples": [
        {"person_id": "X", "region": "R1", "age_group": "a1", "d": 1},
        {"person_id": "Y", "region": "R1", "age_group": "a2", "d": 1},
        {"person_id": "Z", "region": "R2", "age_group": "a1", "d": 1},
    ]}).status_code == 200
    assert c.put("/targets", json={"targets": [
        {"dimension": "region", "code": "R1", "population": 4},
        {"dimension": "age_group", "code": "a1", "population": 1},
    ]}).status_code == 200
    job = poll(c, c.post("/calculate").json()["job_id"])
    assert job["status"] == "infeasible"
    r = c.post("/jackknife", json={
        "job_id": job["job_id"],
        "records": [rec("X", 1, 1), rec("Y", 1, 2), rec("Z", 1, 3)]})
    assert r.status_code == 409


def test_jackknife_validation_422(client):
    c, _ = client
    job = setup_succeeded_job(c)
    jid = job["job_id"]

    def post(records, job_id=jid):
        return c.post("/jackknife", json={"job_id": job_id, "records": records})

    # 重复 person_id
    assert post(jk_records() + [rec("P1", 1, 1)]).status_code == 422
    # 缺人
    assert post(jk_records(ids=list(JK_KWH)[:-1])).status_code == 422
    # 快照外人员
    assert post(jk_records(ids=list(JK_KWH)[1:]) + [rec("XX", 1, 1)]
                ).status_code == 422
    # 只有 1 个组
    assert post(jk_records(groups={p: 7 for p in JK_KWH})).status_code == 422
    # 组不均等（3,3,2）
    assert post(jk_records(groups={"P1": 1, "P2": 1, "P3": 1, "P4": 2,
                                   "P5": 2, "P6": 2, "P7": 3, "P8": 3})
                ).status_code == 422
    # kwh 为布尔 / 字符串 / 负数
    bad = jk_records()
    bad[0]["kwh"] = True
    assert post(bad).status_code == 422
    bad = jk_records()
    bad[0]["kwh"] = "120"
    assert post(bad).status_code == 422
    bad = jk_records()
    bad[0]["kwh"] = -5
    assert post(bad).status_code == 422
    # NaN：httpx 的 json= 严格序列化不允许，手动构造请求体
    bad = jk_records()
    bad[0]["kwh"] = float("nan")
    r = c.post("/jackknife",
               content=json.dumps({"job_id": jid, "records": bad}),
               headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    # group 为布尔 / 浮点
    bad = jk_records()
    bad[0]["group"] = True
    assert post(bad).status_code == 422
    bad = jk_records()
    bad[0]["group"] = 1.5
    assert post(bad).status_code == 422
    # 额外字段禁止
    bad = jk_records()
    bad[0]["extra"] = 1
    assert post(bad).status_code == 422
    # 记录为空
    assert post([]).status_code == 422
    # 未知字段（顶层）
    assert c.post("/jackknife", json={"job_id": jid, "records": jk_records(),
                                      "foo": 1}).status_code == 422


def test_jackknife_round_infeasible_reports_group_no_estimates(client):
    c, _ = client
    samples = [
        {"person_id": "P1", "region": "R1", "age_group": "a", "d": 2},
        {"person_id": "P2", "region": "R1", "age_group": "a", "d": 2},
        {"person_id": "P3", "region": "R2", "age_group": "a", "d": 2},
        {"person_id": "P4", "region": "R2", "age_group": "a", "d": 2},
    ]
    targets = [
        {"dimension": "region", "code": "R1", "population": 8},
        {"dimension": "region", "code": "R2", "population": 8},
    ]
    assert c.put("/samples", json={"samples": samples}).status_code == 200
    assert c.put("/targets", json={"targets": targets}).status_code == 200
    job = poll(c, c.post("/calculate").json()["job_id"])
    assert job["status"] == "succeeded"
    records = [rec("P1", 10, 1), rec("P2", 20, 1), rec("P3", 30, 2), rec("P4", 40, 2)]
    r = c.post("/jackknife", json={"job_id": job["job_id"], "records": records})
    assert r.status_code == 202
    code, body = poll_jk(c, r.json()["jk_id"])
    assert code == 200
    assert body["status"] == "failed"
    f = body["failure"]
    assert f["kind"] == "infeasible"
    assert f["dropped_group"] == 1 and f["round"] == 1
    assert {(t["dimension"], t["code"]) for t in f["infeasible_targets"]} == {
        ("region", "R1")}
    # 整次不提供估计
    assert body["mean_full"] is None
    assert body["rounds"] is None
    assert body["jk_se"] is None


def test_jackknife_numeric_failure_round(client, monkeypatch):
    c, _ = client
    job = setup_succeeded_job(c)

    def boom(samples, targets):
        return solver.QPResult(status="failed", reason="simulated numeric failure")

    monkeypatch.setattr(solver, "solve_qp", boom)
    r = c.post("/jackknife", json={"job_id": job["job_id"],
                                   "records": jk_records()})
    assert r.status_code == 202
    code, body = poll_jk(c, r.json()["jk_id"])
    assert code == 200
    assert body["status"] == "failed"
    assert body["failure"]["kind"] == "numeric"
    assert "simulated numeric failure" in body["failure"]["reason"]
    assert body["failure"]["dropped_group"] == 1
    assert body["mean_full"] is None and body["jk_se"] is None


def test_jackknife_invalidation_mid_run_returns_409(client, monkeypatch):
    c, _ = client
    assert c.put("/samples", json={"samples": DET_SAMPLES}).status_code == 200
    assert c.put("/targets", json={"targets": DET_TARGETS}).status_code == 200
    job = poll(c, c.post("/calculate").json()["job_id"])
    assert job["status"] == "succeeded"

    orig = solver.solve_qp

    def slow(samples, targets):
        time.sleep(0.6)
        return orig(samples, targets)

    monkeypatch.setattr(solver, "solve_qp", slow)
    records = [rec("P1", 10, 1), rec("P2", 20, 1), rec("P3", 30, 2), rec("P4", 40, 2)]
    r = c.post("/jackknife", json={"job_id": job["job_id"], "records": records})
    assert r.status_code == 202
    jk_id = r.json()["jk_id"]
    time.sleep(0.2)  # 第一轮求解进行中
    # 计算期间照常更正目标（不阻塞），同时令源作业失效
    assert c.put("/targets", json={"targets": DET_TARGETS}).status_code == 200
    # 已开始的当轮收尾、剩余轮次跳过 -> 409
    deadline = time.time() + 10
    code = None
    while time.time() < deadline:
        got = c.get(f"/jackknife/{jk_id}")
        if got.status_code == 409:
            code = 409
            break
        time.sleep(0.05)
    assert code == 409
    # 源作业本身也已 404
    assert c.get(f"/jobs/{job['job_id']}").status_code == 404


def test_jackknife_result_invalidated_after_input_change(client):
    c, _ = client
    job = setup_succeeded_job(c)
    r = c.post("/jackknife", json={"job_id": job["job_id"],
                                   "records": jk_records()})
    jk_id = r.json()["jk_id"]
    code, body = poll_jk(c, jk_id)
    assert code == 200 and body["status"] == "succeeded"
    # 任何输入更正后，历史杰克刀估计不再可取用
    assert c.put("/samples", json={"samples": FEAS_SAMPLES}).status_code == 200
    assert c.get(f"/jackknife/{jk_id}").status_code == 409


def test_jackknife_restart_marks_running_interrupted(tmp_path, monkeypatch):
    db_file = tmp_path / "restart.db"
    monkeypatch.setenv("WEIGHTS_DB", str(db_file))
    import app.main as main
    importlib.reload(main)
    from fastapi.testclient import TestClient

    with TestClient(main.app) as c:
        job = setup_succeeded_job(c)
        # 直接落库一个 queued 运行（不入线程池），模拟重启残留
        payload = {"source_job_id": job["job_id"], "samples": FEAS_SAMPLES,
                   "targets": FEAS_TARGETS, "weights": [1.0] * 8,
                   "y": [1.0] * 8, "groups": [1, 1, 2, 2, 3, 3, 4, 4],
                   "labels": [1, 2, 3, 4], "records": jk_records()}
        run = main.app.state.db.create_jk_run(
            payload, expected_epoch=main.app.state.db.get_epoch())
    main.app.state.jobs.shutdown()
    db2 = main.Database(str(db_file))
    row = db2.get_jk_run(run["jk_id"])
    assert row["status"] == "interrupted"
    db2.close()


def test_jackknife_list_and_openapi(client):
    c, _ = client
    job = setup_succeeded_job(c)
    r = c.post("/jackknife", json={"job_id": job["job_id"],
                                   "records": jk_records()})
    jk_id = r.json()["jk_id"]
    poll_jk(c, jk_id)
    listing = c.get("/jackknife").json()
    assert any(x["jk_id"] == jk_id for x in listing["runs"])
    spec = c.get("/openapi.json").json()
    assert "/jackknife" in spec["paths"]
    assert "/jackknife/{jk_id}" in spec["paths"]
    assert c.get("/docs").status_code == 200
