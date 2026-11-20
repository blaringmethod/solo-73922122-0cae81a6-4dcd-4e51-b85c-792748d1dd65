"""删组杰克刀（delete-group jackknife）均值与抽样误差估计（纯函数，不依赖 Web/数据库）。

给定一次成功校准作业的快照（样本、人口目标、校准权重 w）以及每位受访者的
家庭月用电量 y 与删除组编号：

- 全样本均值：ȳ = Σ w_i y_i / Σ w_i（使用原作业权重）；
- 第 g 轮：删去第 g 组，将剩余样本基础权重 d 乘以 G/(G−1)（G 为组数），
  个人权重界限随之调整为 0.5d′ ≤ w′ ≤ 2d′，用原人口目标重新校准，
  得到删组均值 ȳ_(g) = Σ′ w′_i y_i / Σ′ w′_i；
- 删组杰克刀方差：v = (G−1)/G · Σ_g (ȳ_(g) − ȳ)²，标准误 = √v。

任一轮校准失败（冲突或数值失败）时整次估计作废，只返回失败依据。
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable

from . import solver

MIN_GROUPS = 2
MAX_GROUPS = 8


class JackknifeError(ValueError):
    """杰克刀输入组织错误（覆盖不全、分组不合法等），对应 HTTP 422。"""


def organize(
    records: list[dict],
    snapshot_ids: list[str],
) -> tuple[list[float], list[int], list[int]]:
    """校验记录对作业快照的覆盖与分组合法性，并按快照顺序对齐。

    返回 (y, groups, labels)：
    - y[i]、groups[i] 与快照第 i 个样本对齐；
    - labels 为升序去重后的删除组编号。

    规则：每人恰好一条、完整覆盖快照、删除组 2～8 个、各组等大非空。
    组编号只要求是可区分的整数（不要求连续）。
    """
    by_id: dict[str, dict] = {}
    for r in records:
        pid = r["person_id"]
        if pid in by_id:
            raise JackknifeError(f"person_id 重复提交：{pid}（每人须恰好一条）")
        by_id[pid] = r

    rec_ids = set(by_id)
    snap_set = set(snapshot_ids)
    missing = [p for p in snapshot_ids if p not in rec_ids]
    extra = [p for p in by_id if p not in snap_set]
    if missing or extra:
        parts = []
        if missing:
            show = "、".join(missing[:5]) + (" 等" if len(missing) > 5 else "")
            parts.append(f"缺少 {len(missing)} 人的记录：{show}")
        if extra:
            show = "、".join(extra[:5]) + (" 等" if len(extra) > 5 else "")
            parts.append(f"{len(extra)} 条记录不在作业快照中：{show}")
        raise JackknifeError(
            "记录必须完整覆盖作业快照且每人恰好一条（" + "；".join(parts) + "）"
        )

    labels = sorted({r["group"] for r in records})
    g_count = len(labels)
    if not (MIN_GROUPS <= g_count <= MAX_GROUPS):
        raise JackknifeError(
            f"删除组数须为 {MIN_GROUPS}～{MAX_GROUPS} 个，实际为 {g_count} 个"
        )
    sizes = Counter(r["group"] for r in records)
    distinct_sizes = sorted({sizes[g] for g in labels})
    if len(distinct_sizes) != 1:
        detail = "、".join(f"组{g}={sizes[g]}人" for g in labels)
        raise JackknifeError(f"各删除组必须等大非空，实际人数不均：{detail}")

    y = [float(by_id[p]["kwh"]) for p in snapshot_ids]
    groups = [int(by_id[p]["group"]) for p in snapshot_ids]
    return y, groups, labels


def compute(
    samples: list[dict],
    targets: list[dict],
    weights: list[float],
    y: list[float],
    groups: list[int],
    cancelled: Callable[[], bool] | None = None,
) -> dict:
    """执行删组杰克刀估计。

    samples/targets/weights 来自源作业快照与结果（weights 与 samples 对齐），
    y/groups 已由 organize() 按快照顺序对齐。cancelled 在每轮开始前检查：
    一旦为真（源作业已失效），当前已开始的求解收尾后跳过剩余轮次。

    返回 dict，status ∈ {succeeded, failed, invalidated}：
    - succeeded：mean_full、rounds（各轮均值）、jk_variance、jk_se 等；
    - failed：failure 指出组号及冲突目标（infeasible）或数值原因（numeric）；
    - invalidated：已完成轮数 completed_rounds，不给任何估计。
    """
    cancelled = cancelled or (lambda: False)
    n = len(samples)
    labels = sorted(set(groups))
    g_count = len(labels)
    factor = g_count / (g_count - 1.0)

    w = [float(x) for x in weights]
    sum_w = math.fsum(w)
    mean_full = math.fsum(wi * yi for wi, yi in zip(w, y)) / sum_w

    rounds: list[dict] = []
    for rnd, label in enumerate(labels, start=1):
        if cancelled():
            return {
                "status": "invalidated",
                "completed_rounds": len(rounds),
                "total_rounds": g_count,
            }
        keep = [i for i in range(n) if groups[i] != label]
        # 剩余样本基础权重 × G/(G−1)，个人界限 0.5d′/2d′ 随 d′ 自动调整
        sub_samples = [
            {**samples[i], "d": samples[i]["d"] * factor} for i in keep
        ]
        try:
            res = solver.solve_qp(sub_samples, targets)
        except solver.SolverError as exc:
            return {
                "status": "failed",
                "failure": {
                    "round": rnd,
                    "dropped_group": label,
                    "kind": "numeric",
                    "reason": f"第 {rnd} 轮（删去组 {label}）可行性判定数值失败: {exc}",
                    "infeasible_targets": None,
                },
            }
        if res.status == "infeasible":
            iis = [targets[i] for i in (res.infeasible_targets or [])]
            return {
                "status": "failed",
                "failure": {
                    "round": rnd,
                    "dropped_group": label,
                    "kind": "infeasible",
                    "reason": None,
                    "infeasible_targets": iis,
                },
            }
        if res.status != "solved" or res.weights is None:
            return {
                "status": "failed",
                "failure": {
                    "round": rnd,
                    "dropped_group": label,
                    "kind": "numeric",
                    "reason": f"第 {rnd} 轮（删去组 {label}）数值求解失败: "
                              f"{res.reason or '未知原因'}",
                    "infeasible_targets": None,
                },
            }
        wg = [float(x) for x in res.weights]
        sum_wg = math.fsum(wg)
        mean_g = math.fsum(wi * y[i] for wi, i in zip(wg, keep)) / sum_wg
        rounds.append({
            "round": rnd,
            "dropped_group": label,
            "kept": len(keep),
            "weight_sum": sum_wg,
            "mean": mean_g,
        })

    variance = (g_count - 1.0) / g_count * math.fsum(
        (r["mean"] - mean_full) ** 2 for r in rounds
    )
    return {
        "status": "succeeded",
        "group_count": g_count,
        "group_size": n // g_count,
        "groups": labels,
        "mean_full": mean_full,
        "rounds": rounds,
        "jk_variance": variance,
        "jk_se": math.sqrt(variance),
    }
