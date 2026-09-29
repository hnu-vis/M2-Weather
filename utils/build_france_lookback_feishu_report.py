"""Build the DocxXML report for the M3-France lookback-window sweep."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from html import escape
from pathlib import Path


MODELS = (
    "PatchTST", "xPatch", "STELLA", "S2Transformer",
    "DUET", "iTransformer", "HiSTGNN",
)
LOOKBACKS = (24, 48, 72, 96)
VARIABLES = ("T", "WS", "RH", "P")


def cell(text: str, header: bool = False) -> str:
    tag = "th" if header else "td"
    color = ' background-color="light-gray"' if header else ""
    return f"<{tag}{color}><p>{text}</p></{tag}>"


def table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(cell(escape(item), True) for item in headers)
    body = "".join(
        "<tr>" + "".join(cell(item) for item in row) + "</tr>"
        for row in rows
    )
    return (
        "<table><thead><tr>" + head + "</tr></thead><tbody>" + body
        + "</tbody></table>"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    with args.csv_path.open(encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    if len(source) != len(MODELS) * len(LOOKBACKS):
        raise ValueError(f"Expected 28 result rows, found {len(source)}")
    data = {(row["model"], int(row["lookback"])): row for row in source}
    missing = [key for key in ((m, l) for m in MODELS for l in LOOKBACKS) if key not in data]
    if missing:
        raise ValueError(f"Missing rows: {missing}")

    means: dict[int, float] = {}
    for lookback in LOOKBACKS:
        means[lookback] = sum(
            float(data[model, lookback]["normalized_mse"]) for model in MODELS
        ) / len(MODELS)
    mean_best = min(means, key=means.get)
    best_by_model = {
        model: min(
            LOOKBACKS,
            key=lambda lookback: float(data[model, lookback]["normalized_mse"]),
        )
        for model in MODELS
    }
    grouped: dict[int, list[str]] = defaultdict(list)
    for model, lookback in best_by_model.items():
        grouped[lookback].append(model)

    lines = [
        "<title>M3 France 不同回看窗口实验（预测长度 72）</title>",
        '<h1 seq="auto">关键发现</h1>',
        (
            f"<p>七个模型的归一化总体 MSE 取简单平均后，回看窗口 "
            f"<b>{mean_best}</b> 最低（{means[mean_best]:.4f}）。该平均仅用于概览，"
            "没有按模型规模或类别加权。</p>"
        ),
        "<p>按单个模型的归一化总体 MSE 选择最佳窗口：</p>",
        "<ul>" + "".join(
            f"<li>窗口 {lookback}：{escape('、'.join(models))}</li>"
            for lookback, models in sorted(grouped.items())
        ) + "</ul>",
        (
            "<p>这些结果来自单个随机种子，只能说明本次实验中的观测差异，"
            "不能据此判断差异具有统计显著性。</p>"
        ),
        '<h1 seq="auto">实验设置与指标口径</h1>',
        table(
            ["项目", "设置"],
            [
                ["数据集", "M3 France"],
                ["预测长度", "72 小时"],
                ["回看窗口", "24、48、72、96 小时"],
                ["模型", escape("、".join(MODELS))],
                ["重复次数", "1 次（seed 2024）"],
                ["指标", "归一化 MSE、归一化 MAE；数值越低越好"],
            ],
        ),
        (
            "<p>窗口 48 使用同一配置、同一 seed 的既有完整结果；窗口 24、72、96 "
            "为本次新增训练。HiSTGNN-L96 因显存限制使用 batch 32，xPatch-L96 "
            "因卷积索引上限使用 batch 512；batch 调整不改变损失、样本集合或指标定义。</p>"
        ),
        '<h1 seq="auto">总体归一化结果</h1>',
    ]

    for metric, title in (
        ("normalized_mse", "总体归一化 MSE"),
        ("normalized_mae", "总体归一化 MAE"),
    ):
        lines.append(f'<h2 seq="auto">{title}</h2>')
        result_rows = []
        for model in MODELS:
            values = [float(data[model, lookback][metric]) for lookback in LOOKBACKS]
            best = min(values)
            rendered = [
                f"<b>{value:.4f}</b>" if value == best else f"{value:.4f}"
                for value in values
            ]
            result_rows.append([
                escape(data[model, 24]["category"]), escape(model), *rendered
            ])
        lines.append(table(
            ["类别", "模型", *[f"L={value}" for value in LOOKBACKS]],
            result_rows,
        ))
        lines.append("<p>粗体为该模型在四个窗口中的最低值。</p>")

    lines.append('<h1 seq="auto">逐变量归一化结果</h1>')
    for variable in VARIABLES:
        lines.append(f'<h2 seq="auto">{variable}</h2>')
        result_rows = []
        for model in MODELS:
            rendered = []
            for lookback in LOOKBACKS:
                row = data[model, lookback]
                rendered.append(
                    f"{float(row[f'{variable}_normalized_mse']):.4f} / "
                    f"{float(row[f'{variable}_normalized_mae']):.4f}"
                )
            result_rows.append([escape(model), *rendered])
        lines.append(table(
            ["模型（MSE / MAE）", *[f"L={value}" for value in LOOKBACKS]],
            result_rows,
        ))

    lines.extend([
        '<h1 seq="auto">限制与复现说明</h1>',
        "<ul>",
        "<li>每个配置只运行一次，因此没有均值、标准差或置信区间。</li>",
        "<li>不同模型沿用各自原始超参数；本实验只改变回看窗口并固定预测长度。</li>",
        "<li>表中只写入归一化结果，没有混入反归一化后的物理量指标。</li>",
        "<li>完整机器可读结果保存在 lookback_results.csv 和 lookback_results.json。</li>",
        "</ul>",
    ])
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
