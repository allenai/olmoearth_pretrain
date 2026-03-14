#!/usr/bin/env python3
"""Monitor wandb runs and send reports via Telegram every 2 hours.

Report 1: S1 dropout — baseline vs no_s1_dropout vs old_exp17
Report 2: Spectral — spectral_mixer, spectral_attn_d128, sup_spectral,
          sup_spectral_less_con vs exp17 vs base_token_masked_all (3 bandsets)

Usage:
    python scripts/monitor_runs.py
    nohup python -u scripts/monitor_runs.py > monitor.log 2>&1 &
"""

import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime

import wandb

PROJECT = "eai-ai2/2026_02_08_masked_neg"

REPORT_1_RUNS = {
    "baseline": "7vs95bhi",
    "no_s1_drop": "eiics9gx",
    "old_exp17": "n0mjrl4o",
}

REPORT_2_RUNS = {
    "attn_d128": "jbe5m5ex",
    "exp17": "n0mjrl4o",
    "3bandset": "0q8si8ko",
}

TRAIN_KEYS = [
    "train/InfoNCE",
    "train/ModalityPatchDiscMasked",
    "train/contrastive/accuracy",
    "train/patchdisc/loss/sentinel2_l2a",
    "train/patchdisc/loss/sentinel1",
    "train/patchdisc/accuracy/sentinel2_l2a",
    "train/patchdisc/accuracy/sentinel1",
    "train/patchdisc/accuracy/landsat",
]

EVAL_KEYS = [
    "eval/m-eurosat/accuracy",
    "eval/m_so2sat/accuracy",
    "eval/mados/miou",
    "eval/pastis/miou",
]

# Some older runs use shorter eval key names
EVAL_KEY_ALIASES = {
    "eval/m-eurosat/accuracy": "eval/m-eurosat",
    "eval/m_so2sat/accuracy": "eval/m_so2sat",
    "eval/mados/miou": "eval/mados",
    "eval/pastis/miou": "eval/pastis",
}

POLL_INTERVAL = 2 * 3600

TELEGRAM_BOT_TOKEN = "8566288306:AAEtaRbO2PnJngjusEBvGovkDg-qe-nQz94"
TELEGRAM_CHAT_ID = "8753280737"


def fetch_latest(api, run_id):
    run = api.run(f"{PROJECT}/{run_id}")
    info = {"name": run.name, "state": run.state, "step": run.lastHistoryStep}
    s = run.summary
    for k in TRAIN_KEYS + EVAL_KEYS:
        v = s.get(k)
        if v is None and k in EVAL_KEY_ALIASES:
            v = s.get(EVAL_KEY_ALIASES[k])
        if isinstance(v, (int, float)):
            info[k] = v
    return info


def short_key(k):
    return (
        k.replace("train/patchdisc/", "pd/")
        .replace("train/contrastive/", "c/")
        .replace("train/", "")
        .replace("eval/", "")
        .replace("/accuracy", "/acc")
    )


def pct(step, total=667200):
    return f"{step / total * 100:.0f}%"


def fv(v, digits=3):
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def build_dropout_report(data):
    """Report 1: S1 dropout experiment."""
    b = data["baseline"]
    n = data["no_s1_drop"]
    o = data["old_exp17"]
    labels = ["baseline", "no_s1_drop", "old_exp17"]
    col_names = ["base", "no_s1", "old17"]

    lines = []
    lines.append(f"<b>[1/2] S1 Dropout — {datetime.now().strftime('%Y-%m-%d %H:%M')}</b>")
    lines.append("")
    for label in labels:
        d = data[label]
        lines.append(f"<b>{label}</b>: {d['state']}, step {d['step']:,} ({pct(d['step'])})")
    lines.append("")

    lines.append("<pre>")
    lines.append(f"{'metric':<20} " + " ".join(f"{c:>8}" for c in col_names))
    lines.append("-" * 48)

    for k in TRAIN_KEYS + ["---"] + EVAL_KEYS:
        if k == "---":
            lines.append("")
            continue
        vals = []
        for label in labels:
            v = data[label].get(k)
            vals.append(f"{v:>8.4f}" if v is not None else f"{'-':>8}")
        lines.append(f"{short_key(k):<20} {''.join(vals)}")
    lines.append("</pre>")

    lines.append("")
    lines.append("<b>Delta (no_s1 - baseline):</b>")
    lines.append("<pre>")
    for k in TRAIN_KEYS + EVAL_KEYS:
        bv, nv = b.get(k), n.get(k)
        if bv is not None and nv is not None:
            d = nv - bv
            sign = "+" if d >= 0 else ""
            lines.append(f"{short_key(k):<20} {sign}{d:.4f}")
    lines.append("</pre>")

    # Summary
    lines.append("")
    lines.append("<b>Summary:</b>")
    step_pct = b["step"] / 667200 * 100

    eurosat_b = b.get("eval/m-eurosat/accuracy")
    eurosat_n = n.get("eval/m-eurosat/accuracy")
    eurosat_o = o.get("eval/m-eurosat/accuracy")
    if eurosat_b is not None and eurosat_n is not None:
        ed = eurosat_n - eurosat_b
        old_str = f", old17 final={fv(eurosat_o)}" if eurosat_o is not None else ""
        if abs(ed) < 0.005:
            lines.append(f"- EuroSat: no meaningful diff ({fv(eurosat_b)} vs {fv(eurosat_n)}){old_str}")
        elif ed > 0:
            lines.append(f"- EuroSat: no_s1_drop ahead ({fv(eurosat_n)} vs {fv(eurosat_b)}, +{ed:.3f}){old_str}")
        else:
            lines.append(f"- EuroSat: baseline ahead ({fv(eurosat_b)} vs {fv(eurosat_n)}, {ed:.3f}){old_str}")

    s1b = b.get("train/patchdisc/accuracy/sentinel1")
    s1n = n.get("train/patchdisc/accuracy/sentinel1")
    if s1b is not None and s1n is not None:
        sd = s1n - s1b
        better = "no_s1_drop" if sd > 0 else "baseline"
        if abs(sd) > 0.01:
            lines.append(f"- S1 pd/acc: {better} better ({fv(s1n)} vs {fv(s1b)})")
        else:
            lines.append(f"- S1 pd/acc: similar ({fv(s1b)} vs {fv(s1n)})")

    lines.append(f"- Progress: {step_pct:.0f}% through training")
    if step_pct < 30:
        lines.append("- Too early, divergence expected after ~200k steps")

    return "\n".join(lines)


def build_spectral_report(data):
    """Report 2: Spectral mixer/attention experiment."""
    labels = list(data.keys())

    lines = []
    lines.append(f"<b>[2/2] Spectral — {datetime.now().strftime('%Y-%m-%d %H:%M')}</b>")
    lines.append("")
    for label in labels:
        d = data[label]
        lines.append(f"<b>{label}</b>: {d['state']}, step {d['step']:,} ({pct(d['step'])})")
    lines.append("")

    lines.append("<pre>")
    lines.append(f"{'metric':<12} " + " ".join(f"{c:>10}" for c in labels))
    lines.append("-" * (12 + 11 * len(labels)))
    lines.append(f"{'step':<12} " + " ".join(f"{data[l]['step']:>10,}" for l in labels))

    for k in EVAL_KEYS:
        vals = []
        for label in labels:
            v = data[label].get(k)
            vals.append(f"{v:>10.4f}" if v is not None else f"{'-':>10}")
        lines.append(f"{short_key(k):<12} {''.join(vals)}")

    lines.append("")
    for k in ["train/InfoNCE", "train/ModalityPatchDiscMasked"]:
        vals = []
        for label in labels:
            v = data[label].get(k)
            vals.append(f"{v:>10.4f}" if v is not None else f"{'-':>10}")
        lines.append(f"{short_key(k):<12} {''.join(vals)}")
    lines.append("</pre>")

    # Summary
    lines.append("")
    lines.append("<b>Summary:</b>")
    exp17_eurosat = data["exp17"].get("eval/m-eurosat/accuracy")
    base3_eurosat = data["3bandset"].get("eval/m-eurosat/accuracy")
    lines.append(f"- Reference: exp17={fv(exp17_eurosat)}, 3bandset={fv(base3_eurosat)}")

    for label in ["attn_d128"]:
        d = data[label]
        eurosat = d.get("eval/m-eurosat/accuracy")
        step_p = d["step"] / 667200 * 100
        if eurosat is not None:
            parts = [f"eurosat={fv(eurosat)}"]
            if exp17_eurosat is not None:
                diff = eurosat - exp17_eurosat
                parts.append(f"vs exp17 {'+' if diff >= 0 else ''}{diff:.3f}")
            if base3_eurosat is not None:
                diff = eurosat - base3_eurosat
                parts.append(f"vs 3b {'+' if diff >= 0 else ''}{diff:.3f}")
            lines.append(f"- {label}: {', '.join(parts)} @ {step_p:.0f}%")
        else:
            lines.append(f"- {label}: no eval yet @ {step_p:.0f}%")

    return "\n".join(lines)


def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = urllib.parse.urlencode(
        {"chat_id": TELEGRAM_CHAT_ID, "parse_mode": "HTML", "text": text}
    ).encode()
    req = urllib.request.Request(url, data=payload)
    resp = urllib.request.urlopen(req)
    result = json.loads(resp.read())
    return result.get("ok", False)


def main():
    os.environ.setdefault(
        "WANDB_API_KEY",
        "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",
    )

    while True:
        # Fresh API each iteration to avoid caching stale data
        api = wandb.Api()

        # Report 1: S1 Dropout
        try:
            data1 = {}
            for label, rid in REPORT_1_RUNS.items():
                data1[label] = fetch_latest(api, rid)
            report1 = build_dropout_report(data1)
            print(report1)
            ok1 = send_telegram(report1)
            print(f"\n[{datetime.now()}] Report 1 sent: {ok1}")
        except Exception as e:
            print(f"[{datetime.now()}] Report 1 error: {e}")
            try:
                send_telegram(f"Report 1 error: {e}")
            except Exception:
                pass

        # Report 2: Spectral
        try:
            data2 = {}
            for label, rid in REPORT_2_RUNS.items():
                data2[label] = fetch_latest(api, rid)
            report2 = build_spectral_report(data2)
            print(report2)
            ok2 = send_telegram(report2)
            print(f"\n[{datetime.now()}] Report 2 sent: {ok2}")
        except Exception as e:
            print(f"[{datetime.now()}] Report 2 error: {e}")
            try:
                send_telegram(f"Report 2 error: {e}")
            except Exception:
                pass

        # Check if any active runs remain
        all_runs = {**REPORT_1_RUNS, **REPORT_2_RUNS}
        active = []
        for label, rid in all_runs.items():
            try:
                run = api.run(f"{PROJECT}/{rid}")
                if run.state == "running":
                    active.append(label)
            except Exception:
                pass

        if not active:
            send_telegram("All monitored runs finished!")
            print("All monitored runs finished. Exiting.")
            break

        print(f"Next check in {POLL_INTERVAL // 3600}h. Active: {active}")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
