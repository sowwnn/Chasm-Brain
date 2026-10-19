"""Backfill exp3 results/*.json from existing stdout logs.

Parses lines of the form:
  S1 EpNNN | L:... TrP:... | Val s1:0.469 ... mean:0.469 MSE:0.123
  ✓ Saved best Stage 1: 0.4685 → rebutal/exp3/checkpoints/swap_subj1/best_stage1.pth

Picks the line whose mean equals the best-saved value (the final improvement) and
writes a summary identical in shape to what train_exp2.py now dumps.
"""
import json
import re
from pathlib import Path

LOG_DIR = Path("rebutal/exp3/logs")
RESULTS = Path("rebutal/exp3/results")
CFG_DIR = Path("rebutal/exp3/configs")
CKPT_DIR = Path("rebutal/exp3/checkpoints")

EP_RE   = re.compile(r"^S1 Ep\s*(\d+) .* Val (.*) mean:([\d.]+) MSE:([\d.]+)")
BEST_RE = re.compile(r"Saved best Stage 1: ([\d.]+)")
SUBJ_RE = re.compile(r"s(\d+):([\d.]+)")

RESULTS.mkdir(parents=True, exist_ok=True)

variants = ["swap", "whatonly", "whereonly"]
subjects = [1, 2, 5, 7]

for variant in variants:
    for sid in subjects:
        log_path = LOG_DIR / f"{variant}_subj{sid}_s1.log"
        if not log_path.exists():
            continue

        text = log_path.read_text().splitlines()
        # Walk lines; whenever we see "Saved best", attribute it to the most
        # recent epoch line. The final accepted epoch is the best.
        best_v, best_epoch, best_mse, per_subj = -1.0, 0, None, {}
        last_epoch_info = None
        for line in text:
            m = EP_RE.search(line)
            if m:
                last_epoch_info = (int(m.group(1)), m.group(2), float(m.group(4)))
                continue
            mb = BEST_RE.search(line)
            if mb and last_epoch_info is not None:
                v = float(mb.group(1))
                if v > best_v:
                    best_v = v
                    best_epoch, val_str, best_mse = last_epoch_info
                    per_subj = {int(sm.group(1)): {'pearson': float(sm.group(2))}
                                for sm in SUBJ_RE.finditer(val_str)}
        if best_v < 0:
            print(f"  skip {log_path.name}: no best line"); continue

        summary = {
            'tag': f'{variant}_subj{sid}',
            'config': str(CFG_DIR / f'exp3_{variant}_subj{sid}.yaml'),
            'stage': 1,
            'branch_mode': {'whatonly': 'what_only', 'whereonly': 'where_only'}.get(variant, variant),
            'train_subjects': [sid],
            'best_epoch': best_epoch,
            'best_val_pearson_mean': best_v,
            'best_val_mse_mean': best_mse,
            'best_val_per_subject': per_subj,
            'checkpoint': str(CKPT_DIR / f'{variant}_subj{sid}/best_stage1.pth'),
            'source': 'backfilled_from_log',
        }
        out = RESULTS / f'{variant}_subj{sid}_stage1_summary.json'
        out.write_text(json.dumps(summary, indent=2))
        print(f"  ✓ {out}  (epoch {best_epoch}, P={best_v:.4f})")
