"""Collect metrics of one experiment run into results.csv.

Parses experiments/<task>/<run>/test_log.txt (fallback: train_log.txt) and
extracts PSNR / LPIPS / COCO mAP, then appends or updates one row in
results.csv at the repo root.

Usage:
    python collect_results.py --task det --run det_voc_x4_sr4ir_e10_s42 [--notes "..."]
"""
import argparse
import csv
import datetime
import os
import os.path as osp
import re
import subprocess

FIELDS = ['run', 'date', 'git_commit', 'task', 'scale', 'setting', 'epochs', 'seed',
          'map_50', 'map_5095', 'ap_small', 'psnr', 'lpips', 'source_log', 'notes']

RE_AP_5095 = re.compile(r"Average Precision\s+\(AP\) @\[ IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*\|\s*maxDets=100\s*\]\s*=\s*([\d.-]+)")
RE_AP_50 = re.compile(r"Average Precision\s+\(AP\) @\[ IoU=0\.50\s*\|\s*area=\s*all\s*\|\s*maxDets=100\s*\]\s*=\s*([\d.-]+)")
RE_AP_S = re.compile(r"Average Precision\s+\(AP\) @\[ IoU=0\.50:0\.95\s*\|\s*area=\s*small\s*\|\s*maxDets=100\s*\]\s*=\s*([\d.-]+)")
RE_PSNR = re.compile(r"Test:.*?PSNR\s+([\d.]+)")
RE_LPIPS = re.compile(r"Test:.*?LPIPS\s+([\d.]+)")


def last_match(regex, text):
    matches = regex.findall(text)
    return matches[-1] if matches else ''


def parse_yml_meta(run, task):
    """Best-effort scale/epoch/seed extraction from the config copied into the exp dir."""
    meta = {'scale': '', 'epochs': '', 'seed': ''}
    exp_dir = osp.join('experiments', task, run)
    yml_files = []
    for root, _, files in os.walk(exp_dir):
        if os.sep + 'models' in root or os.sep + 'checkpoints' in root or os.sep + 'visualize' in root:
            continue
        yml_files += [osp.join(root, f) for f in files if f.endswith('.yml')]
    if not yml_files:
        return meta
    with open(sorted(yml_files)[0], encoding='utf-8') as f:
        text = f.read()
    for key, pattern in [('scale', r"^scale:\s*(\d+)"), ('epochs', r"^\s+epoch:\s*(\d+)"),
                         ('seed', r"^manual_seed:\s*(\d+)")]:
        m = re.search(pattern, text, re.M)
        if m:
            meta[key] = m.group(1)
    return meta


def guess_setting(run):
    name = run.lower()
    for token in ['sr4ir', 'swt', 's2t', 't2s', 'l2t', 'h2t', 'sr']:
        if token in name:
            return {'swt': 'S+T', 's2t': 'S2T', 't2s': 'T2S', 'l2t': 'L2T',
                    'h2t': 'H2T', 'sr4ir': 'SR4IR', 'sr': 'SR'}[token]
    return ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, help='experiment name (folder under experiments/<task>/)')
    parser.add_argument('--task', default='det', choices=['det', 'seg', 'cls'])
    parser.add_argument('--csv', default='results.csv')
    parser.add_argument('--notes', default='')
    args = parser.parse_args()

    exp_dir = osp.join('experiments', args.task, args.run)
    source_log = None
    for candidate in ['test_log.txt', 'train_log.txt']:
        path = osp.join(exp_dir, candidate)
        if osp.exists(path):
            source_log = path
            break
    if source_log is None:
        raise SystemExit(f'ERROR: no test_log.txt/train_log.txt under {exp_dir}')

    with open(source_log, encoding='utf-8', errors='replace') as f:
        text = f.read()

    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], text=True).strip()
    except Exception:
        commit = ''

    meta = parse_yml_meta(args.run, args.task)
    row = {
        'run': args.run,
        'date': datetime.date.today().isoformat(),
        'git_commit': commit,
        'task': args.task,
        'scale': meta['scale'],
        'setting': guess_setting(args.run),
        'epochs': meta['epochs'],
        'seed': meta['seed'],
        'map_50': last_match(RE_AP_50, text),
        'map_5095': last_match(RE_AP_5095, text),
        'ap_small': last_match(RE_AP_S, text),
        'psnr': last_match(RE_PSNR, text),
        'lpips': last_match(RE_LPIPS, text),
        'source_log': source_log.replace('\\', '/'),
        'notes': args.notes,
    }

    rows = []
    if osp.exists(args.csv):
        with open(args.csv, newline='', encoding='utf-8') as f:
            rows = [r for r in csv.DictReader(f)]
    rows = [r for r in rows if r.get('run') != args.run] + [row]

    with open(args.csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[collect_results] {args.run}: mAP@50={row['map_50']} mAP@[.5:.95]={row['map_5095']} "
          f"PSNR={row['psnr']} LPIPS={row['lpips']} -> {args.csv}")


if __name__ == '__main__':
    main()
