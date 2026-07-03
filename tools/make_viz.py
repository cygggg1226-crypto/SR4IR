"""Copy the fixed visualization samples of a run into viz/<run>/ as small thumbnails.

The framework (with test.visualize_first_n, default 10) writes annotated images for
the first N validation samples into experiments/<task>/<run>/visualize/. Because the
test loader is sequential, these are the same samples for every run.

Usage:
    python tools/make_viz.py --task det --run det_voc_x4_sr4ir_e10_s42
"""
import argparse
import os
import os.path as osp

import cv2

MAX_SIDE = 512
JPEG_QUALITY = 82


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--task', default='det', choices=['det', 'seg', 'cls'])
    args = parser.parse_args()

    src_dir = osp.join('experiments', args.task, args.run, 'visualize')
    dst_dir = osp.join('viz', args.run)
    if not osp.isdir(src_dir):
        raise SystemExit(f'ERROR: {src_dir} does not exist (run test_only with --visualize first)')
    os.makedirs(dst_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.png')))
    for fname in files:
        img = cv2.imread(osp.join(src_dir, fname))
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = MAX_SIDE / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        out = osp.splitext(fname)[0] + '.jpg'
        cv2.imwrite(osp.join(dst_dir, out), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    print(f'[make_viz] {len(files)} images -> {dst_dir}')


if __name__ == '__main__':
    main()
