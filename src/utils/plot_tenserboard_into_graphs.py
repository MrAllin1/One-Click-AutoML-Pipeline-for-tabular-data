#!/usr/bin/env python3
"""
plot_r2.py

Scan a directory tree of TensorBoard event logs for three pipelines
(TabPFN, TabNet, Tree-based), extract their R² metrics under specific tags,
and plot them side-by-side as boxplots + jitter, with a dashed line for the final test R².
Includes annotation text explaining the colored scatter points.

Usage:
    python plot_r2.py --root /path/to/modelsFinal-exam \
                      --final-test-r2 0.934565641845142 \
                      [--out r2_distribution.png]
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def extract_r2_from_event(path, model_type):
    """
    Load a TensorBoard event file and extract R² values:
      - TabPFN:    all values under any tag containing 'oob_r2_iter'
      - TabNet:    all values under any tag containing 'final_val_r2'
      - Tree-based: take final values under any tags containing
        'val_r2_catboost', 'val_r2_lightgbm', 'val_r2_ensemble';
        fallback to last entries under any tag containing 'val_r2'
    """
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={
            event_accumulator.COMPRESSED_HISTOGRAMS: 0,
            event_accumulator.IMAGES: 0,
            event_accumulator.AUDIO: 0,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 0,
        }
    )
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    values = []

    if model_type == 'tabpfn':
        substr = 'oob_r2_iter'
    elif model_type == 'tabnet':
        substr = 'final_val_r2'
    else:
        # tree-based: look for specific tree tags first
        preferred = ['val_r2_catboost', 'val_r2_lightgbm', 'val_r2_ensemble']
        for key in preferred:
            for tag in tags:
                if key in tag.lower():
                    evs = ea.Scalars(tag)
                    if evs:
                        values.append(evs[-1].value)
        # fallback to any 'val_r2' tags
        if not values:
            substr = 'val_r2'
        else:
            return values

    # for tabpfn and tabnet, or fallback tree
    matched = [tag for tag in tags if substr in tag.lower()]
    for tag in matched:
        evs = ea.Scalars(tag)
        for e in evs:
            if model_type == 'tree':
                # only final entry
                pass
        if model_type == 'tree':
            # take only the last value
            if evs:
                values.append(evs[-1].value)
        else:
            # take full series
            values.extend(e.value for e in evs)

    return values


def gather_all(root):
    tabpfn_vals, tabnet_vals, tree_vals = [], [], []
    for dirpath, _, files in os.walk(root):
        for fname in files:
            if not fname.startswith('events.out.tfevents'):
                continue
            full = os.path.join(dirpath, fname)
            low = dirpath.lower()
            if 'tabpfn' in low:
                model = 'tabpfn'
            elif 'tabnet' in low:
                model = 'tabnet'
            elif 'tree' in low or 'based' in low:
                model = 'tree'
            else:
                continue
            try:
                vals = extract_r2_from_event(full, model)
            except Exception as e:
                print(f"Warning: could not read {full}: {e}")
                continue
            if model == 'tabpfn':
                tabpfn_vals.extend(vals)
            elif model == 'tabnet':
                tabnet_vals.extend(vals)
            else:
                tree_vals.extend(vals)
    return tabpfn_vals, tabnet_vals, tree_vals


def plot_distributions(tabpfn, tabnet, tree, final_r2, out_path=None):
    data = [tabpfn, tabnet, tree]
    labels = ['TabPFN', 'TabNet', 'TreeBased']
    colors = ['C0', 'C1', 'C2']

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    for i, (ys, col) in enumerate(zip(data, colors), start=1):
        if not ys:
            continue
        x = np.random.normal(i, 0.05, size=len(ys))
        ax.scatter(x, ys, alpha=0.7, edgecolors='none', color=col)
    ax.axhline(final_r2,
               linestyle='--', linewidth=1.5,
               label=f'Final Test R² = {final_r2:.3f}')
    # annotation box
    annotation = (
        "• Blue points: TabPFN validation R² (bootstraps)\n"
        "• Orange points: TabNet validation R² (folds)\n"
        "• Green points: Tree-based validation R² (CatBoost, LGBM, ensemble)"
    )
    ax.text(0.02, 0.02, annotation, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))
    ax.set_title('R² Distribution by Pipeline Component')
    ax.set_ylabel('R² score')
    ax.legend(loc='lower right')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300)
        print(f"Plot saved to {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot R² distributions from TensorBoard logs"
    )
    parser.add_argument('--root', '-r', required=True,
                        help='Root directory containing your TensorBoard logs')
    parser.add_argument('--final-test-r2', '-f', type=float, required=True,
                        help='Final ensemble R² on test data (dashed line)')
    parser.add_argument('--out', '-o', default=None,
                        help='Path to save the plot (PNG). If omitted, displays interactively.')
    args = parser.parse_args()
    tabpfn_vals, tabnet_vals, tree_vals = gather_all(args.root)
    print(f"Collected R² counts: TabPFN={len(tabpfn_vals)}, "
          f"TabNet={len(tabnet_vals)}, TreeBased={len(tree_vals)}")
    plot_distributions(tabpfn_vals, tabnet_vals, tree_vals,
                       args.final_test_r2, out_path=args.out)


if __name__ == '__main__':
    main()
