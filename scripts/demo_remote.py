#!/usr/bin/env python3
"""Run the bundled GPU demo without a desktop and save results for the Mac."""
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.argv = [
    'demo.py', '--root', str(ROOT), '--dataset', 'argo', '--split', 'val',
    '--if_gpu', '--num_frames', '2', '--range_x', '10000', '--range_y', '10000',
    '--range_z', '-10000', '--ground_slack', '0', '--if_hdbscan',
    '--num_clusters', '200', '--min_cluster_size', '20', '--epsilon', '0.25',
    '--speed', '1', '--thres_dist', '0.1', '--max_points', '10000',
    '--thres_box', '0.1', '--thres_rot', '0.1', '--thres_error', '0.2',
    '--thres_iou', '0.2', '--batch_size', '1', '--num_workers', '0',
]
result = runpy.run_path(str(ROOT / 'demo.py'), run_name='__main__')

import numpy as np
import plotly.graph_objects as go
import torch
from utils_eval import compute_epe_test

flow = result['flow']
source, target = result['point_src'], result['point_dst']
pairs = result['pairs'].cpu().numpy()
assert len(pairs) > 0, 'No segments matched in the bundled demo'
assert flow.shape == source.shape and np.isfinite(flow).all()
metric_names = ['EPE3D', 'ACC3DS', 'ACC3DR', 'Outlier', 'Routlier']
metrics = dict(zip(metric_names, map(float, compute_epe_test(flow, result['data']['scene_flow']))))
metrics.update(gpu=torch.cuda.get_device_name(0), matched_segments=len(pairs), points=len(source))
output = ROOT / 'results'
output.mkdir(exist_ok=True)
np.savez_compressed(output / 'demo_flow.npz', source=source, target=target, flow=flow,
                    pairs=pairs, transformations=result['transformations'].cpu().numpy())
(output / 'demo_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')

figure = go.Figure()
for points, name, color in [(source, 'Source', '#25ab65'), (target, 'Target', '#4b87ef'),
                            (source + flow, 'Source + predicted flow', '#ef5555')]:
    stride = max(1, int(np.ceil(len(points) / 8000)))
    sampled = points[::stride]
    figure.add_trace(go.Scatter3d(x=sampled[:, 0], y=sampled[:, 1], z=sampled[:, 2],
                                 mode='markers', name=name, marker=dict(size=1.5, color=color)))
figure.update_layout(title=f"ICP-Flow GPU demo — EPE {metrics['EPE3D']:.4f} m",
                     scene=dict(aspectmode='data'), margin=dict(l=0, r=0, b=0, t=50))
figure.write_html(output / 'demo.html', include_plotlyjs=True)
print(json.dumps(metrics, indent=2))
print(f'Results saved in {output}. Open demo.html on the Mac to rotate the point clouds.')
