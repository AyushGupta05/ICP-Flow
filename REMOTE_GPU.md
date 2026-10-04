# ICP-Flow on AWS from your Mac

The remote VS Code workspace is `/home/ubuntu/ICP-Flow` on the `icp-flow` SSH host.
Editing files, running Python, debugging, and opening terminals in that window all operate on Linux.
The original Mac checkout is a separate copy; changes are not automatically synchronized.

## Open, stop, and reconnect

Run these commands in a **Mac** terminal:

```bash
icp-flow             # Start if necessary, update the SSH rule, open VS Code
icp-flow status      # Show the instance state and stable IP
icp-flow stop        # Stop GPU billing; keep files and the Elastic IP
icp-flow ssh         # Start if necessary, then open a Linux shell
```

You can also double-click `Open ICP-Flow GPU.command` on your Mac Desktop.
If the shortcut is unavailable in an existing terminal, open a new terminal or use:

```bash
bash /Users/ayush/dev/ICP-Flow/scripts/aws_gpu.sh open
```

An expired AWS CLI session will open an AWS sign-in flow. The helper replaces the
dedicated SSH rule with your current public IPv4 address, so reconnecting works after changing networks.
Only SSH (TCP 22) is exposed, restricted to that one address. The private key stays
at `~/.ssh/icp-flow-aws` on your Mac; AWS account credentials are not installed on Linux.

## Run and inspect the demo

In the **remote VS Code terminal**, the `icp_flow` environment activates automatically:

```bash
python scripts/demo_remote.py
```

Alternatively, press F5 and select **ICP-Flow: GPU demo with browser output**.
The bundled `demo.npz` is sufficient; full Waymo, nuScenes, and Argoverse datasets are not installed.
The runner saves predictions in `results/demo_flow.npz`, metrics in
`results/demo_metrics.json`, and an interactive point-cloud viewer in `results/demo.html`.
It uses the same demo pipeline without requiring an Open3D desktop window.

To preview the interactive results from the remote terminal:

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory results
```

VS Code's **Ports** tab can forward port 8765 and open it in your Mac browser.
The HTTP server listens only on Linux's loopback interface and does not need a public AWS port.
The workspace also includes a **Preview ICP-Flow results** task for this command.

## Machine and installed environment

| Item | Value |
| --- | --- |
| AWS region | `us-east-1` |
| Instance | `i-08db002a48331631d` — ICP-Flow GPU |
| Type | `g5.2xlarge`: 8 vCPUs, 32 GB RAM, NVIDIA A10G GPU |
| Stable Elastic IP | `23.20.177.21` |
| OS | Ubuntu 22.04, AWS Deep Learning Base image |
| Storage | 200 GB encrypted persistent gp3 volume |
| Python environment | `/home/ubuntu/miniforge3/envs/icp_flow` |
| Core versions | Python 3.9, PyTorch 1.12, CUDA toolkit 11.6.2, PyTorch3D 0.7.4, NumPy 1.23.5 |
| Native modules | Modified Patchwork++ Python binding and custom CUDA histogram |

The image's NVIDIA driver reports its maximum supported CUDA version. PyTorch and
the repo's extension use the separately installed CUDA 11.6 toolchain.
The environment is exported to `~/setup-logs/environment-installed.yml` on Linux.
Installation and build logs are in `~/setup-logs/`.

The AWS price retrieved during setup is **$1.212/hour while running**, plus storage
and public IPv4 charges. Stopping preserves the disk and Elastic IP and still incurs
their charges. The instance has termination protection, and the disk is configured
to remain even if the instance is eventually terminated.

To rebuild the Patchwork++ Python module, activate `icp_flow` and run:

```bash
cmake -S patchwork-plusplus -B patchwork-plusplus/build \
  -DFETCHCONTENT_SOURCE_DIR_PYBIND11=/home/ubuntu/deps/pybind11 \
  -DPYBIND11_FINDPYTHON=ON \
  -DPython_EXECUTABLE=/home/ubuntu/miniforge3/envs/icp_flow/bin/python
cmake --build patchwork-plusplus/build --target pypatchworkpp -j4
```

The newer pybind11 source supports Python 3.9 while preserving the repo's modified
ground-segmentation implementation. The demo parser also accepts the missing
`--thres_rot` argument used by the supplied `demo.sh`.
