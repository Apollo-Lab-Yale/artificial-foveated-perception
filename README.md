# Artificial Foveated Perception

Code for [Artificial Foveated Perception for Mitigating Shortcut Learning in Robotic Foundation Models](https://arxiv.org/abs/2607.10655).

AFP is a compact task-conditioned mask predictor. Given a sequence of RGB frames and a task description, it predicts for every frame a continuous mask in [0, 1] over the task-relevant objects and the robot end-effector. During fine-tuning of a robotic foundation model the mask is an auxiliary grounding signal on the policy's attention over image tokens. At inference the policy runs without AFP.

## Layout

| Folder | Contents |
| --- | --- |
| `afp/` | The AFP model and `AFPModel`, an API for running a trained checkpoint on frames. Inference only. |
| `afp_integrations/` | The AFP auxiliary loss for policy fine-tuning: the attention-capture hook, the loss, and the projected-gradient (PCGrad) update that combines it with the action loss. |
| `afp_training/` | Training code for the AFP model. |

## Training AFP

Training data are episode folders labeled with the [AFP labeling tool](https://github.com/Apollo-Lab-Yale/afp-annotation-tool). Put the RGB frames of an episode in a folder named `image` and label that folder with the tool. It writes `foreground` and `alpha` next to it. Add a `task_description.txt` with one line of task text. The loader picks up every folder under the data root that has all three subfolders:

```
data/<scenario>/<episode>/
    image/                  RGB frames
    foreground/             foreground frames from the labeling tool
    alpha/                  masks from the labeling tool
    task_description.txt    one line, optional
```

If `task_description.txt` is missing, the loader builds a prompt from the first line of `possible_task_descriptions.txt` in the scenario folder, or from the scenario name alone.

Install the training dependencies and build the deformable attention CUDA op:

```bash
cd afp_training
pip install -r requirements.txt
cd models/ops && sh make.sh && python test.py && cd ../..
```

Set `--vm_path` in `configs/afp/mv3_afp.sh` to your data root and drop the `--wandb` flags if you do not use Weights & Biases, then train. The first argument is the number of GPUs:

```bash
./tools/run_dist_launch.sh 2 ./configs/afp/mv3_afp.sh
```

Checkpoints go to `outputs/mv3_afp/`, with `checkpoint.pth` as the latest epoch. The config trains on 5-frame windows with CLIP text conditioning, which is what `afp/` expects.

## Running AFP on a video

Install `torch`, `torchvision`, `numpy`, `Pillow` and CLIP with `pip install -r requirements.txt` at the repo root, and put a trained checkpoint at `afp/checkpoints/checkpoint.pth`.

```python
import numpy as np
from PIL import Image
from afp import AFPModel

model = AFPModel("afp/checkpoints/checkpoint.pth")

frames = ...  # (N, H, W, 3) uint8 RGB frames in temporal order
masks = model.predict(frames, task_text="put the mug in the drawer")

for i, m in enumerate(masks):  # each mask is (H, W) float32 in [0, 1]
    Image.fromarray((m * 255).astype(np.uint8)).save(f"mask_{i:05d}.png")
```

Frames are processed in windows of five, the same length as in training. `model.predict_single(frame, task_text)` handles one frame. Without a task description, pass `task_text=None` and the model falls back to a generic prompt.

## Citation

```bibtex
@article{sun2026artificial,
  title={Artificial Foveated Perception for Mitigating Shortcut Learning in Robotic Foundation Models},
  author={Sun, Xiatao and Zhuang, Yuan and Negrete, Mateo Sanchez Lopez and Coldea, Matei-Victor and Liang, Chen and Zhang, Haoyang and Liu, Che and Zeng, Ziyao and Li, Shawn and Wang, Qian and others},
  journal={arXiv preprint arXiv:2607.10655},
  year={2026}
}
```
