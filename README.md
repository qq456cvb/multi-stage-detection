# Multi-Stage Reinforcement Learning for Object Detection

A TensorFlow/[Tensorpack](https://github.com/tensorpack/tensorpack) implementation of [Multi-Stage Reinforcement Learning for Object Detection](https://doi.org/10.1007/978-3-030-17795-9_13) (König et al., CVC 2019), which formulates object localization as a sequential decision process: a DQN agent iteratively zooms and shifts a focus window onto the object (stage 1), then a second stage refines the bounding box with small translation/scale adjustments.

## How It Works

- `env.py` — the localization environment. Stage 1 uses a coarse action space (sub-window zooms over the current focus); stage 2 (`step_refine`, `REFINE_STEP` steps) uses fine box adjustments. The state is the current crop plus a history of past actions (`HISTORY_LEN`).
- `DQNModel.py` / `DQN.py` — DQN on VGG16 features with vanilla, Double (default), and Dueling variants, trained with experience replay (`expreplay.py`) via Tensorpack.
- `vgg16.py` — VGG16 feature extractor; expects `vgg16_weights.npz` ([download](https://www.cs.toronto.edu/~frossard/post/vgg16/)).
- `evaluator.py` — episode rollout and detection evaluation.
- `datasets/` — PASCAL VOC data layer (from the standard py-faster-rcnn `imdb` code).
- `config.py` — task difficulty knobs: `HISTORY_LEN`, `MAX_STEP` (stage-1 steps), `REFINE_STEP` (stage-2 steps).

## Setup

- TensorFlow 1.x (with `tf.contrib`), Tensorpack, OpenCV, numpy
- Pretrained VGG16 weights: place `vgg16_weights.npz` in the repository root
- Datasets: download the **VOC2007** and **VOC2012** devkits into a folder named `data/` under the repository root (so that `data/VOCdevkit/VOC2012/...` exists)

## Train / Evaluate

```bash
python DQN.py --task train                      # train
python DQN.py --task eval --load <checkpoint>   # evaluate
```

Use `--gpu` to select GPUs and `--algo` to switch the DQN variant; see `python DQN.py --help`.

## Reference

```bibtex
@inproceedings{konig2019multi,
  title={Multi-Stage Reinforcement Learning for Object Detection},
  author={K{\"o}nig, Jonas and Malberg, Simon and Martens, Martin and Niehaus, Sebastian and Krohn-Grimberghe, Artus and Ramaswamy, Arunselvan},
  booktitle={Advances in Computer Vision: Proceedings of the 2019 Computer Vision Conference (CVC)},
  pages={178--191},
  year={2019},
  organization={Springer}
}
```
