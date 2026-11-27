# Evaluation

Each probe script freezes the backbone, trains one small head (or kNN / zero-shot
readout) on top of its features, and prints the metric reported in the paper and
model cards.

## Protocols

| Script | Head | Protocol |
|---|---|---|
| `linear.py` | `Linear(1152, 1000)` on `global_features` | SGD 0.1 / Nesterov / cosine / 20 epochs / bs 1024; resize short side by 256/224 (438 for 384) then center-crop 384 |
| `knn.py` | cosine-weighted vote on `global_features` | k ∈ {10, 20, 100} (paper reports k = 20), T = 0.07 |
| `segmentation.py` | `BatchNorm2d -> Conv1x1(150)` on patch tokens + broadcast global feature | 512² crops, AdamW 1e-3 / wd 1e-4, 1.5k warm-up, 40k iters, 512/341 sliding window, best val mIoU |
| `depth.py` | 4x upsample -> `Conv1x1(256)` on patch tokens + broadcast spatial mean | 256 bins over [1e-3, 10] m, SigLoss + 0.5 × gradient loss, AdamW 1e-4 / wd 0.01, 16k warm-up, 50k iters, flip TTA, best-δ1 checkpoint |
| `zeroshot.py` | frozen PE-Core-G14-448 text classifier applied to `global_features` | ImageNet-1k, 80 prompt templates averaged, logit scale 100 |

The dense probes (`segmentation.py` and `depth.py`) follow Appendix A.3.4 of the
paper. `utils.py` holds the shared helpers: `encode()` (normalize → patchify →
forward → unblock to raster), `to_raster()`, `load_backbone()`, and the
ImageNet feature cache shared by `linear.py` and `knn.py`.

`zeroshot.py` is the one probe that needs an extra readout head: IronViT has no
text tower, so it maps `global_features` into PE-Core-G14-448's embedding space
through a small MLP head. That head, the teacher normalization it inverts, and
the fixed text classifier ship together as `pe_zeroshot_head.pt` alongside each
model and are loaded automatically from `--model` (or overridden via `--head`).

## Usage

```shell
python evals/linear.py       --model IRONVIT/IronViT-Softmax-400M --data-root <PATH/TO/IMAGENET> --image-size 384
python evals/knn.py          --model IRONVIT/IronViT-Softmax-400M --data-root <PATH/TO/IMAGENET> --image-size 384
python evals/segmentation.py --model IRONVIT/IronViT-Softmax-400M --data-root <PATH/TO/ADE20K>
python evals/depth.py        --model IRONVIT/IronViT-Softmax-400M --data-root <PATH/TO/NYU>
python evals/zeroshot.py     --model IRONVIT/IronViT-Softmax-400M --data-root <PATH/TO/IMAGENET>
```

Reduce `--iterations` / `--epochs` for a quick sanity run.

## Data Preparation

- **ImageNet-1k** — `<data-root>/train/<class>/*.JPEG` and
  `<data-root>/val/<class>/*.JPEG`. The probes fit the linear head and build
  the kNN reference bank on `train/`, and evaluate on `val/`.

  Download requires accepting the terms at <https://image-net.org/download.php>;
  the archives cannot be fetched without an account. You need
  `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar`.

  ```shell
  # train: the archive holds one tar per synset, which is already the layout
  mkdir -p ${DATASETS_ROOT}/IMAGENET/train
  tar -xf ILSVRC2012_img_train.tar -C ${DATASETS_ROOT}/IMAGENET/train
  cd ${DATASETS_ROOT}/IMAGENET/train
  for f in n*.tar; do mkdir -p "${f%.tar}" && tar -xf "$f" -C "${f%.tar}" && rm "$f"; done

  # val: the archive is flat and must be sorted into synset folders. valprep.sh
  # carries the file-to-synset mapping and uses relative paths, so run it inside
  # the val directory.
  mkdir -p ${DATASETS_ROOT}/IMAGENET/val
  tar -xf ILSVRC2012_img_val.tar -C ${DATASETS_ROOT}/IMAGENET/val
  cd ${DATASETS_ROOT}/IMAGENET/val
  wget -q https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
  bash valprep.sh && rm valprep.sh
  ```

  The gated Hugging Face mirror `ILSVRC/imagenet-1k` holds the same images as
  parquet shards, which would need re-exporting to this layout.

- **ADE20K** — `ADEChallengeData2016/{images,annotations}/{training,validation}`.
  Mask values run 1..150; class 0 means "unlabeled" and is ignored.

  ```shell
  wget https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip -O /tmp/ade.zip
  unzip /tmp/ade.zip -d ${DATASETS_ROOT}/ADE20K
  ```

- **NYUv2** — the subset extracted by [BTS](https://github.com/cleinc/bts):
  `nyu_train.txt` / `nyu_test.txt` with `<rgb> <depth> <focal>` lines and uint16
  depth PNGs in mm. Paths are taken relative to `--data-root`; the leading `/`
  in `nyu_train.txt` is stripped, so both splits resolve under the same root.

  ```shell
  wget https://raw.githubusercontent.com/cleinc/bts/master/train_test_inputs/nyudepthv2_train_files_with_gt.txt \
      -O ${DATASETS_ROOT}/NYU/nyu_train.txt
  wget https://raw.githubusercontent.com/cleinc/bts/master/train_test_inputs/nyudepthv2_test_files_with_gt.txt \
      -O ${DATASETS_ROOT}/NYU/nyu_test.txt
  ```

