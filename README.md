## StyleGAN2-ADA &mdash; Official PyTorch implementation

![Teaser image](./docs/stylegan2-ada-teaser-1024x252.png)

**Training Generative Adversarial Networks with Limited Data**<br>
Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, Timo Aila<br>
https://arxiv.org/abs/2006.06676<br>

Abstract: *Training generative adversarial networks (GAN) using too little data typically leads to discriminator overfitting, causing training to diverge. We propose an adaptive discriminator augmentation mechanism that significantly stabilizes training in limited data regimes. The approach does not require changes to loss functions or network architectures, and is applicable both when training from scratch and when fine-tuning an existing GAN on another dataset. We demonstrate, on several datasets, that good results are now possible using only a few thousand training images, often matching StyleGAN2 results with an order of magnitude fewer images. We expect this to open up new application domains for GANs. We also find that the widely used CIFAR-10 is, in fact, a limited data benchmark, and improve the record FID from 5.59 to 2.42.*

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/)

## Release notes

This repository is a faithful reimplementation of [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada/) in PyTorch, focusing on correctness, performance, and compatibility.

**Correctness**
* Full support for all primary training configurations.
* Extensive verification of image quality, training curves, and quality metrics against the TensorFlow version.
* Results are expected to match in all cases, excluding the effects of pseudo-random numbers and floating-point arithmetic.

**Performance**
* Training is typically 5%&ndash;30% faster compared to the TensorFlow version on NVIDIA Tesla V100 GPUs.
* Inference is up to 35% faster in high resolutions, but it may be slightly slower in low resolutions.
* GPU memory usage is comparable to the TensorFlow version.
* Faster startup time when training new networks (<50s), and also when using pre-trained networks (<4s).
* New command line options for tweaking the training performance.

**Compatibility**
* Compatible with old network pickles created using the TensorFlow version.
* New ZIP/PNG based dataset format for maximal interoperability with existing 3rd party tools.
* TFRecords datasets are no longer supported &mdash; they need to be converted to the new format.
* New JSON-based format for logs, metrics, and training curves.
* Training curves are also exported in the old TFEvents format if TensorBoard is installed.
* Command line syntax is mostly unchanged, with a few exceptions (e.g., `dataset_tool.py`).
* Comparison methods are not supported (`--cmethod`, `--dcap`, `--cfg=cifarbaseline`, `--aug=adarv`)
* **Truncation is now disabled by default.**

## Data repository

| Path | Description
| :--- | :----------
| [stylegan2-ada-pytorch](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/) | Main directory hosted on Amazon S3
| &ensp;&ensp;&boxvr;&nbsp; [ada-paper.pdf](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/ada-paper.pdf) | Paper PDF
| &ensp;&ensp;&boxvr;&nbsp; [images](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/images/) | Curated example images produced using the pre-trained models
| &ensp;&ensp;&boxvr;&nbsp; [videos](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/videos/) | Curated example interpolation videos
| &ensp;&ensp;&boxur;&nbsp; [pretrained](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/) | Pre-trained models
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; ffhq.pkl | FFHQ at 1024x1024, trained using original StyleGAN2
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; metfaces.pkl | MetFaces at 1024x1024, transfer learning from FFHQ using ADA
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; afhqcat.pkl | AFHQ Cat at 512x512, trained from scratch using ADA
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; afhqdog.pkl | AFHQ Dog at 512x512, trained from scratch using ADA
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; afhqwild.pkl | AFHQ Wild at 512x512, trained from scratch using ADA
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; cifar10.pkl | Class-conditional CIFAR-10 at 32x32
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; brecahad.pkl | BreCaHAD at 512x512, trained from scratch using ADA
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; [paper-fig7c-training-set-sweeps](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/) | Models used in Fig.7c (sweep over training set size)
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; [paper-fig11a-small-datasets](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig11a-small-datasets/) | Models used in Fig.11a (small datasets & transfer learning)
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; [paper-fig11b-cifar10](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig11b-cifar10/) | Models used in Fig.11b (CIFAR-10)
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; [transfer-learning-source-nets](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/) | Models used as starting point for transfer learning
| &ensp;&ensp;&ensp;&ensp;&boxur;&nbsp; [metrics](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/) | Feature detectors used by the quality metrics

## Requirements

* Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
* 1&ndash;8 high-end NVIDIA GPUs with at least 12 GB of memory. We have done all testing and development using NVIDIA DGX-1 with 8 Tesla V100 GPUs.
* 64-bit Python 3.7 and PyTorch 1.7.1. See [https://pytorch.org/](https://pytorch.org/) for PyTorch install instructions.
* CUDA toolkit 11.0 or later.  Use at least version 11.1 if running on RTX 3090.  (Why is a separate CUDA toolkit installation required?  See comments in [#2](https://github.com/NVlabs/stylegan2-ada-pytorch/issues/2#issuecomment-779457121).)
* Python libraries: `pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3`.  We use the Anaconda3 2020.11 distribution which installs most of these by default.
* Docker users: use the [provided Dockerfile](./Dockerfile) to build an image with the required library dependencies.

The code relies heavily on custom PyTorch extensions that are compiled on the fly using NVCC. On Windows, the compilation requires Microsoft Visual Studio. We recommend installing [Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/) and adding it into `PATH` using `"C:\Program Files (x86)\Microsoft Visual Studio\<VERSION>\Community\VC\Auxiliary\Build\vcvars64.bat"`.

## Getting started

Pre-trained networks are stored as `*.pkl` files that can be referenced using local filenames or URLs:

```.bash
# Generate curated MetFaces images without truncation (Fig.10 left)
python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \
    --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

# Generate uncurated MetFaces images with truncation (Fig.12 upper left)
python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \
    --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

# Generate class conditional CIFAR-10 images (Fig.17 left, Car)
python generate.py --outdir=out --seeds=0-35 --class=1 \
    --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

# Style mixing example
python style_mixing.py --outdir=out --rows=85,100,75,458,1500 --cols=55,821,1789,293 \
    --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
```

Outputs from the above commands are placed under `out/*.png`, controlled by `--outdir`. Downloaded network pickles are cached under `$HOME/.cache/dnnlib`, which can be overridden by setting the `DNNLIB_CACHE_DIR` environment variable. The default PyTorch extension build directory is `$HOME/.cache/torch_extensions`, which can be overridden by setting `TORCH_EXTENSIONS_DIR`.

**Docker**: You can run the above curated image example using Docker as follows:

```.bash
docker build --tag sg2ada:latest .
./docker_run.sh python3 generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \
    --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
```

Note: The Docker image requires NVIDIA driver release `r455.23` or later.

**Legacy networks**: The above commands can load most of the network pickles created using the previous TensorFlow versions of StyleGAN2 and StyleGAN2-ADA. However, for future compatibility, we recommend converting such legacy pickles into the new format used by the PyTorch version:

```.bash
python legacy.py \
    --source=https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-cat-config-f.pkl \
    --dest=stylegan2-cat-config-f.pkl
```

## Projecting images to latent space

To find the matching latent vector for a given image file, run:

```.bash
python projector.py --outdir=out --target=~/mytargetimg.png \
    --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
```

For optimal results, the target image should be cropped and aligned similar to the [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset). The above command saves the projection target `out/target.png`, result `out/proj.png`, latent vector `out/projected_w.npz`, and progression video `out/proj.mp4`. You can render the resulting latent vector by specifying `--projected_w` for `generate.py`:

```.bash
python generate.py --outdir=out --projected_w=out/projected_w.npz \
    --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
```

## Using networks from Python

You can use pre-trained networks in your own Python code as follows:

```.python
with open('ffhq.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
z = torch.randn([1, G.z_dim]).cuda()    # latent codes
c = None                                # class labels (not used in this example)
img = G(z, c)                           # NCHW, float32, dynamic range [-1, +1]
```

The above code requires `torch_utils` and `dnnlib` to be accessible via `PYTHONPATH`. It does not need source code for the networks themselves &mdash; their class definitions are loaded from the pickle via `torch_utils.persistence`.

The pickle contains three networks. `'G'` and `'D'` are instantaneous snapshots taken during training, and `'G_ema'` represents a moving average of the generator weights over several training steps. The networks are regular instances of `torch.nn.Module`, with all of their parameters and buffers placed on the CPU at import and gradient computation disabled by default.

The generator consists of two submodules, `G.mapping` and `G.synthesis`, that can be executed separately. They also support various additional options:

```.python
w = G.mapping(z, c, truncation_psi=0.5, truncation_cutoff=8)
img = G.synthesis(w, noise_mode='const', force_fp32=True)
```

Please refer to [`generate.py`](./generate.py), [`style_mixing.py`](./style_mixing.py), and [`projector.py`](./projector.py) for further examples.

## Preparing datasets

Datasets are stored as uncompressed ZIP archives containing uncompressed PNG files and a metadata file `dataset.json` for labels.

Custom datasets can be created from a folder containing images; see [`python dataset_tool.py --help`](./docs/dataset-tool-help.txt) for more information. Alternatively, the folder can also be used directly as a dataset, without running it through `dataset_tool.py` first, but doing so may lead to suboptimal performance.

Legacy TFRecords datasets are not supported &mdash; see below for instructions on how to convert them.

**FFHQ**:

Step 1: Download the [Flickr-Faces-HQ dataset](https://github.com/NVlabs/ffhq-dataset) as TFRecords.

Step 2: Extract images from TFRecords using `dataset_tool.py` from the [TensorFlow version of StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada/):

```.bash
# Using dataset_tool.py from TensorFlow version at
# https://github.com/NVlabs/stylegan2-ada/
python ../stylegan2-ada/dataset_tool.py unpack \
    --tfrecord_dir=~/ffhq-dataset/tfrecords/ffhq --output_dir=/tmp/ffhq-unpacked
```

Step 3: Create ZIP archive using `dataset_tool.py` from this repository:

```.bash
# Original 1024x1024 resolution.
python dataset_tool.py --source=/tmp/ffhq-unpacked --dest=~/datasets/ffhq.zip

# Scaled down 256x256 resolution.
#
# Note: --resize-filter=box is required to reproduce FID scores shown in the
# paper.  If you don't need to match exactly, it's better to leave this out
# and default to Lanczos.  See https://github.com/NVlabs/stylegan2-ada-pytorch/issues/283#issuecomment-1731217782
python dataset_tool.py --source=/tmp/ffhq-unpacked --dest=~/datasets/ffhq256x256.zip \
    --width=256 --height=256 --resize-filter=box
```

**MetFaces**: Download the [MetFaces dataset](https://github.com/NVlabs/metfaces-dataset) and create ZIP archive:

```.bash
python dataset_tool.py --source=~/downloads/metfaces/images --dest=~/datasets/metfaces.zip
```

**AFHQ**: Download the [AFHQ dataset](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq) and create ZIP archive:

```.bash
python dataset_tool.py --source=~/downloads/afhq/train/cat --dest=~/datasets/afhqcat.zip
python dataset_tool.py --source=~/downloads/afhq/train/dog --dest=~/datasets/afhqdog.zip
python dataset_tool.py --source=~/downloads/afhq/train/wild --dest=~/datasets/afhqwild.zip
```

**CIFAR-10**: Download the [CIFAR-10 python version](https://www.cs.toronto.edu/~kriz/cifar.html) and convert to ZIP archive:

```.bash
python dataset_tool.py --source=~/downloads/cifar-10-python.tar.gz --dest=~/datasets/cifar10.zip
```

**LSUN**: Download the desired categories from the [LSUN project page](https://www.yf.io/p/lsun/) and convert to ZIP archive:

```.bash
python dataset_tool.py --source=~/downloads/lsun/raw/cat_lmdb --dest=~/datasets/lsuncat200k.zip \
    --transform=center-crop --width=256 --height=256 --max_images=200000

python dataset_tool.py --source=~/downloads/lsun/raw/car_lmdb --dest=~/datasets/lsuncar200k.zip \
    --transform=center-crop-wide --width=512 --height=384 --max_images=200000
```

**BreCaHAD**:

Step 1: Download the [BreCaHAD dataset](https://figshare.com/articles/BreCaHAD_A_Dataset_for_Breast_Cancer_Histopathological_Annotation_and_Diagnosis/7379186).

Step 2: Extract 512x512 resolution crops using `dataset_tool.py` from the [TensorFlow version of StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada/):

```.bash
# Using dataset_tool.py from TensorFlow version at
# https://github.com/NVlabs/stylegan2-ada/
python dataset_tool.py extract_brecahad_crops --cropsize=512 \
    --output_dir=/tmp/brecahad-crops --brecahad_dir=~/downloads/brecahad/images
```

Step 3: Create ZIP archive using `dataset_tool.py` from this repository:

```.bash
python dataset_tool.py --source=/tmp/brecahad-crops --dest=~/datasets/brecahad.zip
```

## Training new networks

In its most basic form, training new networks boils down to:

```.bash
python train.py --outdir=~/training-runs --data=~/mydataset.zip --gpus=1 --dry-run
python train.py --outdir=~/training-runs --data=~/mydataset.zip --gpus=1
```

The first command is optional; it validates the arguments, prints out the training configuration, and exits. The second command kicks off the actual training.

In this example, the results are saved to a newly created directory `~/training-runs/<ID>-mydataset-auto1`, controlled by `--outdir`. The training exports network pickles (`network-snapshot-<INT>.pkl`) and example images (`fakes<INT>.png`) at regular intervals (controlled by `--snap`). For each pickle, it also evaluates FID (controlled by `--metrics`) and logs the resulting scores in `metric-fid50k_full.jsonl` (as well as TFEvents if TensorBoard is installed).

The name of the output directory reflects the training configuration. For example, `00000-mydataset-auto1` indicates that the *base configuration* was `auto1`, meaning that the hyperparameters were selected automatically for training on one GPU. The base configuration is controlled by `--cfg`:

| Base config           | Description
| :-------------------- | :----------
| `auto`&nbsp;(default) | Automatically select reasonable defaults based on resolution and GPU count. Serves as a good starting point for new datasets but does not necessarily lead to optimal results.
| `stylegan2`           | Reproduce results for StyleGAN2 config F at 1024x1024 using 1, 2, 4, or 8 GPUs.
| `paper256`            | Reproduce results for FFHQ and LSUN Cat at 256x256 using 1, 2, 4, or 8 GPUs.
| `paper512`            | Reproduce results for BreCaHAD and AFHQ at 512x512 using 1, 2, 4, or 8 GPUs.
| `paper1024`           | Reproduce results for MetFaces at 1024x1024 using 1, 2, 4, or 8 GPUs.
| `cifar`               | Reproduce results for CIFAR-10 (tuned configuration) using 1 or 2 GPUs.

The training configuration can be further customized with additional command line options:

* `--aug=noaug` disables ADA.
* `--cond=1` enables class-conditional training (requires a dataset with labels).
* `--mirror=1` amplifies the dataset with x-flips. Often beneficial, even with ADA.
* `--resume=ffhq1024 --snap=10` performs transfer learning from FFHQ trained at 1024x1024.
* `--resume=~/training-runs/<NAME>/network-snapshot-<INT>.pkl` resumes a previous training run.
* `--gamma=10` overrides R1 gamma. We recommend trying a couple of different values for each new dataset.
* `--aug=ada --target=0.7` adjusts ADA target value (default: 0.6).
* `--augpipe=blit` enables pixel blitting but disables all other augmentations.
* `--augpipe=bgcfnc` enables all available augmentations (blit, geom, color, filter, noise, cutout).

Please refer to [`python train.py --help`](./docs/train-help.txt) for the full list.

## Expected training time

The total training time depends heavily on resolution, number of GPUs, dataset, desired quality, and hyperparameters. The following table lists expected wallclock times to reach different points in the training, measured in thousands of real images shown to the discriminator ("kimg"):

| Resolution | GPUs | 1000 kimg | 25000 kimg | sec/kimg          | GPU mem | CPU mem
| :--------: | :--: | :-------: | :--------: | :---------------: | :-----: | :-----:
| 128x128    | 1    | 4h 05m    | 4d 06h     | 12.8&ndash;13.7   | 7.2 GB  | 3.9 GB
| 128x128    | 2    | 2h 06m    | 2d 04h     | 6.5&ndash;6.8     | 7.4 GB  | 7.9 GB
| 128x128    | 4    | 1h 20m    | 1d 09h     | 4.1&ndash;4.6     | 4.2 GB  | 16.3 GB
| 128x128    | 8    | 1h 13m    | 1d 06h     | 3.9&ndash;4.9     | 2.6 GB  | 31.9 GB
| 256x256    | 1    | 6h 36m    | 6d 21h     | 21.6&ndash;24.2   | 5.0 GB  | 4.5 GB
| 256x256    | 2    | 3h 27m    | 3d 14h     | 11.2&ndash;11.8   | 5.2 GB  | 9.0 GB
| 256x256    | 4    | 1h 45m    | 1d 20h     | 5.6&ndash;5.9     | 5.2 GB  | 17.8 GB
| 256x256    | 8    | 1h 24m    | 1d 11h     | 4.4&ndash;5.5     | 3.2 GB  | 34.7 GB
| 512x512    | 1    | 21h 03m   | 21d 22h    | 72.5&ndash;74.9   | 7.6 GB  | 5.0 GB
| 512x512    | 2    | 10h 59m   | 11d 10h    | 37.7&ndash;40.0   | 7.8 GB  | 9.8 GB
| 512x512    | 4    | 5h 29m    | 5d 17h     | 18.7&ndash;19.1   | 7.9 GB  | 17.7 GB
| 512x512    | 8    | 2h 48m    | 2d 22h     | 9.5&ndash;9.7     | 7.8 GB  | 38.2 GB
| 1024x1024  | 1    | 1d 20h    | 46d 03h    | 154.3&ndash;161.6 | 8.1 GB  | 5.3 GB
| 1024x1024  | 2    | 23h 09m   | 24d 02h    | 80.6&ndash;86.2   | 8.6 GB  | 11.9 GB
| 1024x1024  | 4    | 11h 36m   | 12d 02h    | 40.1&ndash;40.8   | 8.4 GB  | 21.9 GB
| 1024x1024  | 8    | 5h 54m    | 6d 03h     | 20.2&ndash;20.6   | 8.3 GB  | 44.7 GB

The above measurements were done using NVIDIA Tesla V100 GPUs with default settings (`--cfg=auto --aug=ada --metrics=fid50k_full`). "sec/kimg" shows the expected range of variation in raw training performance, as reported in `log.txt`. "GPU mem" and "CPU mem" show the highest observed memory consumption, excluding the peak at the beginning caused by `torch.backends.cudnn.benchmark`.

In typical cases, 25000 kimg or more is needed to reach convergence, but the results are already quite reasonable around 5000 kimg. 1000 kimg is often enough for transfer learning, which tends to converge significantly faster. The following figure shows example convergence curves for different datasets as a function of wallclock time, using the same settings as above:

![Training curves](./docs/stylegan2-ada-training-curves.png)

Note: `--cfg=auto` serves as a reasonable first guess for the hyperparameters but it does not necessarily lead to optimal results for a given dataset. For example, `--cfg=stylegan2` yields considerably better FID  for FFHQ-140k at 1024x1024 than illustrated above. We recommend trying out at least a few different values of `--gamma` for each new dataset.

## Quality metrics

By default, `train.py` automatically computes FID for each network pickle exported during training. We recommend inspecting `metric-fid50k_full.jsonl` (or TensorBoard) at regular intervals to monitor the training progress. When desired, the automatic computation can be disabled with `--metrics=none` to speed up the training slightly (3%&ndash;9%).

Additional quality metrics can also be computed after the training:

```.bash
# Previous training run: look up options automatically, save result to JSONL file.
python calc_metrics.py --metrics=pr50k3_full \
    --network=~/training-runs/00000-ffhq10k-res64-auto1/network-snapshot-000000.pkl

# Pre-trained network pickle: specify dataset explicitly, print result to stdout.
python calc_metrics.py --metrics=fid50k_full --data=~/datasets/ffhq.zip --mirror=1 \
    --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
```

The first example looks up the training configuration and performs the same operation as if `--metrics=pr50k3_full` had been specified during training. The second example downloads a pre-trained network pickle, in which case the values of `--mirror` and `--data` must be specified explicitly.

Note that many of the metrics have a significant one-off cost when calculating them for the first time for a new dataset (up to 30min). Also note that the evaluation is done using a different random seed each time, so the results will vary if the same metric is computed multiple times.

We employ the following metrics in the ADA paper. Execution time and GPU memory usage is reported for one NVIDIA Tesla V100 GPU at 1024x1024 resolution:

| Metric        | Time   | GPU mem | Description |
| :-----        | :----: | :-----: | :---------- |
| `fid50k_full` | 13 min | 1.8 GB  | Fr&eacute;chet inception distance<sup>[1]</sup> against the full dataset
| `kid50k_full` | 13 min | 1.8 GB  | Kernel inception distance<sup>[2]</sup> against the full dataset
| `pr50k3_full` | 13 min | 4.1 GB  | Precision and recall<sup>[3]</sup> againt the full dataset
| `is50k`       | 13 min | 1.8 GB  | Inception score<sup>[4]</sup> for CIFAR-10

In addition, the following metrics from the [StyleGAN](https://github.com/NVlabs/stylegan) and [StyleGAN2](https://github.com/NVlabs/stylegan2) papers are also supported:

| Metric        | Time   | GPU mem | Description |
| :------------ | :----: | :-----: | :---------- |
| `fid50k`      | 13 min | 1.8 GB  | Fr&eacute;chet inception distance against 50k real images
| `kid50k`      | 13 min | 1.8 GB  | Kernel inception distance against 50k real images
| `pr50k3`      | 13 min | 4.1 GB  | Precision and recall against 50k real images
| `ppl2_wend`   | 36 min | 2.4 GB  | Perceptual path length<sup>[5]</sup> in W, endpoints, full image
| `ppl_zfull`   | 36 min | 2.4 GB  | Perceptual path length in Z, full paths, cropped image
| `ppl_wfull`   | 36 min | 2.4 GB  | Perceptual path length in W, full paths, cropped image
| `ppl_zend`    | 36 min | 2.4 GB  | Perceptual path length in Z, endpoints, cropped image
| `ppl_wend`    | 36 min | 2.4 GB  | Perceptual path length in W, endpoints, cropped image

References:
1. [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500), Heusel et al. 2017
2. [Demystifying MMD GANs](https://arxiv.org/abs/1801.01401), Bi&nacute;kowski et al. 2018
3. [Improved Precision and Recall Metric for Assessing Generative Models](https://arxiv.org/abs/1904.06991), Kynk&auml;&auml;nniemi et al. 2019
4. [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498), Salimans et al. 2016
5. [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948), Karras et al. 2018

# StyleGAN2-ADA: Architecture & Implementation Guide

## Table of Contents
- [Overview](#overview)
- [Core Components](#core-components)
- [Generator Architecture](#generator-architecture)
- [Discriminator Architecture](#discriminator-architecture)
- [Loss Functions](#loss-functions)
- [Adaptive Discriminator Augmentation](#adaptive-discriminator-augmentation)
- [Training Pipeline](#training-pipeline)
- [Architecture Parameters](#architecture-parameters)
- [Data Flow](#data-flow)

---

## PERSONAL Overview

**StyleGAN2-ADA** (StyleGAN2 with Adaptive Discriminator Augmentation) is a state-of-the-art Generative Adversarial Network architecture designed for high-quality image generation, particularly effective when training with limited data.

### Key Capabilities
- Generate photorealistic images at resolutions up to 1024×1024
- Train effectively with datasets as small as a few thousand images
- Support transfer learning from pre-trained models
- Enable conditional generation based on class labels
- Achieve state-of-the-art FID scores with adaptive augmentation

### Main Innovation
**Adaptive Discriminator Augmentation (ADA)** dynamically adjusts data augmentation probability during training to prevent discriminator overfitting without requiring architectural modifications.

---

## Core Components

The StyleGAN2-ADA architecture consists of three main components:

```
┌─────────────────────────────────────────────────────────────┐
│                      StyleGAN2-ADA                          │
├─────────────────────────────────────────────────────────────┤
│  1. Generator (G)                                           │
│     ├── Mapping Network (z → W)                             │
│     └── Synthesis Network (W → Image)                       │
│                                                              │
│  2. Discriminator (D)                                       │
│     ├── Progressive Downsampling Blocks                     │
│     └── Minibatch Discrimination                            │
│                                                              │
│  3. Augmentation Pipeline (ADA)                             │
│     └── Adaptive probability adjustment                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Generator Architecture

The Generator transforms random noise into realistic images through a two-stage process.

### 1. Mapping Network

**Purpose:** Map random latent codes to an intermediate latent space with better properties for image generation.

```
Input:  z ∈ ℝ^512 (random Gaussian noise)
        c ∈ ℝ^c_dim (conditional label, optional)

Architecture:
┌─────────────────────────────────────────┐
│  Conditional Embedding (if c_dim > 0)   │
│  ├── Input: c (class labels)            │
│  └── Output: embed ∈ ℝ^512              │
├─────────────────────────────────────────┤
│  8× Fully Connected Layers              │
│  ├── Input dim: 512 (+ embed_dim)       │
│  ├── Hidden dim: 512                    │
│  ├── Activation: Leaky ReLU             │
│  ├── LR multiplier: 0.01                │
│  └── Output: W ∈ ℝ^512                  │
├─────────────────────────────────────────┤
│  Broadcasting                            │
│  └── W → [W₁, W₂, ..., W_n]            │
│      (one style vector per layer)       │
└─────────────────────────────────────────┘

Output: W^+ ∈ ℝ^(n×512) where n = num synthesis layers
```

**Key Features:**
- **2nd Moment Normalization:** Normalizes inputs for stable training
- **Moving Average Tracking:** Enables truncation trick for quality/diversity trade-off
- **Style Mixing:** During training, randomly mixes styles from two latent codes (prob=0.9)

### 2. Synthesis Network

**Purpose:** Generate images progressively from low to high resolution using style-based modulation.

```
Resolution Progression (for 1024×1024 output):
4 → 8 → 16 → 32 → 64 → 128 → 256 → 512 → 1024

Each Synthesis Block:
┌──────────────────────────────────────────────────────┐
│  Input Processing                                     │
│  ├── First block (4×4): Learned constant             │
│  └── Other blocks: Upsampled features (2×)           │
├──────────────────────────────────────────────────────┤
│  Synthesis Layer 1 (Modulated Conv)                  │
│  ├── Affine transform: W → style                     │
│  ├── Modulated 3×3 convolution                       │
│  │   ├── Weight modulation by style                  │
│  │   ├── Weight demodulation                         │
│  │   └── Fused modulated convolution                 │
│  ├── Noise injection (learned weight)                │
│  ├── Bias addition                                   │
│  └── Activation: Leaky ReLU                          │
├──────────────────────────────────────────────────────┤
│  Synthesis Layer 2 (Modulated Conv)                  │
│  └── [Same structure as Layer 1]                     │
├──────────────────────────────────────────────────────┤
│  ToRGB Layer (converts features to RGB)              │
│  ├── Style-modulated 1×1 convolution                 │
│  ├── No demodulation                                 │
│  └── Output: RGB image at current resolution         │
├──────────────────────────────────────────────────────┤
│  Skip Connection (progressive growth)                │
│  └── Upsample previous RGB + current RGB             │
└──────────────────────────────────────────────────────┘
```

**Channel Configuration** (channel_base=32768, channel_max=512):

| Resolution | Channels | Operations per Block |
|------------|----------|---------------------|
| 4×4        | 512      | 1 conv + toRGB      |
| 8×8        | 512      | 2 conv + toRGB      |
| 16×16      | 512      | 2 conv + toRGB      |
| 32×32      | 512      | 2 conv + toRGB      |
| 64×64      | 512      | 2 conv + toRGB      |
| 128×128    | 256      | 2 conv + toRGB      |
| 256×256    | 128      | 2 conv + toRGB      |
| 512×512    | 64       | 2 conv + toRGB      |
| 1024×1024  | 32       | 2 conv + toRGB      |

**Total W vectors needed:** ~26 (varies by resolution)

**Key Innovations:**

1. **Modulated Convolution:**
   ```
   w' = w · style     (per-sample weight modulation)
   w'' = w' / √(Σw'²) (weight demodulation)
   output = Conv(x, w'') + noise + bias
   ```

2. **Noise Injection:**
   - Per-pixel Gaussian noise scaled by learned parameter
   - Enables stochastic variation in generated images
   - Different noise per resolution level

3. **Progressive Growth:**
   - Image built from 4×4 upward
   - Each resolution adds detail
   - Skip connections merge RGB outputs

4. **Mixed Precision Training:**
   - FP16 for resolutions ≥ 128×128 (configurable)
   - FP32 for lower resolutions
   - Activation clamping prevents overflow

---

## Discriminator Architecture

The Discriminator evaluates image authenticity through progressive downsampling, mirroring the Generator in reverse.

```
Input: RGB Image (e.g., 1024×1024×3)

Architecture (ResNet-based):
┌──────────────────────────────────────────────────────┐
│  Progressive Downsampling                             │
│  Resolution: 1024 → 512 → 256 → 128 → 64 → 32 → 16 → 8 │
│                                                        │
│  Each Discriminator Block:                            │
│  ├── FromRGB (1×1 conv, for skip connections)        │
│  ├── Conv Block 1:                                    │
│  │   ├── 3×3 Convolution (same resolution)           │
│  │   └── Leaky ReLU activation                       │
│  ├── Conv Block 2:                                    │
│  │   ├── 3×3 Convolution (downsample 2×)             │
│  │   └── Leaky ReLU activation                       │
│  └── Skip Connection (ResNet):                        │
│      └── 1×1 Conv + downsample 2×                    │
└──────────────────────────────────────────────────────┘

Epilogue Block (4×4 resolution):
┌──────────────────────────────────────────────────────┐
│  Minibatch Standard Deviation Layer                   │
│  ├── Compute std across minibatch groups             │
│  ├── Group size: 4 (default)                         │
│  └── Append as extra channel                         │
├──────────────────────────────────────────────────────┤
│  3×3 Convolution (512 channels)                      │
│  └── Leaky ReLU                                      │
├──────────────────────────────────────────────────────┤
│  Flatten + Fully Connected                           │
│  ├── Input: 512 × 4 × 4 = 8192                       │
│  └── Output: 512 features                            │
├──────────────────────────────────────────────────────┤
│  Output Layer                                         │
│  ├── Unconditional: 1 logit (real/fake score)       │
│  └── Conditional: cmap_dim logits                    │
│                                                        │
│  Conditional Projection (if c_dim > 0):              │
│  └── logit = (output · c_embedding).sum()           │
└──────────────────────────────────────────────────────┘
```

**Channel Configuration** (reverse of Generator):

| Resolution | Input Channels | Output Channels |
|------------|----------------|-----------------|
| 1024×1024  | 3 (RGB)        | 32              |
| 512×512    | 32             | 64              |
| 256×256    | 64             | 128             |
| 128×128    | 128            | 256             |
| 64×64      | 256            | 512             |
| 32×32      | 512            | 512             |
| 16×16      | 512            | 512             |
| 8×8        | 512            | 512             |
| 4×4        | 512            | 512             |

**Key Features:**

1. **ResNet Skip Connections:**
   - Improves gradient flow
   - Enables deeper discriminator
   - Better feature learning

2. **Minibatch Standard Deviation:**
   - Encourages diversity in generated samples
   - Computes statistics across minibatch groups
   - Prevents mode collapse

3. **Progressive Analysis:**
   - Analyzes images at multiple scales
   - FromRGB at each resolution for skip architecture
   - Hierarchical feature extraction

4. **Mixed Precision:**
   - FP16 for high-resolution layers
   - FP32 for final classification layers

---

## Loss Functions

### Generator Loss

**Objective:** Fool the discriminator while maintaining high-quality, diverse outputs.

#### 1. Main Loss (Non-saturating GAN Loss)
```python
L_Gmain = 𝔼[softplus(-D(G(z)))]
        = 𝔼[-log(sigmoid(D(G(z))))]
```

**Interpretation:** Maximize discriminator's confidence that generated images are real.

#### 2. Path Length Regularization (PPL)
```python
L_Gpl = 𝔼[(||J_W^T y||₂ - a)²] × pl_weight

where:
- J_W = Jacobian of G(W) w.r.t. W
- y = random image-space direction
- a = exponential moving average of path lengths
- pl_weight = 2.0 (default)
```

**Purpose:**
- Encourages smooth, well-behaved latent space
- Improves interpolation quality
- Reduces artifacts

**Total Generator Loss:**
```python
L_G = L_Gmain + L_Gpl
```

### Discriminator Loss

**Objective:** Distinguish real from fake images while maintaining smooth decision boundaries.

#### 1. Main Loss (Non-saturating)
```python
L_Dmain = 𝔼[softplus(-D(x_real))] + 𝔼[softplus(D(G(z)))]
        = -𝔼[log(sigmoid(D(x_real)))] - 𝔼[log(1 - sigmoid(D(G(z))))]
```

**Interpretation:**
- Maximize confidence on real images
- Minimize confidence on fake images

#### 2. R1 Regularization (Gradient Penalty)
```python
L_Dr1 = (γ/2) × 𝔼[||∇_x D(x_real)||²]

where:
- γ (gamma) = regularization strength
- Default γ = 10.0 for 1024×1024
- Scales with resolution: γ = 0.0002 × (resolution²) / batch_size
```

**Purpose:**
- Penalizes large gradients on real data manifold
- Smooths discriminator decision boundaries
- Improves training stability

**Lazy Regularization:**
- R1 applied every 16 iterations (not every step)
- Reduces computational cost
- Effective regularization with lower overhead

**Total Discriminator Loss:**
```python
L_D = L_Dmain + L_Dr1
```

### Training Phases

Training alternates between four phases:

```python
1. Gmain: Update G to maximize D(G(z))
2. Greg:  Update G with path length regularization
3. Dmain: Update D to classify real/fake
4. Dreg:  Update D with R1 gradient penalty

Schedule:
- Gmain every iteration
- Greg every iteration (if pl_weight > 0)
- Dmain every iteration
- Dreg every 16 iterations (lazy regularization)
```

---

## Adaptive Discriminator Augmentation

**Problem:** Limited training data causes discriminator to memorize/overfit, leading to training collapse.

**Solution:** Adaptively augment real images fed to discriminator, adjusting augmentation strength based on overfitting indicators.

### Augmentation Pipeline

```
┌─────────────────────────────────────────────────────┐
│  1. Pixel Blitting                                  │
│     ├── Horizontal flip (xflip)                     │
│     ├── 90° rotations (rotate90)                    │
│     └── Integer translations (xint)                 │
├─────────────────────────────────────────────────────┤
│  2. Geometric Transformations                       │
│     ├── Isotropic scaling (scale)                   │
│     ├── Arbitrary rotation (rotate)                 │
│     ├── Anisotropic scaling (aniso)                 │
│     └── Fractional translation (xfrac)              │
├─────────────────────────────────────────────────────┤
│  3. Color Transformations                           │
│     ├── Brightness adjustment                       │
│     ├── Contrast adjustment                         │
│     ├── Luma flip                                   │
│     ├── Hue rotation                                │
│     └── Saturation adjustment                       │
├─────────────────────────────────────────────────────┤
│  4. Image Filtering                                 │
│     └── Wavelet-based filters (low/high pass)      │
├─────────────────────────────────────────────────────┤
│  5. Noise & Cutout                                  │
│     ├── Additive Gaussian noise                     │
│     └── Random rectangular cutouts                  │
└─────────────────────────────────────────────────────┘

Global Probability Multiplier: p ∈ [0, 1]
```

### Adaptive Mechanism

**Key Idea:** Monitor discriminator overfitting and adjust augmentation probability.

```python
Overfitting Indicator:
r_t = 𝔼[sign(D(x_real))]

Target value: r_target = 0.6
- If r_t > 0.6: D is overfitting → increase p
- If r_t < 0.6: D is underfitting → decrease p

Update Rule (every 4 training images):
p ← p + sign(r_t - r_target) × (batch_size / 500)

Constraints:
- p starts at 0
- p ∈ [0, 1]
- Separate tracking for real/fake statistics
```

**Benefits:**
- No manual tuning of augmentation strength
- Automatically adapts to dataset size
- Works with any GAN architecture
- Prevents discriminator overfitting without architectural changes

### Augmentation Configuration

**Common presets:**

```python
# No augmentation (sufficient data)
--aug=noaug

# Fixed augmentation
--aug=ada      # Adaptive (default)
--aug=fixed    # Fixed probability

# Custom augmentation probability
--augpipe=bgc  # Brightness, geometric, color
--ada-target=0.6  # Target overfitting metric
```

---

## Training Pipeline

### Optimization Configuration

```yaml
Generator Optimizer:
  type: Adam
  learning_rate: 0.002
  betas: [0.0, 0.99]
  eps: 1e-8
  
Discriminator Optimizer:
  type: Adam
  learning_rate: 0.002
  betas: [0.0, 0.99]
  eps: 1e-8

Exponential Moving Average (EMA):
  enabled: true
  half_life: 10k images
  rampup: 5% of training (for smooth start)
```

### Training Schedule

```yaml
Default Configuration (auto):
  total_kimg: 25000        # 25 million images
  batch_size: auto         # Based on resolution & GPUs
  metrics: [fid50k_full]   # Quality evaluation
  snap_interval: 50 ticks  # Checkpoint frequency
  
Resolution-Specific Adjustments:
  1024×1024:
    batch_size: 32 (8 GPUs)
    gamma: 10.0
    lr: 0.002
  
  512×512:
    batch_size: 64 (8 GPUs)
    gamma: 2.56
    lr: 0.0025
  
  256×256:
    batch_size: 64 (8 GPUs)
    gamma: 1.0
    lr: 0.0025
```

### Data Flow During Training

```
┌─────────────────────────────────────────────────────────┐
│  Iteration Step                                         │
├─────────────────────────────────────────────────────────┤
│  1. Sample Data                                         │
│     ├── z ~ N(0, I) ∈ ℝ^(batch×512)                    │
│     ├── c ~ dataset labels (if conditional)            │
│     └── x_real ~ dataset images                        │
├─────────────────────────────────────────────────────────┤
│  2. Generator Forward Pass                              │
│     ├── W ← MappingNetwork(z, c)                       │
│     ├── Apply style mixing (90% prob)                  │
│     └── x_fake ← SynthesisNetwork(W)                   │
├─────────────────────────────────────────────────────────┤
│  3. Apply Augmentation                                  │
│     ├── x_real_aug ← AugmentPipe(x_real, p)           │
│     └── x_fake_aug ← AugmentPipe(x_fake, p)           │
├─────────────────────────────────────────────────────────┤
│  4. Discriminator Evaluation                            │
│     ├── logits_real ← D(x_real_aug, c)                │
│     └── logits_fake ← D(x_fake_aug, c)                │
├─────────────────────────────────────────────────────────┤
│  5. Compute Losses                                      │
│     ├── L_Gmain = softplus(-logits_fake)              │
│     ├── L_Gpl = path_length_penalty (if Greg phase)   │
│     ├── L_Dmain = softplus(-logits_real) +            │
│     │             softplus(logits_fake)                │
│     └── L_Dr1 = gradient_penalty (if Dreg phase)      │
├─────────────────────────────────────────────────────────┤
│  6. Backward Pass & Updates                             │
│     ├── ∇L_G → Update G parameters                    │
│     ├── ∇L_D → Update D parameters                    │
│     └── Update G_ema (exponential moving average)     │
├─────────────────────────────────────────────────────────┤
│  7. Update Augmentation Probability                     │
│     └── p ← p + adjust_step (every 4 imgs)            │
├─────────────────────────────────────────────────────────┤
│  8. Logging & Checkpointing                             │
│     ├── Log training statistics                        │
│     ├── Evaluate metrics (periodic)                    │
│     └── Save checkpoints (every 50 ticks)             │
└─────────────────────────────────────────────────────────┘
```

---

## Architecture Parameters

### Default Configurations

#### Auto Configuration (Recommended)
```python
# Automatically selects parameters based on resolution and GPU count
--cfg=auto

Parameters selected dynamically:
- batch_size: Based on GPU memory
- gamma: Scales with resolution²
- learning_rate: Optimized per resolution
- channel_multiplier: Based on resolution
```

#### StyleGAN2 Configuration
```python
# Reproduce original StyleGAN2 results
--cfg=stylegan2

Fixed parameters:
  ref_gpus: 8
  batch_size: 32
  mbstd_group_size: 4
  channel_base: 32768 (fmaps=1)
  learning_rate: 0.002
  gamma: 10.0
  ema_kimg: 10
  mapping_layers: 8
```

#### Resolution-Specific Presets

```python
# 256×256 (paper256)
--cfg=paper256
  batch_size: 64
  gamma: 1.0
  channel_base: 16384 (fmaps=0.5)
  
# 512×512 (paper512)
--cfg=paper512
  batch_size: 64
  gamma: 2.56
  channel_base: 16384 (fmaps=0.5)
  
# 1024×1024 (paper1024)
--cfg=paper1024
  batch_size: 32
  gamma: 10.0
  channel_base: 32768 (fmaps=1)
```

### Complete Parameter Reference

```python
Generator Architecture:
  z_dim: 512                    # Input latent dimension
  w_dim: 512                    # Intermediate latent dimension
  c_dim: 0                      # Conditional dimension (0=unconditional)
  img_resolution: 1024          # Output resolution
  img_channels: 3               # RGB channels
  mapping_layers: 8             # Mapping network depth
  channel_base: 32768           # Base channel multiplier
  channel_max: 512              # Max channels per layer
  num_fp16_res: 4               # FP16 for top N resolutions
  conv_clamp: 256               # Activation clamping

Discriminator Architecture:
  c_dim: 0                      # Conditional dimension
  img_resolution: 1024          # Input resolution
  img_channels: 3               # RGB channels
  architecture: 'resnet'        # 'orig', 'skip', 'resnet'
  channel_base: 32768           # Base channel multiplier
  channel_max: 512              # Max channels per layer
  num_fp16_res: 4               # FP16 for top N resolutions
  conv_clamp: 256               # Activation clamping
  mbstd_group_size: 4           # Minibatch std group size

Loss Configuration:
  style_mixing_prob: 0.9        # Style mixing probability
  r1_gamma: 10.0                # R1 regularization weight
  pl_weight: 2.0                # Path length regularization
  pl_batch_shrink: 2            # PL batch size divisor
  pl_decay: 0.01                # PL moving average decay

Training Configuration:
  batch_size: 32                # Total batch size
  batch_gpu: 4                  # Per-GPU batch size
  total_kimg: 25000             # Training duration (kimgs)
  ema_kimg: 10                  # EMA half-life
  ema_rampup: 0.05              # EMA rampup ratio

Augmentation:
  ada_target: 0.6               # Target overfitting metric
  augment_p: 0.0                # Initial augmentation prob
  ada_kimg: 500                 # ADA update interval
```

---

## Data Flow

### Complete Pipeline Visualization

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING ITERATION                        │
└─────────────────────────────────────────────────────────────┘

INPUT PREPARATION
─────────────────
Real Images               Random Latent           Labels
x_real ~ Dataset         z ~ N(0,I)^512          c (optional)
     │                        │                        │
     │                   ┌────▼────┐                  │
     │                   │ Mapping  │◄─────────────────┘
     │                   │ Network  │
     │                   └────┬────┘
     │                        │
     │                   Style W ∈ ℝ^(26×512)
     │                        │
     │                   ┌────▼────────┐
     │                   │   Synthesis  │
     │                   │   Network    │
     │                   └────┬────────┘
     │                        │
     │                   Generated Images
     │                   x_fake ∈ ℝ^(3×1024×1024)
     │                        │
     └────────┬───────────────┘
              │
         ┌────▼────┐
         │   ADA   │  Adaptive Augmentation
         │Pipeline │  (probability p)
         └────┬────┘
              │
    ┌─────────┴──────────┐
    │                    │
x_real_aug          x_fake_aug
    │                    │
    └─────────┬──────────┘
              │
         ┌────▼──────┐
         │Discrimin- │
         │   ator    │
         └────┬──────┘
              │
    ┌─────────┴──────────┐
    │                    │
logits_real         logits_fake
    │                    │
    └─────────┬──────────┘
              │
         ┌────▼────┐
         │ Losses  │
         │ L_G, L_D│
         └────┬────┘
              │
         ┌────▼────┐
         │Backward │
         │  Pass   │
         └────┬────┘
              │
         ┌────▼────┐
         │Optimizer│
         │ Updates │
         └─────────┘

GENERATOR PATH:  z → W → x_fake → D → L_G → ∇_G
DISCRIMINATOR:   x → aug → D → L_D → ∇_D
```

### Memory Layout

```
Tensor Shapes (batch_size=4, resolution=1024):

Latents:
  z:        [4, 512]              ~8 KB
  c:        [4, c_dim]            variable
  W:        [4, 26, 512]          ~208 KB

Generated Images:
  x_fake:   [4, 3, 1024, 1024]   ~48 MB (FP32)
  
Intermediate Features (synthesis):
  4×4:      [4, 512, 4, 4]       ~32 KB
  8×8:      [4, 512, 8, 8]       ~128 KB
  16×16:    [4, 512, 16, 16]     ~512 KB
  32×32:    [4, 512, 32, 32]     ~2 MB
  64×64:    [4, 512, 64, 64]     ~8 MB
  128×128:  [4, 256, 128, 128]   ~8 MB
  256×256:  [4, 128, 256, 256]   ~8 MB
  512×512:  [4, 64, 512, 512]    ~8 MB
  1024×1024:[4, 32, 1024, 1024]  ~8 MB

Discriminator Features (reverse progression)
  Similar memory pattern in reverse

Total GPU Memory Usage:
  ~12-15 GB per GPU (1024×1024, mixed precision)
```

---

## File Structure & Responsibilities

```
stylegan2-ada-pytorch/
├── train.py                    # Main training script
├── generate.py                 # Image generation
├── projector.py                # Project images to latent space
├── style_mixing.py             # Style mixing visualization
├── calc_metrics.py             # Compute quality metrics
├── dataset_tool.py             # Dataset preparation
│
├── training/
│   ├── networks.py             # G & D architectures
│   │   ├── Generator           # Complete generator
│   │   │   ├── MappingNetwork  # z → W
│   │   │   └── SynthesisNetwork# W → image
│   │   ├── Discriminator       # Complete discriminator
│   │   │   ├── DiscriminatorBlock
│   │   │   └── DiscriminatorEpilogue
│   │   └── Layers              # Building blocks
│   │       ├── SynthesisLayer  # Modulated conv
│   │       ├── ToRGBLayer      # Feature to RGB
│   │       ├── Conv2dLayer     # Standard conv
│   │       └── FullyConnectedLayer
│   │
│   ├── loss.py                 # Loss functions
│   │   └── StyleGAN2Loss       # Main loss class
│   │       ├── run_G()         # Generator forward
│   │       ├── run_D()         # Discriminator forward
│   │       └── accumulate_gradients()
│   │
│   ├── training_loop.py        # Training orchestration
│   │   └── training_loop()     # Main training function
│   │
│   ├── augment.py              # ADA implementation
│   │   └── AugmentPipe         # Augmentation pipeline
│   │
│   └── dataset.py              # Data loading
│       └── ImageFolderDataset  # Dataset class
│
├── torch_utils/                # PyTorch utilities
│   ├── ops/                    # Custom CUDA operations
│   │   ├── upfirdn2d.py       # Upsampling/downsampling
│   │   ├── bias_act.py        # Fused bias+activation
│   │   ├── conv2d_*.py        # Specialized convolutions
│   │   └── fma.py             # Fused multiply-add
│   ├── persistence.py          # Model serialization
│   ├── misc.py                 # Helper functions
│   └── training_stats.py       # Statistics tracking
│
├── metrics/                    # Quality metrics
│   ├── frechet_inception_distance.py
│   ├── kernel_inception_distance.py
│   ├── precision_recall.py
│   └── perceptual_path_length.py
│
└── dnnlib/                     # Deep learning library
    ├── util.py                 # Utilities
    └── __init__.py
```

---

## Performance Optimizations

### 1. Custom CUDA Kernels

```python
# Fused operations for performance
torch_utils/ops/
├── upfirdn2d       # Efficient resampling
├── bias_act        # Fused bias + activation
├── conv2d_gradfix  # Gradient fixes for conv
└── grid_sample_gradfix  # Grid sampling gradients
```

### 2. Mixed Precision Training

```python
# Automatic mixed precision
num_fp16_res: 4  # Top 4 resolutions use FP16

Benefits:
- 30-50% faster training
- 40% less GPU memory
- Minimal quality impact
```

### 3. Distributed Training

```python
# Multi-GPU support
torch.nn.parallel.DistributedDataParallel

# Efficient synchronization
- Gradient synchronization controlled per phase
- Selective sync for main/regularization phases
```

### 4. Lazy Regularization

```python
# R1 regularization every 16 iterations
if phase in ['Dreg', 'Dboth']:
    apply_r1_penalty()

Speedup: ~15-20% faster training
```

---

## Key Takeaways

### Architecture Strengths

1. **Style-Based Generation**
   - Fine-grained control over generated features
   - Excellent interpolation properties
   - Disentangled latent space

2. **Progressive Synthesis**
   - Stable high-resolution training
   - Hierarchical feature learning
   - Efficient multi-scale generation

3. **Adaptive Augmentation**
   - Enables training with limited data
   - No manual tuning required
   - Architecture-agnostic

4. **Robust Regularization**
   - Path length regularization for smooth latent space
   - R1 penalty for stable discriminator
   - Lazy regularization for efficiency

### Computational Requirements

```
Recommended Setup:
- 8× NVIDIA V100 GPUs (32GB)
- 1024×1024 resolution
- Batch size: 32 total (4 per GPU)
- Training time: ~7 days for 25M images

Minimum Setup:
- 1× GPU with 12GB+ VRAM
- Lower resolution (256×256 or 512×512)
- Smaller batch size
- Longer training time
```

### Common Use Cases

1. **High-Quality Image Generation**
   - Face generation (FFHQ)
   - Animal generation (AFHQ)
   - Artistic generation (MetFaces, WikiArt)

2. **Transfer Learning**
   - Fine-tune pre-trained models
   - Domain adaptation
   - Style transfer

3. **Research Applications**
   - GAN inversion
   - Image editing
   - Latent space exploration
   - Few-shot learning

---

## References

**Original Papers:**
- StyleGAN2-ADA: [Training Generative Adversarial Networks with Limited Data](https://arxiv.org/abs/2006.06676) (Karras et al., 2020)
- StyleGAN2: [Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) (Karras et al., 2020)
- StyleGAN: [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948) (Karras et al., 2019)

**Implementation:**
- Official PyTorch implementation: [NVlabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)
- Original TensorFlow version: [NVlabs/stylegan2-ada](https://github.com/NVlabs/stylegan2-ada)

---

*This documentation provides a comprehensive overview of the StyleGAN2-ADA architecture. For detailed implementation specifics, refer to the source code in the repository.*
