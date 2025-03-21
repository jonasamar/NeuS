# NeuS
*Part of this README and code are adapted from [NeuS](https://lingjie0206.github.io/papers/NeuS/) ; [IDR](https://github.com/lioryariv/idr) ; [NeRF](https://github.com/bmild/nerf) and [PS-NeuS](https://merl.com/research/highlights/ps-neus).*

![](./static/intro_1_compressed.gif)
![](./static/intro_2_compressed.gif)

This is forked and adapted from the official repo for the implementation of **NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction**.

## [Project page](https://lingjie0206.github.io/papers/NeuS/) |  [Paper](https://arxiv.org/abs/2106.10689) | [Data](https://www.dropbox.com/sh/w0y8bbdmxzik3uk/AAAaZffBiJevxQzRskoOYcyja?dl=0)

## Usage

#### Data Convention
The data is organized as follows:

```
<case_name>
|-- cameras_xxx.npz    # camera parameters
|-- image
    |-- 000.png        # target image for each view
    |-- 001.png
    ...
|-- mask
    |-- 000.png        # target mask each view (For unmasked setting, set all pixels as 255)
    |-- 001.png
    ...
```

Here the `cameras_xxx.npz` follows the data format in [IDR](https://github.com/lioryariv/idr/blob/main/DATA_CONVENTION.md), where `world_mat_xx` denotes the world to image projection matrix, and `scale_mat_xx` denotes the normalization matrix.

### Setup

Clone this repository

Create a python environment with **version 3.8** !

```shell
git clone https://github.com/Totoro97/NeuS.git
cd NeuS
pip install -r requirements.txt
```

`torch` version `>2.xxx`can also be installed and works.

<details>
  <summary> Dependencies (click to expand) </summary>

  - torch==1.8.0 # Tested with version 2.XXX and it works !
  - opencv_python==4.5.2.52
  - trimesh==3.9.8 
  - numpy==1.19.2
  - pyhocon==0.3.57
  - icecream==2.1.0
  - tqdm==4.50.2
  - scipy==1.7.0
  - PyMCubes==0.1.2

</details>

### Running

- **Train on DTU data (PROJECT)**

After running the `download_data.sh`bash file, you can run the following instruction:

```shell
python exp_runner.py --mode train --conf ./confs/configuration_file.conf --case scan<id>
```

Several configuration files are provided in the `confs` folder.

- **Extract surface from trained model** 

```shell
python exp_runner.py --mode validate_mesh --conf <config_file> --case <case_name> --is_continue # use latest checkpoint
```

The corresponding mesh can be found in `exp/<case_name>/<exp_name>/meshes/<iter_steps>.ply`.

- **View interpolation**

```shell
python exp_runner.py --mode interpolate_<img_idx_0>_<img_idx_1> --conf <config_file> --case <case_name> --is_continue # use latest checkpoint
```

The corresponding image set of view interpolation can be found in `exp/<case_name>/<exp_name>/render/`.

### Hardware limitations

- **GPU memory**

The GPU memory requirement highly depends on the selected batch-size. With default batch-size of 512 (as in the paper), we recommend 12GB. With batch-size of 256, 8GB sshould be enough; while 16GB are needed for a batch size of 1024 (as in `NPM3D_DTU.conf`).

- **Training time**

With the default model and hyperparameters, around 15 hours are needed to train one mesh on a single NVIDIA 2080Ti / 4070S GPU (300k epochs). With our smaller implementations combined with diminished batch sizes and less epochs, this can be reduced to around 45 mins on the same kind of GPUs (100k epochs ; as in `NPM3D_DTU_small-base.conf`).