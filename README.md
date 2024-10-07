# BiFaceGAN: Bimodal Synthesis and Annotation of Ocular Images 


![Teaser image](./docs/BiFaceGAN_pipeline.png)

**BiFaceGAN: Bimodal Face Image Synthesis**<br>
Darian Tomašević, Peter Peer, Vitomir Štruc<br>
[BiFaceGAN paper](https://link.springer.com/chapter/10.1007/978-981-97-2059-0_11)<br>

Abstract : *Modern face recognition and segmentation systems, like all deep learning approaches, are reliant on large-scale annotated datasets to achieve competitive performance. However, gathering biometric data often raises privacy concerns and presents a labor-intensive and time-consuming task. Researchers are currently also exploring the use of multispectral data to improve existing solutions, limited to the visible spectrum. Unfortunately, the collection of suitable data is even more difficult, especially if aligned images are required. To address the outlined issues, we present a novel synthesis framework, named BiFaceGAN, capable of producing privacy-preserving large-scale synthetic datasets of photorealistic face images, in the visible and the near-infrared spectrum, along with corresponding ground truth pixel-level annotations. The proposed framework leverages an innovative Dual-Branch Style-based Generative Adversarial Network (DB-StyleGAN) to generate per-pixel aligned bimodal images, followed by an ArcFace Privacy Filter (APF) that ensures the removal of privacy-breaching images. Furthermore, we also implement a Semantic Mask Generator (SMG) that produces reference ground truth segmentation masks of the synthetic data, based on the latent representations inside the synthesis model and only a handful of manually labeled examples. We evaluate the quality of generated images and annotations through a series of experiments and analyze the benefits of generating bimodal data with a single network. We also show that privacy-preserving data filtering does not notably degrade the image quality of produced datasets. In addition, we demonstrate that the generated data can be employed to train highly successful deep segmentation models, which can generalize well to other real-world datasets.*

# Release Notes: 

The BiFaceGAN PyTorch framework allows for the generation of high-quality privacy-preserving face images, matching in the visible and the near-infrared spectrum, along with corresponding ground-truth segmentation masks. The framework is made up of a Dual-Branch StyleGAN2, based on the [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) implementation, a Semantic Mask Generator,  based on the [DatasetGAN](https://github.com/nv-tlabs/datasetGAN_release) implementation, and an auxiliary ArcFace Privacy Filter, based on the pre-trained [ArcFace recognition model](https://github.com/chenggongliang/arcface).

This repository follows the [Nvidia Source Code License](https://nvlabs.github.io/stylegan2-ada-pytorch/license.html).

# Requirements and Setup:

* Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
* 1&ndash;8 high-end NVIDIA GPUs with at least 12 GB of memory. We have tested our implementation on a NVIDIA RTX 3060 GPU and a NVIDIA RTX 3090 GPU. Parallelization across multiple GPUs are also supported for training the DB-StyleGAN2 network.
* We highly recommend using Docker to setup the environment. Please use the [provided Dockerfile](./Dockerfile) to build an image with the required library dependencies. (The Docker image requires NVIDIA driver release `r455.23` or later.)
* Otherwise the requirements remain the same as in  [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch). These being 64-bit Python 3.7, PyTorch 1.7.1, and CUDA toolkit 11.0 or later. Use at least version 11.1 if running on RTX 3090. Check the linked repository if you are having any problems.


How to build Docker environment: 
```.bash
docker build --tag sg2ada:latest .
```

# How to Run (using Docker): 
To run the BiFaceGAN framework use the `main_BiOcularGAN.ipynb` Jupyter Notebook, or follow the below steps:

## Step 1. Train the Dual-Branch StyleGAN2 network: 
```.bash
./docker_run.sh python train_DB_StyleGAN2.py --cfg="auto" --snap=20 --gpus=1 --mirror=1 --GPU_DEVICE_NUMBER=0 --batch=12  --data="DATASETS/Tufts_256/train/images" --resume="ffhq256"  --outdir="EXPERIMENTS/DB_SG2/Training_tufts_256"
```

Here the `--data` argument should point to a directory of `RGB` images, structured similarly to the example in the `EXPERIMENTS` directory. It should have a neighbour directory filled with corresponding `NIR` (grayscale) images.

For details on other arguments and the configurations check the [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) documentation.

## Step 1.5 Prepare for Style Interpreter steps:
First save the final trained StyleGAN2 model under `EXPERIMENTS/checkpoints/`.
Then make the required directory in `EXPERIMENTS/interpreter/` with the two `.json` configuration files (`generate.json`, `train_datagan.json`). 
To construct these use the available template in `EXPERIMENTS/interpreter/Training_tufts_256/`. For more details check the [DatasetGAN](https://github.com/nv-tlabs/datasetGAN_release) documentation.

## Step 2. Generate examples of training image pairs: <br>
```.bash
docker_run.sh python make_training_data_DB_SG2.py --exp="EXPERIMENTS/interpreter/Training_tufts_256/generate.json" --sv_path="EXPERIMENTS/interpreter/Training_tufts_256
``` 
This generates image pairs in the `images_to_annotate` directory.

## Step 2.1. Annotate:
Annotate the desired number of images (8 in our experiments) with the desired number of regions (4 and 10 in our experiments). For this you can use [GIMP](https://www.gimp.org/) or any other software. Save the annotations and original images to the `images_GIMP` directory as seen in the example.

## Step 2.2. Preprocess annotations:
Preprocess the annotations so that the classes are in the range `{0, 1, 2, ... num_classes}` and that the annotations are saved as `.npy` files. For this, you can also use the `preprocess_annotated_images_and_latents.ipynb` Jupyter Notebook, however, make sure that the values correspond to your annotations.

## Step 3. Train the Style Interpreter:
```.bash
docker_run.sh python train_interpreter_DB_SG2.py --exp "EXPERIMENTS/interpreter/Training_tufts_256/train_datagan.json"
```

## Step 4. Generate dataset of matching RGB and NIR images with corresponding annotations:
```bash
docker_run.sh python train_interpreter_DB_SG2.py --generate_data True --num_sample=500 --exp "EXPERIMENTS/interpreter/Training_tufts_256/train_datagan.json"  --resume "EXPERIMENTS/interpreter/Training_tufts_256" 
```


## Notes:
* Pre-trained networks are stored as `*.pkl` files. These contain the Generator `'G'` and Discriminator `'D'`, as well as `'G_ema'`, represents a moving average of the generator weights over several training steps. The generator consists of two submodules, `G.mapping` and `G.synthesis`, that can be executed separately. They also support various additional options, such as truncation. For further examples, check [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch).
* To use BiFaceGAN without Docker, simply remove `docker_run.sh` from the above commands. 
* An example experiment is available under `EXPERIMENTS/interpreter/Training_tufts_256/`

## License

Copyright &copy; 2021, NVIDIA Corporation. All rights reserved.

This work is made available under the [Nvidia Source Code License](https://nvlabs.github.io/stylegan2-ada-pytorch/license.html).

## Citation

If you use code or results from this repository, please cite the following publication:

```
REFERENCE_TO_BE_ADDED
```

## Acknowledgements

Supported in parts by the Slovenian Research Agency ARRS through the Research Programmes P2-0250(B) "Metrology and Biometric System" and P2--0214 (A) “Computer Vision”, the ARRS Project J2-2501(A) "DeepBeauty" and the ARRS junior researcher program.

<img src="./docs/ARRS_Logo_2016_eng.png" alt="drawing" width="300"/>
