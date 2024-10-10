# Code for fine-tuning Segment Anything Model 2 (SAM2) For containers, fill level
 This is code for fine tuning SAM2 for semantic  segmentation of Vessel/Containers fill-level/liquid-level and transparent regions. This is mainly focused on transparent vessels, liquids and materials in chemistry labs, kitchens and everyday life and is part of a project to test top models on lab and materials related tasks.

# Requirements:
The requirements are the same as the SAM 2  main repository.
Installations are the same as [SAM2](https://dl.fbaipublicfiles.com/segment_anything_2)  repository.
Download SAM2 checkpoint from [[1](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt) or [2](https://github.com/facebookresearch/sam2?tab=readme-ov-file#download-checkpoints)..
(note by default the code load the small model you can also download other models from [here])

# Running on images:
Train or download a pretrained model from [1] or [2].
Download SAM2 checkpoint from [1](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt) or [2](https://github.com/facebookresearch/sam2?tab=readme-ov-file#download-checkpoints).
(note by default the code uses the small model but you can change this).

In: Run_On_Folder.py

Set parameters:

--in_dir = folder with image to segment

--model_path = path to trained model download from [1](2)

--sam2_checkpoint = path to sam2 standard checkpoint download from [1](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt) or [2](https://github.com/facebookresearch/sam2?tab=readme-ov-file#download-checkpoints).

--outdir  = output-dir were annotations will be saved (annotations will also be displayed on screen)



# Training:
## For training first download (click to download):
[LabPics1 dataset](https://zenodo.org/records/3697452/files/LabPicsV1.zip?download=1)

[LabPics2 Chemistry](https://zenodo.org/records/4736111/files/LabPicsChemistry.zip?download=1)

[LabPics2 Medical](https://zenodo.org/records/4736111/files/LabPicsMedical.zip?download=1).
## Optional Datasets:

Subset of the of the Coco dataset containing vessels  such as glasses bottles [available here](https://icedrive.net/0/c5NqTAyVdn), note this is a subset of the COCO dataset generated using the procedure described [here](https://github.com/sagieppel/Detecting-and-segmenting-and-classifying-materials-inside-vessels-in-images-using-convolutional-net?tab=readme-ov-file#converting-coco-dataset-into-training-data).

[Trans10k dataset for transparent vessels](https://xieenze.github.io/projects/TransLAB/TransLAB.html)


## Run training: TRAIN_Vessel_Filled.py
In TRAIN_Vessel_Filled.py set parameters:

--sam2_checkpoint = path to sam2 standard checkpoint download from [1](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt) or [2](https://github.com/facebookresearch/sam2?tab=readme-ov-file#download-checkpoints).

--labpics1_dir="path to dataset dir for labpics1 simple train set"

--labpics2_chemistry_dir="path to dataset dir for labpics2 chemistry train set"

--labpics2_medical_dir'="path to dataset dir for labpics2 medical train set")

--coco_vessel_dir=path subset of coco dataset containing only vessels this is optional leave empty if unused ("")


--trans10k_dir = path to trans10k (optional leave empty if unused(")

--log_dir = path to log folder where trained model will be saved

## What does it do

Running this will train the model and save the weight in the log dir.


# Evaluation on LabPics dataset: Evaluate_Vessel_Filled.py

Evaluation of the train model on lab pics dataset can be done using:
Evaluate_Vessel_Filled.py















# SAM 2: Segment Anything in Images and Videos (from original repository)

**[AI at Meta, FAIR](https://ai.meta.com/research/)**

[Nikhila Ravi](https://nikhilaravi.com/), [Valentin Gabeur](https://gabeur.github.io/), [Yuan-Ting Hu](https://scholar.google.com/citations?user=E8DVVYQAAAAJ&hl=en), [Ronghang Hu](https://ronghanghu.com/), [Chaitanya Ryali](https://scholar.google.com/citations?user=4LWx24UAAAAJ&hl=en), [Tengyu Ma](https://scholar.google.com/citations?user=VeTSl0wAAAAJ&hl=en), [Haitham Khedr](https://hkhedr.com/), [Roman Rädle](https://scholar.google.de/citations?user=Tpt57v0AAAAJ&hl=en), [Chloe Rolland](https://scholar.google.com/citations?hl=fr&user=n-SnMhoAAAAJ), [Laura Gustafson](https://scholar.google.com/citations?user=c8IpF9gAAAAJ&hl=en), [Eric Mintun](https://ericmintun.github.io/), [Junting Pan](https://junting.github.io/), [Kalyan Vasudev Alwala](https://scholar.google.co.in/citations?user=m34oaWEAAAAJ&hl=en), [Nicolas Carion](https://www.nicolascarion.com/), [Chao-Yuan Wu](https://chaoyuan.org/), [Ross Girshick](https://www.rossgirshick.info/), [Piotr Dollár](https://pdollar.github.io/), [Christoph Feichtenhofer](https://feichtenhofer.github.io/)

[[`Paper`](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)] [[`Project`](https://ai.meta.com/sam2)] [[`Demo`](https://sam2.metademolab.com/)] [[`Dataset`](https://ai.meta.com/datasets/segment-anything-video)] [[`Blog`](https://ai.meta.com/blog/segment-anything-2)] [[`BibTeX`](#citing-sam-2)]

![SAM 2 architecture](assets/model_diagram.png?raw=true)

**Segment Anything Model 2 (SAM 2)** is a foundation model towards solving promptable visual segmentation in images and videos. We extend SAM to video by considering images as a video with a single frame. The model design is a simple transformer architecture with streaming memory for real-time video processing. We build a model-in-the-loop data engine, which improves model and data via user interaction, to collect [**our SA-V dataset**](https://ai.meta.com/datasets/segment-anything-video), the largest video segmentation dataset to date. SAM 2 trained on our data provides strong performance across a wide range of tasks and visual domains.

![SA-V dataset](assets/sa_v_dataset.jpg?raw=true)

## Installation

Please install SAM 2 on a GPU machine using:

```bash
git clone git@github.com:facebookresearch/segment-anything-2.git

cd segment-anything-2; pip install -e .
```

To use the SAM 2 predictor and run the example notebooks, `jupyter` and `matplotlib` are required and can be installed by:

```bash
pip install -e ".[demo]"
```

## Getting Started

### Download Checkpoints

First, we need to download a model checkpoint. All the model checkpoints can be downloaded by running:

```bash
cd checkpoints
./download_ckpts.sh
```

or individually from:

- [sam2_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt)
- [sam2_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt)
- [sam2_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt)
- [sam2_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt)

Then SAM 2 can be used in a few lines as follows for image and video prediction.

### Image prediction

SAM 2 has all the capabilities of [SAM](https://github.com/facebookresearch/segment-anything) on static images, and we provide image prediction APIs that closely resemble SAM for image use cases. The `SAM2ImagePredictor` class has an easy interface for image prompting.

```python
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(<your_image>)
    masks, _, _ = predictor.predict(<input_prompts>)
```

Please refer to the examples in [image_predictor_example.ipynb](./notebooks/image_predictor_example.ipynb) for static image use cases.

SAM 2 also supports automatic mask generation on images just like SAM. Please see [automatic_mask_generator_example.ipynb](./notebooks/automatic_mask_generator_example.ipynb) for automatic mask generation in images.

### Video prediction

For promptable segmentation and tracking in videos, we provide a video predictor with APIs for example to add prompts and propagate masklets throughout a video. SAM 2 supports video inference on multiple objects and uses an inference state to keep track of the interactions in each video.

```python
import torch
from sam2.build_sam import build_sam2_video_predictor

checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(<your_video>)

    # add new prompts and instantly get the output on the same frame
    frame_idx, object_ids, masks = predictor.add_new_points(state, <your prompts>):

    # propagate the prompts to get masklets throughout the video
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        ...
```

Please refer to the examples in [video_predictor_example.ipynb](./notebooks/video_predictor_example.ipynb) for details on how to add prompts, make refinements, and track multiple objects in videos.

## Model Description

|      **Model**       | **Size (M)** |    **Speed (FPS)**     | **SA-V test (J&F)** | **MOSE val (J&F)** | **LVOS v2 (J&F)** |
| :------------------: | :----------: | :--------------------: | :-----------------: | :----------------: | :---------------: |
|   sam2_hiera_tiny    |     38.9     |          47.2          |        75.0         |        70.9        |       75.3        |
|   sam2_hiera_small   |      46      | 43.3 (53.0 compiled\*) |        74.9         |        71.5        |       76.4        |
| sam2_hiera_base_plus |     80.8     | 34.8 (43.8 compiled\*) |        74.7         |        72.8        |       75.8        |
|   sam2_hiera_large   |    224.4     | 24.2 (30.2 compiled\*) |        76.0         |        74.6        |       79.8        |

\* Compile the model by setting `compile_image_encoder: True` in the config.

## Segment Anything Video Dataset

See [sav_dataset/README.md](sav_dataset/README.md) for details.

## License

The models are licensed under the [Apache 2.0 license](./LICENSE). Please refer to our research paper for more details on the models.

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Contributors

The SAM 2 project was made possible with the help of many contributors (alphabetical):

Karen Bergan, Daniel Bolya, Alex Bosenberg, Kai Brown, Vispi Cassod, Christopher Chedeau, Ida Cheng, Luc Dahlin, Shoubhik Debnath, Rene Martinez Doehner, Grant Gardner, Sahir Gomez, Rishi Godugu, Baishan Guo, Caleb Ho, Andrew Huang, Somya Jain, Bob Kamma, Amanda Kallet, Jake Kinney, Alexander Kirillov, Shiva Koduvayur, Devansh Kukreja, Robert Kuo, Aohan Lin, Parth Malani, Jitendra Malik, Mallika Malhotra, Miguel Martin, Alexander Miller, Sasha Mitts, William Ngan, George Orlin, Joelle Pineau, Kate Saenko, Rodrick Shepard, Azita Shokrpour, David Soofian, Jonathan Torres, Jenny Truong, Sagar Vaze, Meng Wang, Claudette Ward, Pengchuan Zhang.

Third-party code: we use a GPU-based connected component algorithm adapted from [`cc_torch`](https://github.com/zsef123/Connected_components_PyTorch) (with its license in [`LICENSE_cctorch`](./LICENSE_cctorch)) as an optional post-processing step for the mask predictions.

## Citing SAM 2

If you use SAM 2 or the SA-V dataset in your research, please use the following BibTeX entry.

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint},
  year={2024}
}
```
