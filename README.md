# pedestrian_traffic_assistant

Repository for a project focusing on object detection, using computer vision, with the intuit of helping blind or visually impaired people, cross traffic road

# Getting Started

To get started, clone this repository using ssh, with the command:

```bash
git clone git@github.com:TheGoncaloSilva/pedestrian_traffic_assistant.git
```

## Dependencies

Python is the chosen language to train and evaluate the model, so first, make sure you have it installed. Preferably, a version between `3.8 - 3.11`. Version `3.12+` aren't advisable, at the moment.

It's highly advisable, to first start a virtual environment using the command:
```bash
python3 -m venv venv
source venv/bin/activate # used to activate the virtual environment, every time a new shell is created
```

To install the dependencies for this project, you just need to use the provided `requirements` file:

```bash
pip install -r requirements.txt
```

## Dataset Transformation

The **YOLO** format requires that paths specified in the `data.yaml` file are absolute, so it's necessary to navigate to the desired dataset folder, and replace the `{PATH}` with the absolute path to the dataset folder. The following example shows the necessary lines to change:

```yaml
# Better to use absolute links than relative
train: {PATH}/train/images
val: {PATH}/valid/images
test: {PATH}/test/images
```

To know what the absolute path is, you can use the `pwd` command, in linux, or `cd` in windows.

# Dataset

Mixed dataset obtained [here](https://universe.roboflow.com/chanyoung/pedestrian-light-crosswalk), with the distribution of images and annotations as follows:

| Use   | nº images | %   |
|-------|-----------|-----|
| Train | 1374      | 70  |
| Valid | 389       | 20  |
| Test  | 197       | 10  |
| Total | 1960      | 100 |

**Note:** Guarantee that `test` images are unique and not repeated

# Training tips

To train the model in a terminal only linux environment, you can use the `screen` command to create a new window, and run the training script in the background. This way, you can close the terminal, and the training will continue to run. For that, the following commands can be used:

```bash
screen -S {window_name}
screen -ad # Run detached
screen -ls # List windows
screen -r {window_id}
```

# Usage of transformer model

* Train
```sh
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=9909 tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml --seed=0 &> log.txt 2>&1
```

* Test
```sh
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=9909 tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -r output/rtdetrv2_r18vd_120e_coco/best.pth --test-only &> log_test.txt 2>&1
```

* Extract to onnx format (first install `onnx`, `onnxruntime` and `onnxruntime-gpu` packages):
```sh
python tools/export_onnx.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -r output/rtdetrv2_r18vd_120e_coco/best.pth --check
```

* Inference
```sh
python tools/inference.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -r output/rtdetrv2_r18vd_120e_coco/best.pth --img-root data/coco/val2017 --save-dir output/rtdetrv2_r18vd_120e_coco/val2017
``` 

# Some analysis

```
Prediction Stats for yolo11n_300_100_16_AdamW_0.0001_0.0001_0:
Inference Time :
  Mean: 40.35 ms
  95% Confidence Interval: ±1.69 ms
----------------------------------------
Total Time for Multiple Objects:
  Mean: 41.42 ms
  95% Confidence Interval: ±1.69 ms
----------------------------------------
```

```
Prediction Stats for yolo11l_300_100_16_AdamW_0.0001_0.0001_0:
Inference Time :
  Mean: 254.80 ms
  95% Confidence Interval: ±7.67 ms
----------------------------------------
Total Time for Multiple Objects:
  Mean: 257.11 ms
  95% Confidence Interval: ±8.35 ms
----------------------------------------
```

```
Prediction Stats for yolo11n_300_5_16:
Inference Time :
  Mean: 38.68 ms
  95% Confidence Interval: ±1.21 ms
----------------------------------------
Total Time for Multiple Objects:
  Mean: 40.40 ms
  95% Confidence Interval: ±1.15 ms
----------------------------------------
```

```
Prediction Stats for yolo11n_300_100_16:
Inference Time :
  Mean: 38.18 ms
  95% Confidence Interval: ±1.56 ms
----------------------------------------
Total Time for Multiple Objects:
  Mean: 39.39 ms
  95% Confidence Interval: ±1.61 ms
----------------------------------------
```

```
Prediction Stats for yolo11n_300_100_16_AdamW_1e-05_0.0001_0:
Inference Time :
  Mean: 37.96 ms
  95% Confidence Interval: ±0.79 ms
----------------------------------------
Total Time for Multiple Objects:
  Mean: 39.28 ms
  95% Confidence Interval: ±0.85 ms
----------------------------------------
```

```
Prediction Stats for yolo11s_300_100_16:
Inference Time :
  Mean: 79.73 ms
  95% Confidence Interval: ±2.68 ms
----------------------------------------
Total Time for Multiple Objects:
  Mean: 81.17 ms
  95% Confidence Interval: ±2.89 ms
----------------------------------------
```

```
Prediction Stats for yolov8n_300_100_16:
Inference Time :
  Mean: 39.12 ms
  95% Confidence Interval: ±2.69 ms
----------------------------------------
Total Time for Multiple Objects:
  Mean: 40.10 ms
  95% Confidence Interval: ±2.69 ms
----------------------------------------
```

```
Prediction Stats for rtdetr-l_300_100_16_AdamW_0.0001_0.0001_0:
Inference Time :
  Mean: 331.21 ms
  95% Confidence Interval: ±9.10 ms
----------------------------------------
Total Time for Multiple Objects:
  Mean: 334.40 ms
  95% Confidence Interval: ±8.94 ms
----------------------------------------
```

```
Prediction Stats for rtdetr-l_300_100_16_auto_0.01_0.0005_0:
Inference Time :
  Mean: 297.18 ms
  95% Confidence Interval: ±2.82 ms
----------------------------------------
Total Time for Multiple Objects:
  Mean: 298.37 ms
  95% Confidence Interval: ±3.00 ms
----------------------------------------
```

