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
| Total | 13767     | 100 |

# Training tips

To train the model in a terminal only linux environment, you can use the `screen` command to create a new window, and run the training script in the background. This way, you can close the terminal, and the training will continue to run. For that, the following commands can be used:

```bash
screen -s {window_name}
screen -ad # Run detached
screen -ls # List windows
screen -r {window_id}
```