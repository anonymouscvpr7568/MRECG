

# Solving Oscillation Problem in Post-Training Quantization Through a Theoretical Perspective

This repository contains the experiments of our paper "Solving Oscillation Problem in Post-Training Quantization Through a Theoretical Perspective".

# Requirements

```python
pip install -r requirements.txt
```



# Running

To start running our code to get the mixed reconstruction granularity, you need to download the pretrained model, and copy the path of pretrained models to "model.path" in the config ymal. You also need to prepare the ImageNet dataset and copy the dataset path to "data.path" in the config yaml.

<font size=4>**MRECG with QDROP**</font>

```python
#!/usr/bin/env bash
cd ./MRECG/application/imagenet_example/PTQ
python3 ptq/ptq.py --config configs/qdrop/xx.yaml
```


<font size=4>**MRECG with BRECQ**</font>

```python
#!/usr/bin/env bash
cd ./MRECG/application/imagenet_example/PTQ
python3 ptq/ptq.py --config configs/brecq/xx.yaml
```


## Experimental Results

Table 1 "Solving Oscillation Problem in Post-Training Quantization Through a Theoretical Perspective".

<!-- ![ ACC](.\acc.png){width=65%} -->
<img src=".\acc.png" width = "967" height = "720" alt="ACC" align=center />

The reconstruction loss distribution of different algorithms on MobileNetV2 is as follows,

<!-- ![ loss_distribute](.\loss_distribute_all_alg.jpg){width=35%} -->
<img src=".\loss_distribute_all_alg.jpg" width = "600" height = "380" alt="loss_distribute" align=center />
