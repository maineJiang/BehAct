# **BehAct** 

## Annotated Tutorial 

**Important: Before starting, change the runtime to GPU.**

This repo is an tutorial on training BehAct from scratch. We will look at training a Multi-task agent on the 11 tasks. Overall, this guide is 
meant to complement the [paper](https://peract.github.io/) by providing concrete implementation details.  

<!-- <img src="https://peract.github.io/media/figures/sim_task.jpg" alt="drawing"/> -->

## Installation

**install the pytorch according to your cuda version**

<code>
pip install requirements.txt
</code>

## Data Preparation

To run the project, you can harvest the data from here

Then set the data path(DATA_FOLDER) and ckpts saving path(SAVE_DIR) in your config

## Run

**Train the model**

ws is the gpu's number
<code>
python train.py --config put_in.yaml --ws 1
</code>

**Evaluate the model on valid dataset**
<code>
python valid.py --config put_in.yaml --ws 1
</code>

##

## Credit
This notebook heavily builds on data-loading and pre-preprocessing code from [`ARM`](https://github.com/stepjam/ARM), [`YARR`](https://github.com/stepjam/YARR), [`PyRep`](https://github.com/stepjam/PyRep), [`RLBench`](https://github.com/stepjam/RLBench) by [Stephen James et al.](https://stepjam.github.io/) The [PerceiverIO](https://arxiv.org/abs/2107.14795) code is adapted from [`perceiver-pytorch`](https://github.com/lucidrains/perceiver-pytorch) by [Phil Wang (lucidrains)](https://github.com/lucidrains). The optimizer is based on [this LAMB implementation](https://github.com/cybertronai/pytorch-lamb). See the corresponding licenses below.

## Licenses
- [ARM License](https://github.com/stepjam/ARM/blob/main/LICENSE)
- [YARR Licence (Apache 2.0)](https://github.com/stepjam/YARR/blob/main/LICENSE)
- [RLBench Licence](https://github.com/stepjam/RLBench/blob/master/LICENSE)
- [PyRep License (MIT)](https://github.com/stepjam/PyRep/blob/master/LICENSE)
- [Perceiver PyTorch License (MIT)](https://github.com/lucidrains/perceiver-pytorch/blob/main/LICENSE)
- [LAMB License (MIT)](https://github.com/cybertronai/pytorch-lamb/blob/master/LICENSE)


## Citations 

**PerAct**
```
@inproceedings{shridhar2022peract,
  title     = {Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation},
  author    = {Shridhar, Mohit and Manuelli, Lucas and Fox, Dieter},
  booktitle = {Proceedings of the 6th Conference on Robot Learning (CoRL)},
  year      = {2022},
}
```

**C2FARM**
```
@inproceedings{james2022coarse,
  title={Coarse-to-fine q-attention: Efficient learning for visual robotic manipulation via discretisation},
  author={James, Stephen and Wada, Kentaro and Laidlow, Tristan and Davison, Andrew J},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13739--13748},
  year={2022}
}
```

**PerceiverIO**
```
@article{jaegle2021perceiver,
  title={Perceiver io: A general architecture for structured inputs \& outputs},
  author={Jaegle, Andrew and Borgeaud, Sebastian and Alayrac, Jean-Baptiste and Doersch, Carl and Ionescu, Catalin and Ding, David and Koppula, Skanda and Zoran, Daniel and Brock, Andrew and Shelhamer, Evan and others},
  journal={arXiv preprint arXiv:2107.14795},
  year={2021}
}
```


**RLBench**
```
@article{james2020rlbench,
  title={Rlbench: The robot learning benchmark \& learning environment},
  author={James, Stephen and Ma, Zicong and Arrojo, David Rovick and Davison, Andrew J},
  journal={IEEE Robotics and Automation Letters},
  volume={5},
  number={2},
  pages={3019--3026},
  year={2020},
  publisher={IEEE}
}
```
