## How to set up the environment

* clone the repository
```
git clone https://github.com/yubink2/AssistiveManipulation.git
```

* create and activate the conda environment
```
conda env create -f environment.yml
conda activate assistive-manip
```

* download the h36m dataset from here: https://drive.google.com/file/d/1lGbtOsasw5F2MjvwWd9AtzCXdIefvmpv/view?usp=sharing

* unzip the dataset and place it following directory structure: AssistiveManipulation/deep_mimic/mocap/data

* when installing pytorch, make sure that the version matches with your CUDA version (this environment is using CUDA 11.8)

## How to visualize the human motion from the dataset and collect the human joint limits

```
python collect_human_arm_limits.py
```

## How to run the wiping and manipulation pipeline

```
python wiping_manipulation_demo.py
```

## Notes

* human motion pipeline is based on: https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs/deep_mimic/mocap

* trajectory generation is based on: https://github.com/SamsungLabs/RAMP