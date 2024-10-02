## How to set up the environment

* clone the repository
```
git clone https://github.com/yubink2/AssistiveManipulation.git
```

* download the h36m dataset from here: https://drive.google.com/file/d/1lGbtOsasw5F2MjvwWd9AtzCXdIefvmpv/view?usp=sharing

* unzip the dataset and place it following directory structure: AssistiveManipulation/deep_mimic/mocap/data

* * build the docker image
```
docker build -t assistive-manip-env .
```

* run the docker container
```
docker run -it --rm \
    --gpus all \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    assistive-manip-env /bin/bash
```

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

* this environment is using CUDA 11.8