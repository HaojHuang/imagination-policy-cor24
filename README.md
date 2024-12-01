# Imagination-Policy: using generative point cloud models for learning manipulation policies

## CoRL 24- [Paper PDF](https://openreview.net/pdf?id=56IzghzjfZ)   - [Project Website](https://haojhuang.github.io/imagine_page/)

## Install dependency (test on ubuntu 20.04/22.04)

using [conda](https://docs.anaconda.com/miniconda/install/) or [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge)

if you have both conda and miniforge3 installed you can switch using
```bash
# using conda
source /{your_path}/anaconda3/bin/activate 

# switch to miniforce3
source /{your_path}/miniforge3/bin/activate
```

**Step 1:** The current code is implemented with python 3.8

```bash
conda create -n imagine python=3.8
conda activate imagine
# a different torch version might also work
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
export FORCE_CUDA=1
# it takes a while to install pytorch3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install -r requirements.txt
# install pyg
conda install pyg -c pyg
# install escnn
apt install gfortran
pip install py3nj
pip install git+https://github.com/AMLab-Amsterdam/lie_learn
pip install git+https://github.com/QUVA-Lab/escnn
```


**Step 2:** Download [CoppeliaSim V4.1.0](https://coppeliarobotics.com/previousVersions) if you want to run the RLbench tasks. (You can skip step 2, 3 and 4 if you only want to test on a small ycb dataset we generated.)

After downloading the CoppeliaSIm, add the following to your ~/.bashrc file: (NOTE: the 'EDIT ME' in the first line)
```
export COPPELIASIM_ROOT=EDIT/ME/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```


**Step 3:** Install Pyrep

```bash
# install Pyrep
cd PyRep && pip install -e .
```

**Step 4:** install RLBench

```bash
# install RLBench
cd RLBench && pip install -e .
```

**Step 5:** test core packages installed
```
python -c "import torch; print(torch.version.cuda)"

python -c "import torch; print(torch.__version__)"

python -c "from pyrep import PyRep"

python -c "import rlbench"
```

**to resintall**
```
conda remove -n imagine --all
```
## Overview: This repo includes

### -training and testing on a small ycb dataset
### -training and testing on RLBench tasks
### - our YCB dataset and pretrained parameter

```bash
git clone 
cd IMAGINATION-POLICY-CORL24
```
[optional] Download our checkpoints from the [google drive link](https://drive.google.com/drive/folders/1CooOtaOGR5mCJ-LWVJwQDm7ISpuzkHSw?usp=drive_link) and put the `checkpoints` under `IMAGINATION-POLICY-CORL24`.



## train, test and visualize the on a small YCB dataset

**Step 1:** training (You can skip this step if you want to use our pretrained parameter)
```bash
# train pick/single generation
python train.py --model pvcnn --use_lan --use_color --n 100000 --save_steps 20000 --aug2 --device 0 --randsample (sample 2048 per object)

# train place/pair generation (--bi)
python train.py --model pvcnn --bi --use_lan --use_color --n 100005 --save_steps 20001 --aug2 --aug1 --device 0 --randsample (sample 2048 per object)
```

- --bi: indicates it is the pair generation or single generation
- --use_lan: using CLIP to encode the language description
- --n: total number of training step
- --aug2: randomly rotate P_b
- --aug1: randomly rotate P_a and P_b
- --device: which torch.device('cuda:0') is used, e.g., you can set it to 1 to use the second gpu  

Please note the checkpoints are saved under `./checkpoints/ycb`.

**Step 2:** testing on a randomly rotated (P_a, P_b) and generating the P_ab. (Check the table of our Ablation Study)
```bash
# test pick/single generation
python test.py --model pvcnn --use_lan --use_color --n 100000 --aug2 --device 0 --randsample --disp

# test pair generation
python test.py --model pvcnn --bi --use_lan --use_color --n 100005 --aug2 --aug1 --device 0 --randsample
```
- --disp: display the encoded pcd feature, input, generation and etc. (different colored pcds indicate different meanings, e.g., the orange color inidcates the generation.)
- --plot: display the loss curve

It saves a trajectory of generation in `./traj`.

**Step 3:** visualizing the trajectory from noise to P_ab

```
python displaypcd_flow.py
```
It generates a video under `./traj`


## Training and testing on RLbench tasks

**Step 1:** collect RLbench data
``` bash
cd RLBench/tools
# phone on base
python dataset_generator_per_var.py --tasks phone_on_base --episodes_per_task 35 --variation 0 --processes 1 --image_size 128,128
#
python dataset_generator_per_var.py --tasks stack_wine --episodes_per_task 35 --variation 0 --processes 1
#
python dataset_generator_per_var.py --tasks plug_charger_in_power_supply --episodes_per_task 35 --variation 0 --processes 1
#
python dataset_generator_per_var.py --tasks put_knife_in_knife_block --episodes_per_task 35 --variation 0 --processes 1
#
python dataset_generator_per_var.py --tasks put_plate_in_colored_dish_rack --episodes_per_task 35 --variation 0 --processes 1
# takes a while
python dataset_generator_per_var.py --tasks put_toilet_roll_on_stand --episodes_per_task 35 --variation 0 --processes 1
```
task list:

- phone_on_base
- stack_wine
- plug_charger_in_power_supply
- put_knife_in_knife_block
- put_plate_in_colored_dish_rack
- put_toilet_roll_on_stand

args:
- episodes_per_task: we collect 35 demos, 10 used for training and 25 used for testing
- image_size: 128,128 is the default image size


**Step 2:** preprocess the raw rlbench demo

```bash
#
python preprocess_raw_rlbench_demo.py --task_name phone_on_base --num_demos 10 (--disp: use --disp to visulize the segmeneted P_a, P_a, and the combined point cloud P_ab)
#
python preprocess_raw_rlbench_demo.py --task_name phone_on_base --num_demos 10
#
python preprocess_raw_rlbench_demo.py --task_name stack_wine --num_demos 10
#
python preprocess_raw_rlbench_demo.py --task_name put_plate_in_colored_dish_rack --num_demos 10
#
python preprocess_raw_rlbench_demo.py --task_name plug_charger_in_power_supply --num_demos 10
#
python preprocess_raw_rlbench_demo.py --task_name put_knife_in_knife_block --num_demos 10
```
It will save the processed demo in `./rlbench_data_processed/{task_name}/`

or use `python preprocess_raw_rlbench_demo.py --task_name all --num_demos 10` to preprocess all the six tasks


**Step 3:** Training a multi-task model


```bash
# train the pick/single generation for pick
python train_rlbench.py --gen pick --aug2 --use_lan --randsample --use_color --n 200000 --save_steps 50000 --model pvcnn --device 0

#train the place/pair generation for preplace and place
python train_rlbench.py --gen place --aug2 --aug1 --bi --use_lan --use_color --randsample --n 200000 --save_steps 50000 --model pvcnn --device 0
```

It will take a while to create and store the data dataset for the first run. The dataset will be stored under `./rlbench_data_processed/pick` and `./rlbench_data_processed/place`. Feel free to play the dataset with `create_rlbench_pick.py` and `create_rlbench_place.py` which consumes the processed rlbench data.

Please note the checkpoints are saved under `./checkpoints/pick` and `./checkpoints/place`.

**Step 4:** Test the model on RLbench simulator (please note the task name is short):
```bash
# test phone-on-base with trained model
python test_simulator.py --task phone_on_base --n 200000 --use_color --n_tests 25 --start_test 10 --device 0 (--disp --plot_action)
#
python test_simulator.py --task stack_wine --n 200000 --use_color --n_tests 25 --start_test 10 --device 0
#
python test_simulator.py --task insert_knife --n 200000 --use_color --n_tests 25 --start_test 10 --device 0
#
python test_simulator.py --task plug_charger --n 200000 --use_color --n_tests 25 --start_test 10 --device 0
#
python test_simulator.py --task put_plate --n 200000 --use_color --n_tests 25 --start_test 10 --device 0
#
python test_simulator.py --task put_roll --n 200000 --use_color --n_tests 25 --start_test 10 --device 0
```

- --disp: open the simulator
- --plot_action: visualize the action
- --n_tests: number of tests
- --start_test: the first test starts at 10th episodes. We collect 35 episodes and the first 10 are used for trainng.

It takes around 20s per generation (on RTX 4090). You can also use `sh test.sh` to rollout policy on all tasks.

**visualize the expert action for the pick-preplace-place settings**

```bash
# test phone on base with expert agent
python test_simulator.py --task phone_on_base --expert --n_tests 25 --start_test 10 --disp --plot_action
#
python test_simulator.py --task stack_wine --expert --n_tests 10
#
python test_simulator.py --task insert_knife --expert --n_tests 10
#
python test_simulator.py --task plug_charger --expert --n_tests 10
#
python test_simulator.py --task put_plate --expert --n_tests 10
#
python test_simulator.py --task put_roll --expert --n_tests 10
```

**other utils function from RLBench**

```bash
#
cd RLBench/tools
python cinematic_recorder.py --tasks phone_on_base

#
python simple_look_task.py
```

## using for Real robotics

check `models_stochastic/imagine_real_actor.py` for details.


**Gotchas**
1. Library "GLU" not found: sudo apt-get install freeglut3-dev
2. On a Linux machine, first run `apt install gfortran` and then `pip install py3nj`
3. "ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found", 
`conda deactivate && export LD_LIBRARY_PATH=/home/hhj/anaconda3/envs/imagine/lib:$LD_LIBRARY_PATH && conda activate imagine`
replace the libary path with yours