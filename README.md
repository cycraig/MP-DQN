#  Multi-Pass Deep Q-Networks

This repository includes several reinforcement learning algorithms for parameterised action space MDPs:

1. P-DQN [[Xiong et al. 2018]](https://arxiv.org/abs/1810.06394)

    - MP-DQN [[Bester et al. 2019]](https://arxiv.org/abs/1905.04388)
    - SP-DQN [[Bester et al. 2019]](https://arxiv.org/abs/1905.04388)
   
2. PA-DDPG [[Hausknecht & Stone 2016]](https://arxiv.org/abs/1511.04143)
3. Q-PAMDP [[Masson et al. 2016]](https://arxiv.org/abs/1509.01644)

Multi-Pass Deep Q-Networks (MP-DQN) fixes the over-paramaterisation problem of P-DQN by splitting the action-parameter inputs to the Q-network using several passes (in a parallel batch). Split Deep Q-Networks (SP-DQN) is a much slower solution which uses multiple Q-networks with/without shared feature-extraction layers. A weighted-indexed action-parameter loss function is also provided for P-DQN.

## Dependencies

- Python 3.5+ (tested with 3.5 and 3.6)
- pytorch 0.4.1 (1.0+ should work but will be slower)
- gym 0.10.5
- numpy
- click

## Domains


Experiment scripts are provided to run each algorithm on the following domains with parameterised actions:

- Platform (https://github.com/cycraig/gym-platform)
- Robot Soccer Goal (https://github.com/cycraig/gym-goal)
- Half Field Offense (https://github.com/cycraig/gym-soccer)

The simplest installation method for the above OpenAI Gym environments is as follows:
```bash
pip install -e git+https://github.com/cycraig/gym-platform#egg=gym_platform
pip install -e git+https://github.com/cycraig/gym-goal#egg=gym_goal
pip install -e git+https://github.com/cycraig/gym-soccer#egg=gym_soccer 
```

If something goes wrong, follow the installation instructions given by the repositories above. Note that gym-soccer has been updated for a later gym version and the reward function changed to reflect the one used in the code by Hausknecht & Stone [2016] (https://github.com/mhauskn/dqn-hfo). So use the one linked above rather than the OpenAI repository.

## Example Usage

Each run file has default flags in place, view the run_*domain*_*algorithm*.py files for more information. The click flags are configured to make it easier to run experiments and hyper-parameter searches in batches, which is better for scripts but makes it more annoying to type out. 

To run vanilla P-DQN on the Platform domain with default flags:
```bash
python run_platform_pdqn.py 
```

SP-DQN on the Robot Soccer Goal domain, rendering each episode:
```bash
python run_goal_pdqn.py --split True --visualise True --render-freq 1
```

MP-DQN on Half Field Offense with four hidden layers (note no spaces) and the weighted-indexed loss function:
```bash
python run_soccer_pdqn.py --multipass True --layers [1024,512,256,128] --weighted True --indexed True
```

## Citing
If this repository has helped your research, please cite the following:

```bibtex
@article{bester2019mpdqn,
	author    = {Bester, Craig J. and James, Steven D. and Konidaris, George D.},
	title     = {Multi-Pass {Q}-Networks for Deep Reinforcement Learning with Parameterised Action Spaces},
	journal   = {arXiv preprint arXiv:1905.04388},
	year      = {2019},
	archivePrefix = {arXiv},
	eprinttype    = {arxiv},
	eprint    = {1905.04388},
	url       = {http://arxiv.org/abs/1905.04388},
}
```
