# Introduction

![IDA Overview](IDA_high_level.png )


Interventional Assist (IA) provides a hyper-parameter free framework for shared autonomy (SA). 
Most previous work in SA typically shares control between a user (pilot) and an assistive algorithm (copilot) using a control sharing hyperparameter.
IA mitigates this hyperparmeter by adopting an intervention based approach for shared control that is similar (the inverse of) to the Student Teacher Framework (TSF).
Specifically, with IA, an assistive algorithm (in this case an RL agent) will intervene in the user's (pilot's) actions when the user's action is deemed bad. 
Our main contribution herein is the development of an intervention function that enables us to determine when the user's action should be intervened.
Our intervention function is modular and can be used with different models.
Herein we combine our IA with a copilot parametrized by a diffusion model, termed interventional diffusion assist (IDA).


## Steps for Reproducing Experiments
This section outlines the steps required to train a copilot and derive an intervention function for using IDA. There are three main components or modules for IDA: a human controller, an assisitive AI controller (copilot), and an intervention function (which determines how control is shared between the human and copilot).

### Expert Training
### Copilot Training
### Running IDA inference


## Installation
installation instructions coming soon