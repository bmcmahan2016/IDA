import pygame
from envs.utils import make_env
from PIL import Image
import numpy as np
import torch

from copilots.modules import SharedAutonomy
from copilots.diffusion import Diffusion

from experts.agents.sac import make_SAC
from intervention.functions import InterventionFunction

import tqdm
from pathlib import Path

def test_with_human(total_episodes=0, 
                    drop_first_episodes=0, 
                    env=None,
                    copilot=None,
                    advantage_fn=None,
                    gamma=None,
                    intervention_margin = 0,
                    output_path=None):
    
    pygame.init()
    DISPLAYSURF = pygame.display.set_mode((1920, 1080))
    pygame.display.set_caption("Click for Joystick Control")
    

    clock = pygame.time.Clock()
    joystick = pygame.joystick.Joystick(0)

    results_file = open(output_path / 'results.txt', 'w')

    key_pressed = False
    # Main loop
    while not key_pressed:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                key_pressed = True  # Exit the loop if the window is closed
            elif event.type == pygame.KEYDOWN:
                key_pressed = True  # Exit the loop if any key is pressed



    # create numpy arrays to hold the states and actions
    # these arrays can be saved later for analysis
    state_dim=9
    action_dim=2
    max_episode_len = 1000
    recorded_states = np.zeros((total_episodes, max_episode_len, state_dim))
    recorded_actions = np.zeros((total_episodes, max_episode_len, action_dim))
    recorded_rewards = np.zeros((total_episodes, max_episode_len, 1))

    successes, crashes, timeouts = 0, 0, 0
    for episode_num in tqdm.tqdm(range(total_episodes+drop_first_episodes)):
        observation, info = env.reset()
        image = env.render()
        clock.tick(1)  # pause for a second to let the user adjust
        episode_over = False

        
        total_return = 0
        t_steps = 0
        advantage_fn.reset()
        while not episode_over:

            # check if the window was closed
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            axis0 = joystick.get_axis(0)
            axis1 = joystick.get_axis(1)
            axis2 = joystick.get_axis(3)
            axis3 = joystick.get_axis(4) 


            action = np.array([axis3, axis0])


            #### have the copilot denoise the action
            state = torch.from_numpy(observation[env.env.goal_mask])
            state_conditioned_action = torch.unsqueeze(torch.hstack([state, torch.from_numpy(action).float()]), axis=0)
            state_conditioned_action = diffusion.sample(copilot, state_conditioned_action, gamma=gamma)

            copilot_action = state_conditioned_action[0, -2:].numpy()

            ### compute the copilot advantage
            behavior_action, adv = advantage_fn.behavior_policy(state.numpy(), action, copilot_action) 
            recorded_states[episode_num-drop_first_episodes, t_steps] = observation
            recorded_actions[episode_num-drop_first_episodes, t_steps] = behavior_action

            observation, reward, terminated, truncated, info = env.step(behavior_action)
            # will get over-written by itself on next loop iter if not terminal state
            recorded_states[episode_num-drop_first_episodes, t_steps+1] = observation
            # reward at timesteps zero corresponds to s[0] and a[0]
            recorded_rewards[episode_num-drop_first_episodes, t_steps] = reward

            total_return += reward
            t_steps += 1

            if terminated or truncated or (t_steps >= 999):
                episode_over = True



            pygame.display.flip()
            clock.tick(30)   # sets the framerate for the game
            image = env.render()
            pygame.display.update()

        if total_return > 200: 
            successes += 1
        elif reward == -100:
            crashes += 1
        elif t_steps == 999:
            timeouts += 1

    results_file.write('successful episodes: ' + str(successes) + '\n')
    results_file.write('crashed episodes: ' + str(crashes) + '\n')
    results_file.write('timeout episodes: ' + str(timeouts) + '\n')
    results_file.write("\n")
    results_file.close()

    np.save(output_path / Path("states.npy"), recorded_states)
    np.save(output_path / Path("behavior_actions.npy"), recorded_actions,)
    np.save(output_path / Path("rewards.npy"), recorded_rewards, )




if __name__ == '__main__':
    import argparse
    import yaml
    from pathlib import Path
    import os

    parser = argparse.ArgumentParser(
        prog="Test IDA in Reacher Environment",
        description="tests IDA in the Reacher environment with a real human in the loop"
    )

    # loads experimental configuration
    parser.add_argument('config_path')
    args = parser.parse_args()
    with open(args.config_path) as config_file:
        config = yaml.safe_load(config_file.read())

    env = make_env(config['env']['env_name'], 
                   render_mode='human', 
                   N=config['env']['num_goals'])

    agent = make_SAC(env.observation_space.low.size,
                 env.action_space.low.size,
                 env.action_space.low,
                 env.action_space.high)
    agent.load(config['intervention']['path'])
    Q_intervention = agent.q_func1

    if config['use_intervention']:
        disable=False
    else:
        disable=True
    advantage_fn = InterventionFunction(Q_intervention, env, num_goals=config['env']['num_goals'], discount=config['intervention']['gamma'],
                                        margin=config['margin'], disable=disable)
    
    copilot = SharedAutonomy(obs_size=config['copilot']['obs_size'])
    copilot.load_state_dict(torch.load(config['copilot']['path']))
    diffusion = Diffusion(action_dim=config['copilot']['action_dim'], 
                          img_size=64, 
                          device='cpu'
                          )

    output_path = Path(config['output_dir'])
    os.makedirs(output_path, exist_ok=True)

    test_with_human(total_episodes = config['num_evaluations'],
                    drop_first_episodes = config['drop_first_episodes'],
                    env=env,
                    copilot=copilot,
                    advantage_fn=advantage_fn,
                    gamma=config['gamma'],
                    intervention_margin=config['margin'],
                    output_path = output_path)


    # write the config file
    with open(output_path / 'config.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    
    pygame.quit()
