import pygame
from envs.utils import make_env
from PIL import Image
import numpy as np
import torch

from copilots.modules import SharedAutonomy
from copilots.diffusion import Diffusion

from experts.agents.sac import make_SAC
from intervention.functions import make_intervetion_function, make_trajectory_intervetion_function



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




    for episode_num in range(total_episodes+drop_first_episodes):
        observation, info = env.reset()
        image = env.render()
        clock.tick(1)  # pause for a second to let the user adjust
        episode_over = False

        action_hist = []
        partial_state_hist = []
        total_return = 0
        t_steps = 0
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
            if advantage_fn is not None:
                copilot_adv = advantage_fn(partial_state_hist + [state.numpy()], action_hist + [copilot_action])
                human_adv = advantage_fn(partial_state_hist + [state.numpy()], action_hist + [action])
                adv = (torch.sum(torch.sign(copilot_adv-human_adv)) / len(copilot_adv)).item()
            else:
                adv = 2  # always follow the copilot


            partial_state_hist.append(state.numpy())
            if adv > intervention_margin:
                observation, reward, terminated, truncated, info = env.step(copilot_action)
                action_hist.append(copilot_action)
            else:
                observation, reward, terminated, truncated, info = env.step(action)
                action_hist.append(action)

            total_return += reward
            t_steps += 1

            if terminated or truncated or (t_steps >= 1000):
                episode_over = True


            pygame.display.flip()
            clock.tick(30)   # sets the framerate for the game
            image = env.render()
            # image = Image.fromarray(image, 'RGB')
            # image = image.resize((1920, 1080), Image.Resampling.LANCZOS)
            # mode, size, data = image.mode, image.size, image.tobytes()
            # image = pygame.image.fromstring(data, size, mode)

            # DISPLAYSURF.blit(image, (0,0))
            pygame.display.update()

        if episode_num > drop_first_episodes:
            results_file.write('episode total return: '  + str(total_return) + '\n')
    results_file.close()




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
    if config['advantage_fn']:
        advantage_fn = make_trajectory_intervetion_function(Q_intervention, 
                                                            env, 
                                                            discount=config['intervention']['gamma']
                                                            )
    else:
        advantage_fn=None
    
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
