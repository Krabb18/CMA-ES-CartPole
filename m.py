import cma
from model import Controller
import gymnasium as gym
import torch
import numpy as np
from gymnasium.vector import AsyncVectorEnv

INITIAL_SIGMA = 0.1
SIGMA_DECAY = 0.992
popsize=16
max_generations=100
rollouts = 10

input_ = 4
output_ = 2

def make_env(name='CartPole-v1'):
    def _init():
        env = gym.make(name, render_mode='rgb_array')
        return env
    return _init

def create_vector_envs(num_envs):
    return AsyncVectorEnv([make_env() for _ in range(num_envs)], 
                          shared_memory=True)

def process_actions(controllers, x):
    return torch.stack(
        [controllers[i](torch.tensor(x[i])).argmax(dim=-1) for i in range(x.shape[0])], dim=0
    )

def evaluate_policies(solutions, controller_class, max_steps,input_shape, output_shape):

    num_policies = len(solutions)

    #load all solutions weights
    controllers = []
    with torch.no_grad():
        for params in solutions:
            controller = Controller(input_shape, output_shape)
            torch.nn.utils.vector_to_parameters(
                torch.tensor(params, dtype=torch.float32),
                controller.parameters()
            )
            controllers.append(controller)

    #env
    envs = create_vector_envs(num_envs=num_policies)
    obs, _ = envs.reset()

    print(obs.shape)

    cumulative_rewards = np.zeros(num_policies)
    dones = np.full(num_policies, False)

    with torch.no_grad():
        for _ in range(max_steps):

            if np.all(dones):
                break

            actions = process_actions(controllers, obs)
            obs, rewards, dones_new, _, _ = envs.step(actions.detach().cpu().numpy())

            dones = np.logical_or(dones, dones_new)
            cumulative_rewards += rewards * (~dones)

    envs.close()
    return cumulative_rewards.tolist()


    

if __name__ == '__main__':

    np.random.seed(101)


    controller = Controller(input_, output_).to("cuda")
    initial_params = torch.nn.utils.parameters_to_vector(
        controller.parameters()
    ).detach().cpu().numpy()

    es = cma.CMAEvolutionStrategy(initial_params,  INITIAL_SIGMA, {'popsize': popsize})

    for generation in range(0, max_generations+1):

        solutions = es.ask()

        mean_rewards = []
        for _ in range(rollouts):
            rewards = evaluate_policies(solutions, Controller, 1000, input_, output_)
            mean_rewards.append(rewards)
        mean_rewards = np.mean(mean_rewards, axis=0) 

        #update cma-es
        es.tell(solutions, [-r for r in mean_rewards])

    #save best controller
    print(es.result.xbest)
    best_controller = Controller(input_, output_)
    torch.nn.utils.vector_to_parameters(
        torch.tensor(es.result.xbest, dtype=torch.float32),
        best_controller.parameters()
    )
    torch.save(best_controller.state_dict(), "models/best_controller.pth")