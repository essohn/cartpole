
import numpy as np

import gym

import torch
from torch import nn
from torch import optim

from model_dqn import agent, brain, replay_memory

import warnings

#..

class App():
    def __init__(self):
        print(torch.__version__)


    def init_agent(self, env, random_hyperparameters=False):

        # https://github.com/openai/gym/wiki/CartPole-v0
        #
        # Observation: (Cart Position, Cart Velocity, Pole Angle, Pole Velocity At Tip)
        # Actions: 0 - Push cart to the left, 1 - Push cart to the right
        num_states = env.observation_space.shape[0] # 입력 갯수: 4 
        num_actions = env.action_space.n # action 갯수: 2

        #------------------------------------------------------------
        # 변경가능 (1): Hyperparameters
        learning_rate = 0.01 # 학습률
        batch_size = 32 # 배치 사이즈
        gamma = 0.99 # 보상 감가율
        replay_memory_capacity = 10000 # 리플레이 메모리 용량
        #------------------------------------------------------------

        #------------------------------------------------------------
        # 변경가능 (2): 모델 구조
        model_obj = nn.Sequential(
            nn.Linear( num_states, 8 ),
            nn.ReLU(),
            nn.Linear( 8, num_actions ),
        )
        #------------------------------------------------------------

        #------------------------------------------------------------
        # 변경가능 (3): 옵티마이저 종류
        optimizer_obj = optim.Adam( model_obj.parameters(), lr=learning_rate )
        

        memory_obj = replay_memory.ReplayMemory( replay_memory_capacity )

        return agent.Agent( num_actions = num_actions, batch_size = batch_size, gamma = gamma, replay_memory = memory_obj, model = model_obj, optimizer = optimizer_obj )


    def run(self):
        
        NUM_EPISODES = 500
        MAX_STEPS = 1000

        best_list = []

        self.env = gym.make('CartPole-v0')

        self.agent = self.init_agent( self.env, random_hyperparameters=False )

        for episode in range(NUM_EPISODES):

            _observation = self.env.reset()

            observation = _observation
            if type(_observation) == tuple:
                observation = _observation[0]

            state = observation
            state = torch.from_numpy(state).type( torch.FloatTensor )
            state = torch.unsqueeze(state, 0)

            for step in range(MAX_STEPS):

                action = self.agent.get_action(state, episode)  # 다음 행동을 결정

                _observation = self.env.step( action.item() )

                if len(_observation) == 5:
                    observation_next, _, done, _, _ = _observation
                else:
                    observation_next, _, done, _ = _observation

                self.env.render()

                if done:

                    if step < 195: 
                        # 도중에 쓰러졌다면 reward -1
                        reward = -1
                    else:
                        # 끝까지 버텼다면 reward 1
                        reward = 1

                    best_list.append(step)

                else:
    
                    reward = 0 # reward 0으로 설정

                    state_next = observation_next  # 관측 결과를 그대로 상태로 사용
                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor)  # numpy 변수를 파이토치 텐서로 변환
                    state_next = torch.unsqueeze(state_next, 0)  # size 4를 size 1*4로 변환

                reward_tensor = torch.FloatTensor([reward])

                self.agent.memorize(state, action, state_next, reward_tensor)

                self.agent.update_q_function()

                # 관측 결과를 업데이트
                state = state_next

                if done:  
                    # 최근 100 episode 평균 reward
                    len_best_list = len(best_list[-100:])
                    avg_best_list = sum(best_list[-100:]) / len_best_list

                    print('episode {:4d}: \tstep: {:4d} \tscore: {:4d}, \taverage step({:3d}): {:6.2f}'
                        .format(episode, step+1, reward, len_best_list, avg_best_list ) )
                    break


        self.env.env.close()


if __name__=="__main__":

    warnings.filterwarnings("ignore")

    app = App()
    app.run()
