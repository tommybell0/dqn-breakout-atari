import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
EVAL=20
env = gym.make("ALE/Breakout-v5",render_mode='human',obs_type="rgb")
eval_env = gym.make("ALE/Breakout-v5",render_mode='human',obs_type="rgb")
# set up matplotlib
def resize(image):
    resized_image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_CUBIC)
    return resized_image
def trans(array):
    return np.mean(array, axis=2)
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()
# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, n_frames_to_stack):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(n_frames_to_stack, 32, kernel_size=5,stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3,stride=2)
        self.linear = nn.Linear(1600, 256)  # 添加一个线性层
        self.fc = nn.Linear(256, n_actions)  # 这里假设前面的池化操作后输出大小是1600（实际应根据具体情况调整）
        
    def forward(self, x):
        #print("x:",x.shape)
        x = x.float() / 255.0   # 标准化输入值
        #print("x/255:",x.shape)
        x = F.relu(self.conv1(x))
        #print("conv1x:",x.shape)
        x = F.max_pool2d(x,kernel_size=2,stride=2)   # 池化操作可以增加非线性特征并减小尺寸 
        #print("poolx:",x.shape)
        x = F.relu(self.conv2(x))
        #print("conv2x:",x.shape)
        x = F.max_pool2d(x,kernel_size=2,stride=2)
        #print("poolx:",x.shape)
         # 将特征张量展平成一维向量
         # 注意：此处的1600应与定义中的全连接层输入尺寸一致
        x = x.view(x.size(0), -1)  
        #print("xview:",x.shape)
        x = F.relu(self.linear(x))  # 使用新添加的线性层进行前向传播
        #print("xline:",x.shape)
        return self.fc(x)
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 5000
TAU = 0.05
LR = 6e-4
SEED=3117
n_actions = env.action_space.n
state, info = env.reset()
n_frames_to_stack = 10
#n_observations = len(state) 
n_observations = (96,96)
print("n_actions:",n_actions)
print("n_observations:",n_observations)
print("observations:",state.shape)
policy_net = DQN(n_observations, n_actions,n_frames_to_stack).to(device)
target_net = DQN(n_observations, n_actions,n_frames_to_stack).to(device)
#if policy_net.load_state_dict(torch.load("modelbreakout10.pt")):
#    print("load modelbreakout10.pt")
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(capacity=2000)
steps_done = 0
def evalselect_action(state):
    global steps_done
    #state = state.view(1, -1)
    #print("select_action state:",state.shape)
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > 0:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
def select_action(state):
    global steps_done
    #state = state.view(1, -1)
    #print("select_action state:",state.shape)
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
episode_durations = []
def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    
    batch = Transition(*zip(*transitions))
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    if batch.next_state is None:
        continous
    else:  
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    #print("batch.state:",batch.state)
    state_batch = torch.cat(batch.state)
    #print("state_batch:",state_batch.shape)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
if torch.cuda.is_available():
    num_episodes = 600000
else:
    num_episodes = 50
total_reward = 0
max_reward = 0
total_t = 0
def eval():
    print("eval")
    total_reward = 0
    max_reward = 0
    total_t = 0
    state, info = eval_env.reset()
    #print("s1:",state.shape)
    state = trans(state)
    #print("s2:",state.shape)
    state = resize(state)
    #print("s3:",state.shape)
    frames= []
    new_frame = torch.tensor(state, dtype=torch.float32, device=device)
    frames.append(new_frame)
    frames.append(new_frame)
    frames.append(new_frame)
    frames.append(new_frame)
    frames.append(new_frame)
    frames.append(new_frame)
    frames.append(new_frame)
    frames.append(new_frame)
    frames.append(new_frame)
    frames.append(new_frame)
    stacked_state = torch.stack(frames)
    #print("stacked_state",stacked_state.shape)
    stacked_next_state = None
    for t in count():    
        action = evalselect_action(stacked_state)
        #print("action" ,action.item())
        observation,reward ,terminated,truncated,_=eval_env.step(action.item())
        observation =trans(observation)
        observation =resize(observation)
        done=(terminated or truncated)
        if done:
            next_state = None  
        else:
            #torch.stack(frames).unsqueeze(0).to(device) 
            next_state = torch.tensor(observation, dtype=torch.float32, device=device)
            frames.pop(0)
            frames.append(next_state)
        # Move all tensors to the same device (if necessary) 
        reward_tensor= torch.tensor([reward],device=device)
        total_reward +=reward
        #frames[9] = frames[9].squeeze()      # 移除不必要的批次维度
        stacked_next_state = torch.stack(frames).unsqueeze(0)
        stacked_state = stacked_next_state
        if done:
            total_t +=t
            print(f'i_episode: {i_episode}, total_reward: {total_reward},   t: {t}  total_t: {total_t}  ')
            episode_durations.append(total_reward)
            if total_reward > max_reward:
                max_reward = total_reward
                print(f'i_episode: {i_episode}, total_reward: {total_reward},max_reward: {max_reward}  ')
                if max_reward > 1.0:
                    torch.save(policy_net.state_dict(), "modelbreakout10.pt")
                    print("save model")
            total_reward=0
            plot_durations()
            break
for i_episode in range(num_episodes):  
    if i_episode%EVAL ==0:
        eval()
        
    # Initialize the environment and get its state
    state, info = env.reset()
    #print("s1:",state.shape)
    state = trans(state)
    #print("s2:",state.shape)
    state = resize(state)
    #print("s3:",state.shape)
    frames= []
    new_frame = torch.tensor(state, dtype=torch.float32, device=device)
    frames.append(new_frame)
    frames.append(new_frame)
    frames.append(new_frame)
    frames.append(new_frame)
    frames.append(new_frame)
    frames.append(new_frame)
    frames.append(new_frame)
    frames.append(new_frame)
    frames.append(new_frame)
    frames.append(new_frame)
    stacked_state = torch.stack(frames).unsqueeze(0)
    #print("stacked_state",stacked_state.shape)
    stacked_next_state = None
    for t in count():    
        #print("stacked_state:",stacked_state.shape)
        action = select_action(stacked_state)
        #print("action",action.item())
        observation,reward ,terminated,truncated,_=env.step(action.item())
        observation =trans(observation)
        observation =resize(observation)
        done=(terminated or truncated)
        if done:
            next_state = None  
        else:
            #torch.stack(frames).unsqueeze(0).to(device) 
            next_state = torch.tensor(observation, dtype=torch.float32, device=device)
            frames.pop(0)
            frames.append(next_state)
        # Move all tensors to the same device (if necessary) 
        reward_tensor= torch.tensor([reward],device=device)
        total_reward +=reward
        #frames[9] = frames[9].squeeze()      # 移除不必要的批次维度
        stacked_next_state = torch.stack(frames).unsqueeze(0)
        memory.push(stacked_state, action, stacked_next_state, reward_tensor)
        stacked_state = stacked_next_state
        optimize_model()        
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        if done:
            total_t +=t
            print(f'i_episode: {i_episode}, total_reward: {total_reward},   t: {t}  total_t: {total_t}  ')
            episode_durations.append(total_reward)
            if total_reward > max_reward:
                max_reward = total_reward
                print(f'i_episode: {i_episode}, total_reward: {total_reward},max_reward: {max_reward}  ')
                #if max_reward > 925.0:
                    #torch.save(policy_net.state_dict(), "modelcar10.pt")
                    #print("save model")
            total_reward=0
            plot_durations()
            break
print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()