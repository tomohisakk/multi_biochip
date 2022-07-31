from enum import IntEnum
import math
import random
import copy
import numpy as np

class Actions(IntEnum):
	N = 0
	E = 1
	S = 2
	W = 3


class Mudlue:
	def __init__(self, x, y):
		self.x = x
		self.y = y

class Droplet:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def is_overlaping(self, another):
		if self.x == another.x and self.y == another.y:
			return True
		else:
			return False

	def is_too_close(self, another):
		distance = self.get_dist(another)
		if distance <= 1:
			return True
		else:
			return False

	def get_dist(self, another):
		diff_x = self.x - another.x
		diff_y = self.y - another.y
		return math.sqrt(diff_x*diff_x + diff_y*diff_y)

	def update(self, action, w, l, m_usage):
		next_state = [self.x, self.y]

		if action == Actions.N:
			next_state[1] -= 1
		elif action == Actions.E:
			next_state[0] += 1
		elif action == Actions.S:
			next_state[1] += 1
		else:
			next_state[0] -= 1
		
		if next_state[0] < 0 or next_state[1] < 0 or next_state[0] > w-1 or next_state[1] > l-1:
			return
		elif m_usage[next_state[0]][next_state[1]] >= 1:
			return

		self.x, self.y = next_state

class Routing:
	def __init__(self, w, l, n_agents):
		self.w = w
		self.l = l
		self.n_agents = n_agents

		self.starts = []
		self.droplets = []
		self.goals = []
		self.dists = []

		for i in range(n_agents):
			self.add_task()

		self.n_steps = [0] * self.n_agents

	def add_task(self):
		self._gen_legal_droplet(self.droplets)
		self._gen_legal_droplet(self.goals)

		while(self.droplets[-1].is_overlaping(self.goals[-1])):
			self.goals.pop()
			self._gen_legal_droplet(self.goals)
		self.dists.append(self.droplets[-1].get_dist(self.goals[-1]))
		self.starts.append(copy.deepcopy(self.droplets[-1]))

	def _gen_legal_droplet(self, dtype):
		state = (random.randint(0, self.w-1), random.randint(0, self.l-1))
		new_droplet = Droplet(state[1], state[0])
		while not self._is_good_droplet(new_droplet, dtype):
			state = (random.randint(0, self.w-1), random.randint(0, self.l-1))
			new_droplet = Droplet(state[1], state[0])
		dtype.append(new_droplet)

	def _is_good_droplet(self, new_d, dtype):
		for d in dtype:
			if d.is_too_close(new_d):
				return False
		return True

	def move_droplets(self, actions, m_usage):
		rewards = []
		for i in range(self.n_agents):
#			print(i)
			rewards.append(self.move_one_droplet(i, actions[i], m_usage))
		return rewards

	def move_one_droplet(self, i, action, m_usage):
		if self.dists[i] == 0:
			self.states[i] = copyy.deepcopy(self.goal[i])
			self.dists[i] = 0
			reward = 0.0
		else:
			self.droplets[i].update(action, self.w, self.l, m_usage)
			dist_ = self.droplets[i].get_dist(self.goals[i])

			if dist_ == 0:
				reward = 1.0
			elif dist_ < self.dists[i]:
				reward = 0.5
			else:
				reward = -0.5
			self.dists[i] = dist_

		return reward

	def refresh(self):
		self.starts.clear()
		self.droplets.clear()
		self.goals.clear()
		self.dists.clear()
		for i in range(self.n_agents):
			self.add_task()

	def is_done(self):
		return [dist == 0 for dist in self.dists]



class MEDAEnv:
	def __init__(self, w=8, l=8, n_agents=2, n_modules=0):
		super(MEDAEnv, self).__init__()
		assert w > 0 and l > 0 and n_agents > 0
		self.w = w
		self.l = l
		self.n_agents = n_agents
		self.agents = [i for i in range(n_agents)]

		self.actions = Actions
		self.action_spaces = len(self.actions)
		self.obs_shape = [(w,l)]*n_agents
		self.rewards = [0.]*n_agents
		self.dones = [False]*n_agents

		self.routing = Routing(w, l, n_agents)

		self.n_steps = 0
		self.n_max_steps = w+l
		self.m_usage = np.zeros((w, l))


	def step(self, actions):
		self.n_steps += 1
		
		rewards = self.routing.move_droplets(actions, self.m_usage)
		for key, r in zip(self.agents, rewards):
			self.rewards[key] = r

		obs = self.get_obs()

		if self.n_steps <= self.n_max_steps:
			is_dones = self.routing.is_done()
			for key, s in zip(self.agents, is_dones):
				self.dones[key] = s
			self.add_usage()
		else:
			for key in self.agents:
				self.dones[key] = True

		return obs, self.rewards, self.dones, {}

	def reset(self, n_modules):
		self.rewards = [0.]*self.n_agents
		self.dones = [False]*self.n_agents

		self.n_steps = 0
		self.routing.refresh()

		obs = self.get_obs()

		return obs

	def add_usage(self):
		for i, agent in enumerate(self.agents):
			if not self.dones[agent]:
				droplet = self.routing.droplets[i]
				self.m_usage[droplet.x][droplet.y] += 1

	def get_obs(self):
		obs = [[]]*self.n_agents
		for i, agent in enumerate(self.agents):
			obs[i] = self.get_one_obs(i)
#		obs[0] = np.reshape(obs[0], -1)
#		obs[1] = np.reshape(obs[1], -1)

		return obs

	def get_one_obs(self, agent_index):
		obs = np.zeros((self.l, self.w))

		for i in range(self.routing.n_agents):
			if i == agent_index:
				continue
			degrade = self.routing.droplets[i]
			obs = self._make_obs(obs, degrade, 1)
		
		goal = self.routing.goals[agent_index]
		obs = self._make_obs(obs, goal, 2)

		state = self.routing.droplets[agent_index]
		obs = self._make_obs(obs, state, 3)

		#print("--- Obs ---")
		#print(obs)

		return obs

	def _make_obs(self, obs, droplet, status):
		x = 0 if droplet.x < 0 else droplet.x
		x = self.w-1 if droplet.x >= self.w else droplet.x

		y = 0 if droplet.y < 0 else droplet.y
		y = self.l-1 if droplet.y >= self.l else droplet.y

		obs[y][x] = status

		return obs


"""
N_AGENTS = 2

env = MEDAEnv(w=10, l=10, n_agents=N_AGENTS)
env.reset(n_modules=0)
scores = 0
dones = env.dones
n_steps = 0

while dones[0] == False:
	n_steps += 1
	a = [random.randint(0,3)]*N_AGENTS
	print("--- Actions ---")
	print(a)
	print()
	obs_, rewards, dones, _ = env.step(a)
	scores += rewards[0]

	print("--- Observation ---")
	print(obs_)
	print()

	print("--- Reward ---")
	print(rewards)
	print()

	print("--- Dones ---")
	print(dones)
	print()


print("--- Game end ---")
print("Total score")
print()

print("--- N_steps ---")
print(n_steps)

"""