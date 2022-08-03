import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from env import MEDAEnv

def obs_list_to_state_vector(observation):
	state = np.array([])
	for obs in observation:
		state = np.concatenate([state, obs])
	return state

if __name__ == '__main__':
	env = MEDAEnv(w=10, l=10, n_agents=2)
	n_agents = env.n_agents
	actor_dims = []

	for i in range(n_agents):
		actor_dims.append(env.obs_shape[i][0]*env.obs_shape[i][1])

#		print("--- Actor_dims ---")
#		print(actor_dims)

	critic_dims = sum(actor_dims)
#	print("--- Critic_dims ---")
#	print(critic_dims)

	scenario = 'scenario'

	# action space is a list of arrays, assume each agent has same action space
	n_actions = env.action_spaces
	maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, fc1=64, fc2=64, alpha=0.01, beta=0.01, scenario=scenario, chkpt_dir='tmp/maddpg/')
	memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, n_actions, n_agents, batch_size=1024)

	N_GAMES = 50000
	total_steps = 0
	evaluate = True

	a = []

	if evaluate:
		maddpg_agents.load_checkpoint()

	for i in range(N_GAMES):
		obs = env.reset(n_modules=0)
		score = 0
		done = [False]*n_agents
		episode_step = 0
		while not any(done):
			actions = maddpg_agents.choose_action(obs)

			a.clear()
			for i in range(len(actions)):
				a.append(np.argmax(actions[i]))

			obs_, reward, done, info = env.step(a)

			state = obs_list_to_state_vector(obs)
			state_ = obs_list_to_state_vector(obs_)

			memory.store_transition(obs, state, actions, reward, obs_, state_, done)

			obs = obs_

			score += sum(reward)
			total_steps += 1
			episode_step += 1

