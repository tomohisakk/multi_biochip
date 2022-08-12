import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from env import MEDAEnv
import time
import torch as T

def obs_list_to_state_vector(observation):
	state = np.array([])
	for obs in observation:
		state = np.concatenate([state, obs])
	return state

if __name__ == '__main__':
	env = MEDAEnv(w=8, l=8, n_agents=2)
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
	n_actions = 1
	maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, fc1=64, fc2=64, alpha=0.01, beta=0.01, scenario=scenario, chkpt_dir='tmp/maddpg/')
	memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, n_actions, n_agents, batch_size=1)

	PRINT_INTERVAL = 100
	N_GAMES = 50000
	total_steps = 0
	score_history = []
	evaluate = False
	best_score = -np.inf

	action = []

	if evaluate:
		maddpg_agents.load_checkpoint()

	for i in range(N_GAMES):
		obs = env.reset(n_modules=0)
		score = 0
		done = [False]*n_agents
		episode_step = 0
		while not any(done):
			probs = maddpg_agents.choose_action(obs)

			print(probs)

#			action_probs = T.distributions.Categorical(probs)
#			print(action_probs)

			#print(sum(probs[1]))

#			action.clear()
#			for j in range(n_agents):
#				action.append(np.random.choice(4, p=probs[j]))
#			print(action)
#			print(i)

#			print("--- Actions ---")
#			print(actions)

			obs_, reward, done, info = env.step(probs)

#			print("--- Observation ---")
#			print(obs_)
#			print("--- Reward ---")
#			print(reward)
#			print("--- Done ---")
#			print(done)

			state = obs_list_to_state_vector(obs)
			state_ = obs_list_to_state_vector(obs_)

			memory.store_transition(obs, state, probs, reward, obs_, state_, done)

			if total_steps % 100 == 0 and not evaluate:
				maddpg_agents.learn(memory)

			obs = obs_

			score += sum(reward)

#			print("--- Score ---")
#			print(score)

			total_steps += 1
			episode_step += 1

#		print(i, score)

		score_history.append(score)
		avg_score = np.mean(score_history[-100:])

#		print(avg_score)

#		print("--- Avg_score ---")
#		print(avg_score)

		if not evaluate:
			if avg_score > best_score:
				maddpg_agents.save_checkpoint()
				best_score = avg_score

		if i % PRINT_INTERVAL == 0 and i > 0:
			print('episode', i, 'average score {:.1f}'.format(avg_score))
