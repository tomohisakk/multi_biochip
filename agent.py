import torch as T
from networks import ActorNetwork, CriticNetwork

class Agent:
	def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir,
					alpha=0.01, beta=0.01, fc1=64, fc2=64, gamma=0.95, tau=0.01):
		self.gamma = gamma
		self.tau = tau
		self.n_actions = n_actions
		self.agent_name = 'agent_%s' % agent_idx
		self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, chkpt_dir=chkpt_dir,  name=self.agent_name+'_actor')
		self.critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions, chkpt_dir=chkpt_dir, name=self.agent_name+'_critic')
		self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, chkpt_dir=chkpt_dir, name=self.agent_name+'_target_actor')
		self.target_critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions, chkpt_dir=chkpt_dir, name=self.agent_name+'_target_critic')

		self.update_network_parameters(tau=1)


	def update_network_parameters(self, tau=None):
		if tau is None:
			tau = self.tau

		# Actor parameters
		# refer to each parameters of network
		target_actor_params = self.target_actor.named_parameters()
		actor_params = self.actor.named_parameters()

		#print("--- Target_actor_params ---")
		#print(target_actor_params)

		# change the parameters to dict
		target_actor_state_dict = dict(target_actor_params)
		actor_state_dict = dict(actor_params)

		#print("--- Target_actor_state_dict ---")
		#print(target_actor_state_dict)

		# renew the value of each of acter_state_dict
		for name in actor_state_dict:
			actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1-tau)*target_actor_state_dict[name].clone()

		self.target_actor.load_state_dict(actor_state_dict)

		# Critic parameters, which is the same of an actors one
		target_critic_params = self.target_critic.named_parameters()
		critic_params = self.critic.named_parameters()

		target_critic_state_dict = dict(target_critic_params)
		critic_state_dict = dict(critic_params)

		for name in critic_state_dict:
			critic_state_dict[name] = tau*critic_state_dict[name].clone() + (1-tau)*target_critic_state_dict[name].clone()

		self.target_critic.load_state_dict(critic_state_dict)


	def choose_action(self, observation):
		state = T.tensor([observation], dtype=T.float).to(self.actor.device)
	#	print(state)
		probs = self.actor.forward(state)
#		noise = T.rand(self.n_actions).to(self.actor.device)
#		probs += noise
#		print(probs)
#		print(action)

#		print("--- Action ---")
#		print(action) # 4

		#print("--- action.detach().cpu().numpy()[0] ---")
		#print(action.detach().cpu().numpy()[0])
		return probs.detach().cpu().numpy()[0]


	def save_models(self):
		self.actor.save_checkpoint()
		self.target_actor.save_checkpoint()
		self.critic.save_checkpoint()
		self.target_critic.save_checkpoint()

	def load_models(self):
		self.actor.load_checkpoint()
		self.target_actor.load_checkpoint()
		self.critic.load_checkpoint()
		self.target_critic.load_checkpoint()
