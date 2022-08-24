import gym
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from binarytree import Node, tree
array = Union[list, np.array]


def traverse_tree(cell, tree):
	if isinstance(tree, ActionNode):
		return tree.action
	elif isinstance(tree, InfoNode):
		feature = tree.feature
		val = tree.split_value
		if cell[feature] <= val:
			subtree = tree.left

		else:
			subtree = tree.right
		return traverse_tree(cell, subtree)


## Generic Classes for Binary Tree ##
class InfoNode(Node):
	def __init__(self, feature, split_value, left, right, lab = None):
		if lab is None:
			lab = "feat_" +str(feature)
		else:
			lab = lab[feature]
		label = lab + "≤" + str(split_value)
		super().__init__(label, left, right)
		self.feature = feature
		self.split_value = split_value


class ActionNode(Node):
	def __init__(self, action, action_label = None):
		if action_label is not None:
			label = action_label[action]
		else :
			label = "act_" + str(action)
		super().__init__(label)
		self.action = action



class DecisionTreePolicy:
	"""
	Generic Decision Tree Policy class (binary).
	Builds a given depth random binary decision tree over a Nd-state-N-action space.

	:param p: Where to split domains. If p=1, domains are partitioned in their centers.
	(see: https://arxiv.org/abs/2102.13045 )
	:param opt_tree_depth: Target tree depth for the final binary decision tree.
	:param all_p: Values in [0,1] indicating where a subspace can be split along a given dimension.
	:param nb_actions: Number of actions possible for decision (leaf) nodes values.
	"""

	def __init__(self, opt_tree_depth: int, p: int, all_p: array, nb_actions: int, nb_base_features: int, seed = None):

		self.opt_tree_depth = opt_tree_depth
		self.p = p
		self.all_p = all_p
		self.nb_actions = nb_actions
		self.nb_base_features = nb_base_features
		temp  = []
		for feat in range(self.nb_base_features):
			temp.append([0,1])

		if seed is not None:
			np.random.seed(seed)
		self.tree = self.recursive_random_labeled_binary_tree(
			domains=temp,
			depth=0,
		)

	def recursive_random_labeled_binary_tree(
		self, domains, depth
	):
		if depth == self.opt_tree_depth:
			a = np.random.randint(self.nb_actions)
			return ActionNode(a)
		else:
			random_dim = np.random.randint(self.nb_base_features)
			true_vals = (
				self.all_p * (domains[random_dim][1] - domains[random_dim][0])
				+ domains[random_dim][0]
			)
			random_val = np.random.choice(true_vals)
			left = domains.copy()
			right = domains.copy()
			left[random_dim] = [domains[random_dim][0], random_val]
			right[random_dim] = [random_val, domains[random_dim][1]]
			depth += 1
			# left_right = np.random.randint(2)
			child_left = self.recursive_random_labeled_binary_tree(
				left, depth
			)
			child_right = self.recursive_random_labeled_binary_tree(
				right, depth
			)

			return InfoNode(random_dim, random_val, child_left, child_right)

	def pol(self, state: array):
		return traverse_tree(state, self.tree)

	def plot(self, filename = "optimal_policy"):
		graph = self.tree.graphviz()
		graph.body
		graph.render(filename, format="png")



class DecisionTreeEnv(gym.Env):
	"""
	Simple continuous N-d states mdp with discrete cardinal actions for which
	the optimal policy is a decision tree of given depth.

	In s_t, the reward is 1 for performing the action opt_pol(s_t) and -1 o.w. .
	The maximum steps per episode is 100.
	The maximum cumulated reward is thus 100.
	:param opt_tree_depth: The max depth of the optimal decision tree policy.
	:param p: Where to split domains. If p=1, domains are partitioned in their centers.
	(see: https://arxiv.org/abs/2102.13045 )
	"""

	# to generalize in N dimensions
	def __init__(
		self,
		opt_tree_depth: int = 2,
		p: int = 1,
		nb_base_features: int = 2,
		seed = None
		):
		self.nb_base_features = nb_base_features
		self.observation_space = gym.spaces.Box(
			np.array([0, 0]), np.array([1, 1]), dtype=np.float32
		)
		self.action_space = gym.spaces.Discrete(2*nb_base_features)
		if opt_tree_depth <= 6:
			self.step_size = 1/100
		else:
			self.step_size = 1/(2**opt_tree_depth)
		self.action_map = []
		for feat in range(self.nb_base_features):
			action = np.zeros(self.nb_base_features)
			action[feat] = self.step_size
			self.action_map.append(action)
			action = np.zeros(self.nb_base_features)
			action[feat] = -self.step_size
			self.action_map.append(action)

		self.opt_tree_depth = opt_tree_depth
		self.p = p
		self.obs_step = (1 / (self.p + 1)) ** self.opt_tree_depth
		self.all_p = np.linspace(0, 1, p + 2)[1:-1]

		self.dtp = self.get_random_dtp(seed)

		self.state = self.observation_space.sample()

	def get_random_dtp(self, seed = None):
		dtp = DecisionTreePolicy(
			self.opt_tree_depth, self.p, self.all_p, self.action_space.n, self.nb_base_features, seed
		)
		return dtp

	def reset(self, seed=None, return_info=False, options=None):
		self.state = self.observation_space.sample()
		return self.state

	def step(self, action):
		opt_action = self.dtp.pol(self.state)
		if opt_action == action:
			reward = 1
		else:
			reward = -1

		self.state += self.action_map[action]
		self.state = np.clip(self.state, 0, 1)

		done = False

		return self.state, reward, done, {}


#
# env.plot_partition()
# env = DecisionTreeEnv()
# env.dtp.plot()
# s = env.reset()
# done = False
# while not done:
# 	a = env.action_space.sample()
# 	s, r, done, _ = env.step(a)
# 	# env.render_()
# 	print(r)
