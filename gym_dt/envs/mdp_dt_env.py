import gym
import numpy as np
import matplotlib.pyplot as plt
from typing import Union

array = Union[list, np.array]


def traverse_tree(cell, tree):
	if isinstance(tree, ActionNode):
		return tree.action
	elif isinstance(tree, InfoNode):
		feature = tree.feature
		val = tree.value
		if cell[feature] <= val:
			subtree = tree.left

		else:
			subtree = tree.right
		return traverse_tree(cell, subtree)


def traverse_tree_build(cell, tree):
	if isinstance(tree, ActionNode):
		return tree.action
	elif isinstance(tree, InfoNode):
		feature = tree.feature
		val = tree.value
		if cell[feature] < val:
			subtree = tree.left

		else:
			subtree = tree.right
		return traverse_tree_build(cell, subtree)



class ActionNode:
	"""
	Generic class for a leaf node of a binary tree.
	"""

	def __init__(self, action: int):
		self.action = action

class InfoNode:
	"""
	Generic class for an internal node of binary tree.
	"""

	def __init__(self, feature: int, value: float, left, right):
		self.feature = feature
		self.value = value
		self.left = left
		self.right = right


class DecisionTreePolicy:
	"""
	Generic Decision Tree Policy class (binary).
	Builds a given depth random binary decision tree over a 2d-state-N-action space.

	:param p: Where to split domains. If p=1, domains are partitioned in their centers.
	(see: https://arxiv.org/abs/2102.13045 )
	:param opt_tree_depth: Target tree depth for the final binary decision tree.
	:param all_p: Values in [0,1] indicating where a subspace can be split along a given dimension.
	:param nb_actions: Number of actions possible for decision (leaf) nodes values.
	"""

	def __init__(self, opt_tree_depth: int, p: int, all_p: array, nb_actions: int):

		self.opt_tree_depth = opt_tree_depth
		self.p = p
		self.all_p = all_p
		self.nb_actions = nb_actions

		self.tree = self.recursive_2D_dt_partition(
			domains=[[0, 1], [0, 1]],
			depth=0,
		)

	def recursive_2D_dt_partition(
		self,
		domains: list,
		depth: int,
	):
		"""
		Recursively generates a random binary decision tree over a 2d-state-N-action space.
		:param domains: A closed subspace of |R^2.
		:param depth: Current number of internal nodes traversed to reach the subspace domains.
		"""
		if depth == self.opt_tree_depth:
			a = np.random.randint(self.nb_actions)
			return ActionNode(a)
		else:
			random_dim = np.random.randint(2)
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
			child_left = self.recursive_2D_dt_partition(
				left, depth
			)
			child_right = self.recursive_2D_dt_partition(
				right, depth
			)

			return InfoNode(random_dim, random_val, child_left, child_right)

	def pol(self, state: array):
		return traverse_tree(state, self.tree)


class DecisionTreeEnv(gym.Env):
	"""
	Simple continuous 2-d states mdp with discrete cardinal actions for which
	the optimal policy is a decision tree of given depth.

	In s_t, the reward is 1 for performing the action opt_pol(s_t) and 0 o.w. .
	The maximum steps per episode is 500.
	The maximum cumulated reward is thus 500.
	:param opt_tree_depth: The max depth of the optimal decision tree policy.
	:param p: Where to split domains. If p=1, domains are partitioned in their centers.
	(see: https://arxiv.org/abs/2102.13045 )
	"""

	# to generalize in N dimensions
	def __init__(
		self,
		opt_tree_depth: int = 2,
		p: int = 1,
		):

		self.observation_space = gym.spaces.Box(
			np.array([0, 0]), np.array([1, 1]), dtype=np.float32
		)
		self.action_space = gym.spaces.Discrete(4)
		self.step_size = 1/(2**opt_tree_depth)
		self.action_map = [
			[0, self.step_size],
			[self.step_size, 0],
			[-self.step_size, 0],
			[0, -self.step_size],
		]

		self.opt_tree_depth = opt_tree_depth
		self.p = p
		self.obs_step = (1 / (self.p + 1)) ** self.opt_tree_depth
		self.all_p = np.linspace(0, 1, p + 2)[1:-1]

		self.dtp = self.get_random_dtp()

		self.state = self.observation_space.sample()

	def get_random_dtp(self):
		dtp = DecisionTreePolicy(
			self.opt_tree_depth, self.p, self.all_p, self.action_space.n
		)
		return dtp

	def get_opt_state_action_space_from_dtp(self):
		sa_space = np.zeros((
			len(np.linspace(0, 1, (self.p + 1) ** (self.opt_tree_depth) + 1)[:-1]),
			len(np.linspace(0, 1, (self.p + 1) ** (self.opt_tree_depth) + 1)[:-1]),
		))
		for x in range(len(sa_space)):
			for y in range(len(sa_space)):
				sa_space[x][len(sa_space) - 1 - y] = traverse_tree_build(
					[x * self.obs_step, y * self.obs_step], self.dtp.tree
				)
		return sa_space.T

	def plot_policy(self):
		sa_space = self.get_opt_state_action_space_from_dtp()
		self.fig = plt.pcolormesh(
			np.linspace(0, 1, (self.p + 1) ** (self.opt_tree_depth) + 1),
			np.linspace(0, 1, (self.p + 1) ** (self.opt_tree_depth) + 1),
			sa_space,
			cmap="Blues",
		)
		plt.savefig("optim_tree_" + str(self.opt_tree_depth) + ".pdf")

	def reset(self, seed=None, return_info=False, options=None):
		self.state = self.observation_space.sample()
		return self.state

	def step(self, action):
		opt_action = self.dtp.pol(self.state)
		if opt_action == action:
			reward = 1
		else:
			reward = 0

		self.state += self.action_map[action]
		self.state = np.clip(self.state, 0, 1)

		done = False

		return self.state, reward, done, {}

	def render_(self):
		self.plot_policy()
		plt.scatter(self.state[0], self.state[1], c="red")
		plt.show(block=False)
		plt.pause(0.5)
		plt.close()


#
# env.plot_partition()
#
# s = env.reset()
# done = False
# while not done:
#     a = env.action_space.sample()
#     s, r, done, _ = env.step(a)
#     # env.render_()
#     print(r)
