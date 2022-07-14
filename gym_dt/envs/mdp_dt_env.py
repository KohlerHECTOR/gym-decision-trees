import gym
import numpy as np
import matplotlib.pyplot as plt



class DecisionTreeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    #to generalize in N dimensions
    def __init__(self, opt_tree_depth = 2, nb_actions = 4, p = 1):
        self.window_size = 512
        self.observation_space = gym.spaces.Box(np.array([0,0]),np.array([1,1]), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(nb_actions)

        self.action_map = [[0,0.01],[0.01,0],[-0.01,0],[0,-0.01]]

        self.opt_tree_depth = opt_tree_depth
        self.p = p
        self.obs_step = (1 / (self.p + 1)) ** self.opt_tree_depth
        self.all_p = np.linspace(0, 1, p + 2)[1:-1]

        self.init_random_partition()

        self.state = self.observation_space.sample()

    def init_random_partition(self):
        self.rnd_partition, self.dt = partition(
            self.obs_step, self.opt_tree_depth, self.p, self.all_p, self.action_space.n
        )

    def plot_partition(self):
        self.fig = plt.pcolormesh(
            np.linspace(0, 1, (self.p + 1) ** (self.opt_tree_depth) + 1),
            np.linspace(0, 1, (self.p + 1) ** (self.opt_tree_depth) + 1),
            self.rnd_partition,
            cmap="Blues",
        )
        plt.savefig("optim_tree_" + str(self.opt_tree_depth) + ".pdf")

    def reset(self, seed = None, return_info = False, options=None):
        super().reset(seed=seed)
        self.state = self.observation_space.sample()
        return self.state


    def step(self, action):
        opt_action = traverse_tree(self.state, self.dt)
        if opt_action == action:
            reward = 1
        else: reward = 0

        self.state += self.action_map[action]
        self.state = np.clip(self.state, 0 , 1)

        done = False

        return self.state, reward, done, {}

    def render_(self):
        self.plot_partition()
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
