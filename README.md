# About the MDP.
You can see the mdp as a 2d plane where the agent can move in the four cardinal directions by some step size.
At first, a decision tree policy of given depth is generated at random. For example if the specified depth is 2 we could generate:

            x<=0.5?
            /       \ 
          x<=0.25?   y<=0.5?
          /   \        /  \
         up  down     left right

State Space: Continuous 2D plane : (x,y) in [0,1] x [0,1]. <br/>
Action Space: {up, right, left, down}. <br/>
Reward: Binary: 1 if the agent follows the optimal decision tree policy. <br/>
Transition: are vector sums:<br/>
for example if the action is "up"<br/>
the deterministic transition is as follows:<br/>

s_next = s + move <=> (x_next, y_next ) = (x, y) + (0, step_size).<br/>
When the agent should be displaced out of bounds, it transitions to its current state (s_next = s). <br/>

Maximum episode lengh is 500 !<br/>

# Install.

For a directory looking like this:
```
  directory/
    rl-library/
      main.py
      virtual_env/
```
You can install as follows:
```console
user@user:~$ cd directory
user@user:~/directory$ git clone https://github.com/KohlerHECTOR/gym-decision-trees.git
user@user:~/directory/rl-library$ cd rl-library
user@user:~/directory/rl-library$ source virtual_env/bin/activate
user@user:~/directory/rl-library$ pip install -e ../gym-decision-trees/
```

# Run.

You can create an instance of the environment as follows:
```python
import gym_dt
import gym

env = gym.make('gym_dt/DecisionTreeEnv-v0',
                **dict(opt_tree_depth=4,
                       p=1
                       )))

env.plot_policy() #saves a plot of the optimal policy
s = env.reset()
done = False
while not done:
  s,r,done,info = env.step(env.action_space.sample())


```

# TODO

Add rendering.
