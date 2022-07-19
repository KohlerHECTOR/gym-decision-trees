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

s = env.reset()
done = False
while not done:
  s,r,done,info = env.step(env.action_space.sample())

env.plot_policy() #saves a plot of the optimal policy
```

# TODO

Add rendering.
