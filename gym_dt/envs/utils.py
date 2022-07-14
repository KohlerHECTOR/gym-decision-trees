import numpy as np

def cartesian_prod(x, y):
    return np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])


def recursive_2D_dt_partition(domains, depth, opt_tree_depth, all_p, nb_actions):
    # global PARTITION
    if depth == opt_tree_depth:
        a = np.random.randint(nb_actions)
        return ActionNode(a)
    else:
        random_dim = np.random.randint(2)
        true_vals = (
            all_p * (domains[random_dim][1] - domains[random_dim][0])
            + domains[random_dim][0]
        )
        random_val = np.random.choice(true_vals)
        left = domains.copy()
        right = domains.copy()
        left[random_dim] = [domains[random_dim][0], random_val]
        right[random_dim] = [random_val, domains[random_dim][1]]
        depth += 1
        # left_right = np.random.randint(2)
        child_left = recursive_2D_dt_partition(
            left, depth, opt_tree_depth, all_p, nb_actions
        )
        child_right = recursive_2D_dt_partition(
            right, depth, opt_tree_depth, all_p, nb_actions
        )

        return InfoNode(random_dim, random_val, child_left, child_right)
        # return {"left": child_left, "right": child_right}


def partition(obs_step, opt_tree_depth, p, all_p, nb_actions):
    PARTITION = np.zeros(
        (
            len(np.linspace(0, 1, (p + 1) ** (opt_tree_depth) + 1)[:-1]),
            len(np.linspace(0, 1, (p + 1) ** (opt_tree_depth) + 1)[:-1]),
        )
    )
    tree = recursive_2D_dt_partition(
        [[0, 1], [0, 1]],
        depth=0,
        opt_tree_depth=opt_tree_depth,
        all_p=all_p,
        nb_actions=nb_actions,
    )
    for x in range(len(PARTITION)):
        for y in range(len(PARTITION)):
            PARTITION[x][len(PARTITION) - 1 - y] = traverse_tree_build(
                [x * obs_step, y * obs_step], tree
            )
    return PARTITION.T, tree


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



class InfoNode:
    def __init__(self, feature, value, left, right):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right


class ActionNode:
    def __init__(self, action):
        self.action = action
