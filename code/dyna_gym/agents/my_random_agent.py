"""
A Random Agent given as an example
"""

from gymnasium import spaces
import dyna_gym.utils.utils as utils


class MyRandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def reset(self, p=None):
        """
        Reset the attributes.
        Expect to receive them in the same order as init.
        p : list of parameters
        """
        if p is not None:
            utils.assert_types(p, [spaces.discrete.Discrete])
            self.__init__(p[0])

    def display(self):
        """
        Display infos about the attributes.
        """
        print('Displaying Random agent:')
        print('Action space       :', self.action_space)

    def act(self, observation=None, reward=None, done=None):
        # cur obs of env
        # reward recieved from previous time step
        # whether the env reached a terminal
        return self.action_space.sample()


from gymnasium.spaces import Discrete


class MyRandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def reset(self, p=None):
        """
        Reset the attributes.
        Expect to receive them in the same order as init.
        p : list of parameters
        """
        if p is not None:
            assert isinstance(p[0], Discrete), "Expected a Discrete action space."
            self.__init__(p[0])

    def display(self):
        """
        Display infos about the attributes.
        """
        print('Displaying Random agent:')
        print('Action space       :', self.action_space)

    def act(self, observation=None, reward=None, done=None):
        return self.action_space.sample()


def main():
    action_space = Discrete(3)  # Create an action space with 3 possible actions (0, 1, 2)
    agent = MyRandomAgent(action_space)
    agent.display()
    action = agent.act()
    print(f'Random action: {action}')
    agent.reset([Discrete(5)])
    agent.display()
    action = agent.act()
    print(f'Random action after reset: {action}')


if __name__ == "__main__":
    main()
