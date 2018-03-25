import random
import tensorflow as tf
import numpy as np
from collections import namedtuple


class Game:
    """Defines state and rules for single game"""

    _state = None

    def render(self):
        """Renders state in human-readable form"""
        pass

    def actions(self):
        """Returns all actions possible to do"""
        return []

    def take_action(self, action):
        """Takes action and returns tuple:
            (new_state, game_over, reward)
            Typical reward values are:
                1 = win
                0 = draw
                -1 = lose

        Keyword arguments:
        action -- action which should be taken,
            element of self.actions()"""
        return (self, True, 0)

    @property
    def state(self):
        return self._state


class Kik(Game):
    State = namedtuple('State', ['player', 'board'])
    _state = State(1, [0] * 9)
    """State is player to move, and row based 3x3 matrix,
        where for each element:
            0  = empty field
            1  = O
            -1 = X"""

    def render(self):
        signs = ['O' if x is 1 else 'X' if x is -1 else ' '
                 for x in self._state.board]
        rendered = """{}|{}|{}
-+-+-
{}|{}|{}
-+-+-
{}|{}|{}""".format(*signs)
        print(rendered)

    def actions(self):
        return [i for i, s in enumerate(self._state.board) if s is 0]

    def _check_win(self):
        p, s = self._state
        p = -p

        # horizontal
        for i in range(0, 3):
            if p == s[i*3] and p == s[i*3+1] and p == s[i*3+2]:
                return True
        # vertical
        for i in range(0, 3):
            if p == s[i] and p == s[i+3] and p == s[i+6]:
                return True
        # diags
        if p == s[0] and p == s[4] and p == s[8]:
            return True
        if p == s[2] and p == s[4] and p == s[6]:
            return True

        return False

    def _done(self):
        return self._check_win() or self._state.board.count(0) is 0

    def take_action(self, action):
        i = action
        assert i in range(0, 9), "Invalid action, out of 0..9 range!"
        assert self._state.board[i] is 0, "Invalid action, field not empty"

        state = self._state.board[:]
        state[i] = self._state.player
        r = Kik()
        r._state = self.State(-self._state.player, state)

        if r._check_win():
            return (r, True, 1)
        elif r._state.count(0) is 0:
            return (r, True, 0)
        else:
            return (r, False, 0)

    def play(self, o_agent, x_agent):
        """Plays game starting from current position,
            taking actions using o_agent for O, and
            x_agent for X; Both agents are just callables
            taking Kik with current object as argument,
            and returning action to take. Returns which
            player wins (1 = O, -1 = X, 0 = draw), and
            actions history"""

        state = self
        winner = 0
        history = []

        while not state._done():
            p, _ = state._state
            agent = o_agent if p is 1 else x_agent
            action = agent(state)
            history.append(action)
            state, _, win = state.take_action(action)
            if win is 1:
                winner = p

        return (winner, history)


def random_agent(state):
    return random.choice(state.actions())


class RLAgent(object):
    _history = []
    _temp = 0.9

    def __init__(self, sess):
        self.sess = sess

        with tf.name_scope('value_network'):
            self.input = tf.placeholder(
                dtype=tf.float32, shape=(None, 9), name='input')
            h1 = tf.layers.dense(self.input, 16, activation=tf.nn.relu)
            h2 = tf.layers.dense(h1, 8, activation=tf.nn.relu)
            self.output = tf.layers.dense(h2, 1)
            self.foutput = tf.squeeze(self.output, axis=1, name='flatten_output')

        with tf.name_scope('training'):
            self.expected = tf.placeholder(
                dtype=tf.float32, shape=(None, 1), name='expected')
            cost = tf.reduce_mean((self.output - self.expected) ** 2)
            self.train = tf.train.AdamOptimizer(epsilon=0.01).minimize(cost)

    def reset(self, temp=0.9):
        self._history = []
        self._temp = temp

    def __call__(self, state):
        player = state.state.player

        def uniformize_board(board):
            return [i * player for i in board]

        actions = state.actions()
        # Checking if is there winning move
#        rewards = [state.take_action(a)[2] for a in actions]
#        if True in rewards:
            # Taking winning action
#            idx = rewards.index(True)
#            action = actions[idx]
#            state = np.array([uniformize_board(
#                state.take_action(action)[0].state.board)])
#            value = self.sess.run(self.foutput, feed_dict={
#                self.input: state
#            })[0]
#            state = state[0]
        if random.random() > self._temp:
            # Taking random action
            idx = random.choice(range(len(actions)))
            action = actions[idx]
            state = np.array([uniformize_board(
                state.take_action(action)[0].state.board)])
            value = self.sess.run(self.foutput, feed_dict={
                self.input: state
            })[0]
            state = state[0]
        else:
            # Taking best action
            states = np.array([uniformize_board(
                                    state.take_action(a)[0].state.board)
                               for a in actions])
            values = self.sess.run(self.foutput, feed_dict={
                self.input: states
            })
            idx = np.argmax(values)
            action = actions[idx]
            state = states[idx]
            value = values[idx]

        self._history.append((player, state, value))
        return action

    def _estimate_new_values(self, moves, reward, dr=0.9, lr=0.1):
        """dr is discount ratio, lr is learning rate"""
        states = np.array([s for s, _ in moves])
        reward *= lr
        discounts = np.array(list(reversed(
            [dr ** idx for idx in range(len(moves))]
        )))
        values = np.array([v for _, v in moves])
        values = discounts * reward + values

        return states.reshape((-1, 9)), values.reshape((-1, 1))

    def learn(self, winner, dr, lr):
        winning = [(s, v) for p, s, v in self._history if p == winner]
        losing = [(s, v) for p, s, v in self._history if p == -winner]
        # We dont wanna drawing all time, so draws would be also trained
        # with small penalty
        drawing = [(s, v) for p, s, v in self._history if winner == 0]

        winning = self._estimate_new_values(winning, 0.1, dr, lr)
        losing = self._estimate_new_values(losing, -1.0, dr, lr)
        drawing = self._estimate_new_values(drawing, -0.1, dr, lr)

        states = np.concatenate((winning[0], losing[0], drawing[0]))
        values = np.concatenate((winning[1], losing[1], drawing[1])) \
            .reshape((-1, 1))

        self.sess.run(self.train, feed_dict={
            self.input: states,
            self.expected: values
        })


os, xs, draws = (0, 0, 0)
with tf.Session() as sess:
    agent = RLAgent(sess)
    sess.run(tf.global_variables_initializer())

    for i in range(100000):
        agent.reset(temp=0.8)
        w, history = Kik().play(agent, agent)
        print("Training #{}: {}, winner: {}".format(i, history, w))
        agent.learn(w, 1.0, 0.3)

        if w is 1:
            os += 1
        elif w is -1:
            xs += 1
        else:
            draws += 1

    print("O wins: {}\nX wins: {}\ndraws: {}".format(os, xs, draws))

    for i in range(10):
        agent.reset(temp=1.0)
        w, history = Kik().play(agent, agent)
        print("Trained #{}: {}, winner: {}".format(i, history, w))

    for i in range(10):
        agent.reset(temp=1.0)
        w, history = Kik().play(agent, random_agent)
        print("O vs Rand #{}: {}, winner: {}".format(i, history, w))

    for i in range(10):
        agent.reset(temp=1.0)
        w, history = Kik().play(random_agent, agent)
        print("X vs Rand #{}: {}, winner: {}".format(i, history, w))


