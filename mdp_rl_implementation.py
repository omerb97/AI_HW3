from copy import deepcopy
import random
import numpy as np
import math

def Index2State(idx, num_row):
    row = idx//num_row
    col = idx % num_row
    return (row, col)

def State2Index(row,col, num_row):
    return row*num_row+col

num2action = {0: "UP", 
                1: "DOWN",
                2: "RIGHT",
                3: "LEFT"}
def Bellman_Eq_Calc(mdp, i, j, Uregular):
    maxUtility = -math.inf
    maxAction = ""
    for (action, prob) in mdp.transition_function.items():
        curr = 0
        for k in range(4):
            nextState = mdp.step((i,j), num2action[k])
            curr += prob[k]*Uregular[nextState[0]][nextState[1]]
        if curr > maxUtility:
            maxUtility = curr
            maxAction = action
    bellmanEq = float(mdp.board[i][j]) + mdp.gamma * maxUtility
    return (maxAction,bellmanEq)


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # TODO:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #

    # ====== YOUR CODE: ======
    Uregular = U_init
    Uprime = U_init
    delta = 0
    #do
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if mdp.board[i][j] == "WALL":
                continue
            Uprime[i][j] = Bellman_Eq_Calc(mdp, i, j, Uregular)[1]
            if abs(Uprime[i][j] - Uregular[i][j]) > delta:
                delta = abs(Uprime[i][j] - Uregular[i][j])
            print(abs(Uprime[i][j] - Uregular[i][j]))
    while delta >= epsilon*(1-mdp.gamma)/mdp.gamma:
        print("ehlloo")
        mdp.print_utility(Uregular)
        Uregular = Uprime
        delta = 0
        for i in range(mdp.num_row):
            for j in range(mdp.num_col):
                if mdp.board[i][j] == "WALL":
                    continue
                Uprime[i,j] = Bellman_Eq_Calc(mdp, i, j, Uregular)[1]
                if abs(Uprime[i][j] - Uregular[i][j]) > delta:
                    delta = abs(Uprime[i][j] - Uregular[i][j])
    return Uregular
    # ========================


def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ======
    Upolicy = np.zeros((mdp.num_row, mdp.num_col), dtype=str)
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if mdp.board[i][j] == "WALL":
                continue
            Upolicy[i,j] = Bellman_Eq_Calc(mdp, i, j, U)[0]
    return Upolicy
    # ========================


def q_learning(mdp, init_state, total_episodes=10000, max_steps=999, learning_rate=0.7, epsilon=1.0,
                      max_epsilon=1.0, min_epsilon=0.01, decay_rate=0.8):
    # TODO:
    # Given the mdp and the Qlearning parameters:
    # total_episodes - number of episodes to run for the learning algorithm
    # max_steps - for each episode, the limit for the number of steps
    # learning_rate - the "learning rate" (alpha) for updating the table values
    # epsilon - the starting value of epsilon for the exploration-exploitation choosing of an action
    # max_epsilon - max value of the epsilon parameter
    # min_epsilon - min value of the epsilon parameter
    # decay_rate - exponential decay rate for exploration prob
    # init_state - the initial state to start each episode from
    # return: the Qtable learned by the algorithm
    #

    # ====== YOUR CODE: ======
    qTable = np.zeros((mdp.num_row * mdp.num_col, len(mdp.transition_function.keys())))
    for episode in range(total_episodes):
        state = init_state
        step = 0
        done = False

        for step in range(max_steps):
            tradeoff = random.uniform(0,1)

            if tradeoff >epsilon:
                actionIdx = np.argmax(qTable[State2Index(state[0], state[1], mdp.num_row),:])
                action = num2action[actionIdx]
            else:
                actionIdx = random.randint(0,3)
                action = num2action[actionIdx]

            newState = mdp.step(state, action)
            qTable[State2Index(state[0], state[1], mdp.num_row), actionIdx] = qTable[State2Index(state[0], state[1], mdp.num_row), actionIdx] + learning_rate * (float(mdp.board[state[0]][state[1]]) + mdp.gamma * np.max(qTable[State2Index(state[0], state[1], mdp.num_row),:]) - qTable[State2Index(state[0], state[1], mdp.num_row), actionIdx])

            state = newState
            if state in mdp.terminal_states:
                break
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode)
    return qTable
    # ========================


def q_table_policy_extraction(mdp, qtable):
    # TODO:
    # Given the mdp and the Qtable:
    # return: the policy corresponding to the Qtable
    #

    # ====== YOUR CODE: ======
    Qpolicy = np.zeros((mdp.num_row, mdp.num_col), dtype=str)
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            Qpolicy[i,j] = num2action[np.argmax(qtable[State2Index(i,j,mdp.num_row),:])]
    return Qpolicy 
    # ========================


# BONUS

def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
