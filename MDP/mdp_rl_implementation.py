from copy import deepcopy
import random
import numpy as np
import math


num2action = {0: "UP", 
                1: "DOWN",
                2: "RIGHT",
                3: "LEFT"}
def Bellman_Eq_Calc(mdp, i, j, U):
    maxUtility = -math.inf
    maxAction = ""
    for (action, prob) in mdp.transition_function.items():
        for i in range(4):
            nextState = mdp.step((i,j), action)
            temp = prob[i]*U[nextState[0],nextState[1]]
            if temp > maxUtility:
                maxUtility = temp
                maxAction = action
    bellmanEq = mdp.board[i,j] + mdp.gamma * maxUtility
    return (maxAction,bellmanEq)


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # TODO:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #

    # ====== YOUR CODE: ======
    U = U_init
    Uprime = U_init
    delta = 0
    #do
    U = Uprime
    delta = 0
    states = mdp
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            U_prime[i][j] = Bellman_Eq_Calc(mdp, i, j, U)[1]
            if abs(Uprime[i][j] - U[i][j]) > delta:
                delta = abs(Uprime[i][j] - U[i][j])
    while delta > epsilon*(1-mdp.gamma)/mdp.gamma:
        U = Uprime
        delta = 0
        states = mdp
        for i in range(mdp.num_row):
            for j in range(mdp.num_col):
                Uprime[i][j] = one_state_mdp(mdp, i, j, U)[1]
                if abs(Uprime[i][j] - U[i][j]) > delta:
                    delta = abs(Uprime[i][j] - U[i][j])
    return U
    # ========================


def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ======
    Upolicy = np.zeros(mdp.num_row, mdp.num_col)
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
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
    qTable = np.zeros(mdp.num_row, mdp.num_col, len(mdp.transion_function.keys()))
    for episode in range(total_episodes):
        state = init_state
        step = 0
        done = False

        for step in range(max_steps):
            tradeoff = random.uniform(0,1)

            if tradeoff >epsilon:
                actionIdx = np.argmax(qTable[state[0],state[1],:])
                action = num2action[actionIdx]
            else:
                actionIdx = random(0,4)
                action = num2action[actionIdx]

            newState = mdp.step(state, action)
            qTable[state[0], state[1], actionIdx] = qTable[state[0], state[1], actionIdx] + learning_rate * (mdp.board[state[0],state[1]] + mdp.gamma * np.max(qTable[newState[0],newState[1],:]) - qTable[state[0], state[1], actionIdx])

            state = newState
            if state in mdp.terminal_states:
                break
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode)
    
    # ========================


def q_table_policy_extraction(mdp, qtable):
    # TODO:
    # Given the mdp and the Qtable:
    # return: the policy corresponding to the Qtable
    #

    # ====== YOUR CODE: ======
    Qpolicy = np.zeros(mdp.num_row, mdp.num_col)
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            Qpolicy[i,j] = num2action[np.argmax(qtable[i,j,:])]
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
