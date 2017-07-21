from utils import  policy_evaluation, estimate_stationary_distribution
from algorithms import ABQalgorithm, nuFunc
from environment.baird import Baird
from value_functions import BairdValueFunction

import matplotlib.pyplot as plt
import numpy as np



if __name__=='__main__':

    env = Baird()
    nA = env.nA
    nS = env.nS
    value_function = BairdValueFunction()

    ### set target_policy and  behaviour_policy ##
    target_policy = np.array([[0., 1.], [0., 1.], [0., 1.], [0., 1.], [0., 1.], [0., 1.], [0., 1.]])
    behaviour_policy = np.array([[6./7, 1./7], [6./7, 1./7], [6./7, 1./7], [6./7, 1./7], [6./7, 1./7], [6./7, 1./7], [6./7, 1./7]])

    ### initialize the parameter vector w/ theta ##
    theta = np.ones(value_function.param_shape)
    theta[6] = 10.
    theta[14] = 10.

    discount_factor = 0.99
    zeta = 1.
    alpha = 0.005
    beta = 0.01

    ## for multiple runs
    n_runs = 5

    print 'estimating stationary distribution ..'
    mu = estimate_stationary_distribution(env, behaviour_policy)


    # to compute MSE error
    trueQ, trueV = policy_evaluation(env, target_policy, discount_factor=discount_factor)

    num_episodes = 1000

    theta_err = np.zeros((n_runs, num_episodes))

    print 'multiple runs...'
    for r in range(n_runs):
        theta_ABQ = theta.copy()

        for episode in range(num_episodes):
            state = env.reset()

            # print theta_ABQ
            theta_err[r, episode] = np.linalg.norm(theta_ABQ)


            q = np.zeros([nS, nA])
            for s in range(nS):
                for a in range(nA):
                    q[s, a] = np.dot(theta, value_function.feature(s, a))

            # error = np.sum(mu.reshape((nS, nA)) * np.power(trueQ - q, 2))
            # mse_err[r, episode] = error
            # print 'error: {}'.format(error)

            etrace = np.zeros(shape=theta.shape)
            h_vec = np.zeros(shape=theta.shape)

            while True:

                action_probs = behaviour_policy[state]
                action = np.random.choice(np.arange(nA), p=action_probs)

                next_state, reward, done, _ = env.step(action)

                theta_ABQ, etrace, h_vec = ABQalgorithm(env, value_function, theta_ABQ, etrace, h_vec, alpha, beta, discount_factor, zeta, target_policy, behaviour_policy, state, action, reward, next_state, done)

                if isinstance(env, Baird):
                    if np.random.binomial(1, 0.01) == 1:
                        done = True

                if done:
                    break
                state = next_state
            env.close()

    ##### plot errors ######
    mse_err_mean = np.mean(theta_err, axis=0)
    mse_err_std = np.std(theta_err, axis=0)

    plt.figure(figsize=(15, 10))
    plt.ylabel(r'$||\theta||_2$', fontsize=25)
    plt.xlabel('episode', fontsize=25)
    # plt.xlabel(r'$\zeta$ for ABQ($\zeta$)', fontsize=25)
    # plt.title(title, fontsize=25)
    plt.fill_between(np.arange(len(mse_err_mean)), mse_err_mean - mse_err_std, mse_err_mean + mse_err_std, alpha=0.3,
                     color='b')
    plt.plot(np.arange(len(mse_err_mean)), mse_err_mean, color="b", label="Multistep")

    plt.legend(loc='best')
    plt.show()