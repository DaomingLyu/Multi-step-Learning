from utils import policy_evaluation, estimate_stationary_distribution
from algorithms import  nuFunc, ABQalgorithm
from environment.counterexample import CounterExample
from value_functions import CounterExampleValueFunction

import matplotlib.pyplot as plt
import numpy as np


if __name__=='__main__':

    env = CounterExample()
    nA = env.nA
    nS = env.nS

    discount_factor = 0.9
    zeta = 0.5
    alpha = 0.001
    beta = 0.001

    n_runs = 5

    value_function = CounterExampleValueFunction()

    target_policy = np.array([[0.1, 0.9], [0.1, 0.9]])
    behaviour_policy = np.array([[0.1, 0.9], [0.9, 0.1]])

    mu = estimate_stationary_distribution(env, behaviour_policy)

    trueQ, trueV = policy_evaluation(env, target_policy, discount_factor=discount_factor)

    num_episodes = 1500

    mse_err = np.zeros((n_runs, num_episodes))
    theta_err = np.zeros((n_runs, num_episodes))

    theta = np.zeros(value_function.param_shape)


    for r in range(n_runs):
        theta_ABQ = theta.copy()

        for episode in range(num_episodes):
            state = env.reset()

            # print theta_ABQ
            theta_err[r, episode] = np.linalg.norm(theta_ABQ)
            # print np.linalg.norm(theta_ABQ)

            q = np.zeros([nS, nA])
            for s in range(nS):
                for a in range(nA):
                    q[s, a] = np.dot(theta_ABQ, value_function.feature(s, a))

            error = np.sum(mu.reshape((nS, nA)) * np.power(trueQ - q, 2))
            mse_err[r, episode] = error / np.sum(mu.reshape((nS, nA)) * np.power(trueQ, 2))

            etrace = np.zeros(shape=theta.shape)
            h_vec = np.zeros(shape=theta.shape)

            while True:

                action_probs = behaviour_policy[state]
                action = np.random.choice(np.arange(nA), p=action_probs)

                next_state, reward, done, _ = env.step(action)

                theta_ABQ, etrace, h_vec = ABQalgorithm(env, value_function, theta_ABQ, etrace, h_vec, alpha, beta, discount_factor, zeta, target_policy, behaviour_policy, state, action, reward, next_state, done)

                if done:
                    break
                state = next_state
            env.close()

    mse_err_mean = np.mean(mse_err, axis=0)
    mse_err_std = np.std(mse_err, axis=0)

    plt.figure(figsize=(15, 10))
    plt.ylabel('normalized MSE', fontsize=25)
    plt.xlabel('episode', fontsize=25)
    plt.xlim((-20, 1520))
    # plt.xlabel(r'$\zeta$ for ABQ($\zeta$)', fontsize=25)
    # plt.title(title, fontsize=25)
    plt.fill_between(np.arange(len(mse_err_mean)), mse_err_mean - mse_err_std, mse_err_mean + mse_err_std, alpha=0.3, color='b')
    plt.plot(np.arange(len(mse_err_mean)), mse_err_mean, color="b", label="Multistep")

    plt.legend(loc='best')

    theta_err_mean = np.mean(theta_err, axis=0)
    theta_err_std = np.std(theta_err, axis=0)

    plt.figure(figsize=(15, 10))
    plt.ylabel(r'$||\theta||$', fontsize=25)
    plt.xlabel('episode', fontsize=25)
    plt.xlim((-20, 1520))
    # plt.xlabel(r'$\zeta$ for ABQ($\zeta$)', fontsize=25)
    # plt.title(title, fontsize=25)
    plt.fill_between(np.arange(len(theta_err_mean)), theta_err_mean - theta_err_std, theta_err_mean + theta_err_std, alpha=0.3,
                     color='b')
    plt.plot(np.arange(len(theta_err_mean)), theta_err_mean, color="b", label="Multistep")

    plt.legend(loc='best')

    plt.show()

