from utils import policy_evaluation, estimate_stationary_distribution
from algorithms import ABQalgorithm, GQalgorithm
from environment.counterexample import CounterExample
from value_functions import CounterExampleValueFunction

import matplotlib.pyplot as plt
import numpy as np


if __name__=='__main__':

    env = CounterExample()
    nA = env.nA
    nS = env.nS

    discount_factor = 0.9

    zeta = 0.5  # zeta for ABQ(zeta)
    alpha = 0.001
    beta = 0.001

    lamd = 0.5  # lambda for GQ(lambda)

    n_runs = 5

    value_function = CounterExampleValueFunction()

    target_policy = np.array([[0.1, 0.9], [0.1, 0.9]])
    behaviour_policy = np.array([[0.1, 0.9], [0.9, 0.1]])

    mu = estimate_stationary_distribution(env, behaviour_policy)

    trueQ, trueV = policy_evaluation(env, target_policy, discount_factor=discount_factor)

    num_episodes = 1500

    ABQ_mse_err = np.zeros((n_runs, num_episodes))
    ABQ_theta_err = np.zeros((n_runs, num_episodes))

    GQ_mse_err = np.zeros((n_runs, num_episodes))
    GQ_theta_err = np.zeros((n_runs, num_episodes))

    theta = np.zeros(value_function.param_shape)
    h_v = np.zeros(value_function.param_shape)


    for r in range(n_runs):
        theta_ABQ = theta.copy()
        theta_GQ = theta.copy()
        h_vec = h_v.copy()
        w_t = h_v.copy()

        theta_GQ = theta.copy()

        for episode in range(num_episodes):
            state = env.reset()

            # print theta_ABQ
            ABQ_theta_err[r, episode] = np.linalg.norm(theta_ABQ)
            GQ_theta_err[r, episode] = np.linalg.norm(theta_GQ)
            # print np.linalg.norm(theta_ABQ)

            q_ABQ = np.zeros([nS, nA])
            q_GQ = np.zeros([nS, nA])
            for s in range(nS):
                for a in range(nA):
                    q_ABQ[s, a] = np.dot(theta_ABQ, value_function.feature(s, a))
                    q_GQ[s, a] = np.dot(theta_GQ, value_function.feature(s, a))

            error = np.sum(mu.reshape((nS, nA)) * np.power(trueQ - q_ABQ, 2))
            ABQ_mse_err[r, episode] = error / np.sum(mu.reshape((nS, nA)) * np.power(trueQ, 2))

            err = np.sum(mu.reshape((nS, nA)) * np.power(trueQ - q_GQ, 2))
            GQ_mse_err[r, episode] = err / np.sum(mu.reshape((nS, nA)) * np.power(trueQ, 2))

            etrace = np.zeros(shape=theta.shape)
            et = np.zeros(shape=theta.shape)


            while True:

                action_probs = behaviour_policy[state]
                action = np.random.choice(np.arange(nA), p=action_probs)

                next_state, reward, done, _ = env.step(action)

                theta_ABQ, etrace, h_vec = ABQalgorithm(env, value_function, theta_ABQ, etrace, h_vec, alpha, beta, discount_factor, zeta, target_policy, behaviour_policy, state, action, reward, next_state, done)

                theta_GQ, et, w_t = GQalgorithm(env, value_function, theta_GQ, lamd, et, w_t, alpha, beta, discount_factor, target_policy, behaviour_policy, state, action, reward, next_state, done)

                if done:
                    break
                state = next_state
            env.close()

    ABQ_mse_mean = np.mean(ABQ_mse_err, axis=0)
    ABQ_mse_std = np.std(ABQ_mse_err, axis=0)

    GQ_mse_mean = np.mean(GQ_mse_err, axis=0)
    GQ_mse_std = np.std(GQ_mse_err, axis=0)

    plt.figure(figsize=(15, 10))
    plt.ylabel('normalized MSE', fontsize=25)
    plt.xlabel('episode', fontsize=25)
    plt.xlim((-20, 1520))
    # plt.xlabel(r'$\zeta$ for ABQ($\zeta$)', fontsize=25)
    # plt.title(title, fontsize=25)
    plt.fill_between(np.arange(len(ABQ_mse_mean)), ABQ_mse_mean - ABQ_mse_std, ABQ_mse_mean + ABQ_mse_std, alpha=0.3, color='b')
    plt.plot(np.arange(len(ABQ_mse_mean)), ABQ_mse_mean, color="b", label="ABQ")

    plt.fill_between(np.arange(len(GQ_mse_mean)), GQ_mse_mean - GQ_mse_std, GQ_mse_mean + GQ_mse_std, alpha=0.3,color='g')
    plt.plot(np.arange(len(GQ_mse_mean)), GQ_mse_mean, color="b", label="GQ")

    plt.legend(loc='best')

    ABQ_theta_mean = np.mean(ABQ_theta_err, axis=0)
    ABQ_theta_std = np.std(ABQ_theta_err, axis=0)

    GQ_theta_mean = np.mean(GQ_theta_err, axis=0)
    GQ_theta_std = np.std(GQ_theta_err, axis=0)

    plt.figure(figsize=(15, 10))
    plt.ylabel(r'$||\theta||$', fontsize=25)
    plt.xlabel('episode', fontsize=25)
    plt.xlim((-20, 1520))
    # plt.xlabel(r'$\zeta$ for ABQ($\zeta$)', fontsize=25)
    # plt.title(title, fontsize=25)
    plt.fill_between(np.arange(len(ABQ_theta_mean)), ABQ_theta_mean - ABQ_theta_std, ABQ_theta_mean + ABQ_theta_std, alpha=0.3,
                     color='b')
    plt.plot(np.arange(len(ABQ_theta_mean)), ABQ_theta_mean, color="b", label="ABQ")

    plt.fill_between(np.arange(len(GQ_theta_mean)), GQ_theta_mean - GQ_theta_std, GQ_theta_mean + GQ_theta_std,alpha=0.3, color='g')
    plt.plot(np.arange(len(GQ_theta_mean)), GQ_theta_mean, color="b", label="GQ")

    plt.legend(loc='best')

    plt.show()

