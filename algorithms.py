import numpy as np

def nuFunc(zeta, pi_policy, mu_policy, state, action):
###############################################
#         Eq.(8), try to compute \nu
###############################################

    pi_s_a = pi_policy[state, action]
    mu_s_a = mu_policy[state, action]
    ########### compute \psi_{0} and \psi_{max} #########################
    if pi_s_a >= mu_s_a:
        psi_0 = 1.0 / np.max(pi_policy[state, :])
        psi_max = 1.0 / np.min(pi_policy[state, :])
    else:
        psi_0 = 1.0 / np.max(mu_policy[state, :])
        psi_max = 1.0 / np.min(mu_policy[state, :])

    psi = 2.0 * zeta * psi_0 + max(0, (2.0 * zeta - 1)) * (psi_max - 2.0 * psi_0)

    nu = min(psi, 1.0 / (max(pi_s_a, mu_s_a)))

    return nu



def ABQalgorithm(env, value_function, theta, etrace, h_vec, alpha, beta, gamma, zeta, pi_policy, mu_policy, state, action, r, next_state, done):
    ########### ABQ(zeta) algorithm #########################
    nA = env.nA
    nS = env.nS

    phi = value_function.feature(state, action)

    nu_current = nuFunc(zeta, pi_policy, mu_policy, state, action)

    if done:
        phi_bar_next = 0.
        phi_tilde_next = 0.
    else:
        phi_bar_next = np.sum([pi_policy[next_state, a] * value_function.feature(next_state, a) for a in np.arange(nA)], axis=0)
        phi_tilde_next = np.sum([nuFunc(zeta, pi_policy, mu_policy, next_state, a)*pi_policy[next_state, a] * value_function.feature(next_state, a) for a in np.arange(nA)], axis=0)

    ########### Eq.(27) compute \delta #########################
    delta = r + gamma * np.dot(theta, phi_bar_next) - np.dot(theta, phi)

    ########### Eq.(29) compute trace vector #########################
    etrace = gamma * nu_current * pi_policy[state, action] * etrace
    etrace = etrace + phi

    ########### Eq.(30) compute w #########################
    theta = theta + alpha * (delta * etrace - gamma * np.dot(etrace, h_vec) * (phi_bar_next - phi_tilde_next))

    ########### Eq.(31) compute extra vector h #########################
    h_vec = h_vec + beta * (delta * etrace - np.dot(h_vec, phi) * phi)

    return theta, etrace, h_vec

def GQalgorithm(env, value_function, theta, lamd, etrace, w_t, alpha, beta, gamma, pi_policy, mu_policy, state, action, r, next_state, done):
    ########### GQ(lambda) algorithm #########################
    nA = env.nA
    nS = env.nS
    phi = value_function.feature(state, action)

    if done:
        phi_bar_next = 0.
    else:
        phi_bar_next = np.sum([pi_policy[next_state, a] * value_function.feature(next_state, a) for a in np.arange(nA)], axis=0)

    ########### compute TD error \delta #########################
    delta = r + gamma * np.dot(theta, phi_bar_next) - np.dot(theta, phi)

    ########### compute importance ratio #########################
    rho = pi_policy[state, action] / mu_policy[state, action]

    ########### compute trace vector #########################
    etrace = rho * gamma * lamd * etrace
    etrace = etrace + phi

    ########### compute parameter vector #########################
    theta = theta + alpha * ( delta*etrace - gamma* (1. - lamd)*(np.dot(etrace, w_t))*phi_bar_next)

    w_t = w_t + beta*(delta*etrace - (np.dot(phi, w_t))*phi)

    return theta, etrace, w_t
