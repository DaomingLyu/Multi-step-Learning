import numpy as np
from environment.baird import Baird


def estimate_stationary_distribution(env, behaviour_policy):
    cnt = np.zeros((env.nS, env.nA))

    for episode in range(10000):
        state = env.reset()
        while True:
            action_probs = behaviour_policy[state]
            action = np.random.choice(np.arange(env.nA), p=action_probs)
            cnt[state, action] += 1
            next_state, _, done, _ = env.step(action)
            if isinstance(env, Baird):
                if np.random.binomial(1, 0.01) == 1:
                    done = True
            if done:
                break

            state = next_state
    cnt = cnt.astype('float')
    mu = cnt / cnt.sum()
    mu = mu.reshape(env.nS * env.nA)
    return mu


def policy_evaluation(env, policy, discount_factor=0.6, threshold=0.00001):

    nA = env.nA
    nS = env.nS

    V = np.zeros(nS)
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):

            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                for prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < threshold:
            break

    # compute state-action value
    Q = np.zeros([nS, nA])
    for s in range(nS):
        for a in range(nA):
            q = 0
            for prob, next_state, reward, done in env.P[s][a]:
                q += prob * (reward + discount_factor * V[next_state])
            Q[s, a] = q
    return Q, V


