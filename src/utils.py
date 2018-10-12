
def run_episode(policy, env, max_steps):
    state = env.reset()
    rewards = []
    for step in range(max_steps):
        action = policy.act(state)
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
        if done:
            break

    return rewards

def calculate_returns(gamma, rewards):
    # calculate discounted return
    discounts = [gamma**i for i in range(len(rewards) + 1)]
    current_return = sum(
        [discount * reward for discount, reward in zip(discounts, rewards)])
    return current_return    