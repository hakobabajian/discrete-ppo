import numpy as np
from agent import Agent
from environment import Environment
from timeit import default_timer


if __name__ == '__main__':
    kenv = Environment(max_runtime=30)
    N = 20
    batch_size = 5
    n_epochs = 4

    agent = Agent(n_actions=kenv.actions,
                  batch_size=batch_size,
                  n_epochs=n_epochs,
                  input_dims=kenv.observation_space_shape)
    n_games = 100

    figure_file = 'plots/cartpole.png'

    best_score = kenv.max_punishment
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = kenv.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(np.array(observation, dtype=np.float32))
            observation_, reward, done = kenv.step(action, observation)
            n_steps += 1
            score += reward
            agent.remember(np.array(observation, dtype=np.float32), action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'time_steps', n_steps,
              'learning_steps', learn_iters)
