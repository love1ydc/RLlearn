from DQN_env import env
from DQN import DeepQN

def run_model():
    step = 0
    for episode in range(100):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    # maze game
    env = env()
    RL = DeepQN(env.n_actions, env.n_features )
    env.after(100, run_model())
    env.mainloop()
    RL.plot_cost()