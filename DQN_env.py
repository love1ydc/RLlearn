import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 40   # pixels
MAZE_H = 2  # grid height
MAZE_W = 9  # grid width


class env(tk.Tk, object):
    def __init__(self):
        super(env, self).__init__()
        self.action_space = ['left', 'right', 'keep', 'sp_up','sp_down']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.title('env_overtake')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_env()

    def _build_env(self):
        self.canvas = tk.Canvas(self, bg='grey',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        # create grids
        # for c in range(0, MAZE_W * UNIT, UNIT):
        #     x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
        self.canvas.create_line(0,42,360,42,fill='yellow',dash=6)
        self.canvas.create_line(0, 38, 360, 38, fill='yellow', dash=6)
        # for r in range(0, MAZE_H * UNIT, UNIT):
        #     x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
        #     self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # hell
        hell1_center = origin + np.array([0, 0])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        # hell
        hell2_center = origin + np.array([UNIT*4, UNIT * 1])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')

        hell3_center = origin + np.array([UNIT * 7, 0])
        self.hell3 = self.canvas.create_rectangle(
            hell3_center[0] - 15, hell3_center[1] - 15,
            hell3_center[0] + 15, hell3_center[1] + 15,
            fill='black')
        # create oval
        oval_center = origin + np.array([UNIT * 7, UNIT * 1])
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        agent_center =origin+np.array([0,UNIT*1])
        self.agent = self.canvas.create_rectangle(
            agent_center[0] - 15, agent_center[1] - 15,
            agent_center[0] + 15, agent_center[1] + 15,
            fill='red')

        self.canvas.pack()



    def reset(self):
        self.update()
        time.sleep(0.1)
        self.canvas.delete(self.agent)
        origin = np.array([20, 20])
        agent_center = origin + np.array([0, UNIT * 1])
        self.agent = self.canvas.create_rectangle(
            agent_center[0] - 15, agent_center[1] - 15,
            agent_center[0] + 15, agent_center[1] + 15,
            fill='red')
        # return observation
        #print(np.array(self.canvas.coords(self.agent)))
        #print(np.array(self.canvas.coords(self.oval)[:2]))
        return (np.array(self.canvas.coords(self.agent)[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)

    def step(self, action):
        s = self.canvas.coords(self.agent)
        base_action = np.array([0, 0])
        if action == 0:   # Turn LEFT
            if s[1] > UNIT:
                base_action[1] -= 0.25*UNIT
                base_action[0] += 0.25*UNIT
                base_action[1] -= 0.5 *UNIT
                base_action[0] += 0.5 *UNIT
                base_action[1] -= 0.25*UNIT
                base_action[0] += 0.25*UNIT
        elif action == 1:   # Turn RIGHT
            if s[1] <  UNIT :

                base_action[1] += 0.25 * UNIT
                base_action[0] += 0.25 * UNIT
                base_action[1] += 0.5 * UNIT
                base_action[0] += 0.5 * UNIT
                base_action[1] += 0.25 * UNIT
                base_action[0] += 0.25 * UNIT
        elif action == 2:   # KEEP
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += 0
                reward=-1
        elif action == 3:  # SP_UP
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 4:   # SP_DOWN
            if s[0] > UNIT:
                base_action[0] -= 1*UNIT

        self.canvas.move(self.agent, base_action[0], base_action[1])  # move agent

        next_coords = self.canvas.coords(self.agent)  # next state

        # reward function
        if next_coords == self.canvas.coords(self.oval):
            reward = 100
            done = True
        elif next_coords in [self.canvas.coords(self.hell1)]:
            reward = -50
            done = True
        elif next_coords in [self.canvas.coords(self.hell2)]:
            reward = -50
            done = True
        elif next_coords in [self.canvas.coords(self.hell3)]:
            reward = -1
            done = True
        else:
            reward = 0
            done = False
        s_ = (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)
        return s_, reward, done
    def render(self):
        time.sleep(0.01)
        self.update()
if __name__ == "__main__":
    q = env()
    q.reset()
    q.mainloop()
