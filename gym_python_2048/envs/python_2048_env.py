import gym
from gym import spaces
from tkinter import Frame, Label, CENTER
from random import randint

GRID_LEN = 4

SIZE = 500
WIN = 1
LOST = -1
NOT_OVER = 0
GRID_PADDING = 10

KEY_UP = 0
KEY_DOWN = 1
KEY_LEFT = 2
KEY_RIGHT = 3

BACKGROUND_COLOR_GAME = "#92877d"
BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"
BACKGROUND_COLOR_DICT = {2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
                         16: "#f59563", 32: "#f67c5f", 64: "#f65e3b",
                         128: "#edcf72", 256: "#edcc61", 512: "#edc850",
                         1024: "#edc53f", 2048: "#edc22e"}
CELL_COLOR_DICT = {2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
                   32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2",
                   256: "#f9f6f2", 512: "#f9f6f2", 1024: "#f9f6f2",
                   2048: "#f9f6f2"}
FONT = ("Verdana", 40, "bold")


class Python_2048Env(gym.Env, Frame):
    metadata = {'render.modes': ['human']}
    score = 0
    rendering = False
    previous_score = 0
    moves = 0
    reward = 0

    def __init__(self):

        # self.master.bind("<Key>", self.key_down)
        self.commands = {KEY_UP: self.up, KEY_DOWN: self.down,
                         KEY_LEFT: self.left, KEY_RIGHT: self.right}

        self.reset()
        self.action_space = spaces.Discrete(4)

    def step(self, action):
        self.take_action(action)
        reward = self.get_reward()
        ob = self.get_state()
        episode_over = self.game_over
        return ob, reward, episode_over, {}

    def reset(self):
        self.game_over = False
        self.init_matrix()
        self.score = 0
        self.moves = 0
        self.previous_score = 0
        if self.rendering:
            self.update_grid_cells()

    def start_render(self):
        Frame.__init__(self)
        self.grid()
        self.init_grid()
        self.rendering = True

    def render(self, mode='human', close=False):
        self.update_grid_cells()

    def take_action(self, action):
        self.moves += 1
        if action in self.commands:
            self.matrix, done = self.commands[action](self.matrix)
            if done:
                self.matrix = self.add_value(self.matrix)
                done = False
                game_value = self.game_state(self.matrix)
                if game_value == WIN:
                    self.game_over = True
                if game_value == LOST:
                    self.game_over = True

    def get_reward(self):
        # Discourage moves that make no progress
        """if self.score == self.previous_score:
            return self.get_largest_tile() * -1
        return self.get_largest_tile() * self.score"""
        if self.score == self.previous_score:
            return -1
        else:
            self.previous_score = self.score
            return 1
            '''
        reward = self.score - self.previous_score
        self.previous_score = self.score
        return reward'''

    def get_state(self):
        return self.matrix, self.moves

    def init_grid(self):
        self.grid_cells = []
        background = Frame(self, bg=BACKGROUND_COLOR_GAME, width=SIZE,
                           height=SIZE)
        background.grid()
        for i in range(GRID_LEN):
            grid_row = []
            for j in range(GRID_LEN):
                cell = Frame(background, bg=BACKGROUND_COLOR_CELL_EMPTY,
                             width=SIZE/GRID_LEN, height=SIZE/GRID_LEN)
                cell.grid(row=i, column=j, padx=GRID_PADDING,
                          pady=GRID_PADDING)
                t = Label(master=cell, text="", bg=BACKGROUND_COLOR_CELL_EMPTY,
                          justify=CENTER, font=FONT, width=4, height=2)
                t.grid()
                grid_row.append(t)

            self.grid_cells.append(grid_row)

    def init_matrix(self):
        self.matrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

        self.matrix = self.add_value(self.matrix)
        self.matrix = self.add_value(self.matrix)

    def update_grid_cells(self):
        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                if self.matrix[i][j] == 0:
                    self.grid_cells[i][j].configure(
                        text="", bg=BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(text=str(
                        self.matrix[i][j]),
                        bg=BACKGROUND_COLOR_DICT[self.matrix[i][j]],
                        fg=CELL_COLOR_DICT[self.matrix[i][j]])
        self.update_idletasks()
        self.master.title('2048 Score: ' + str(self.score))
        self.update()

    def key_down(self, event):
        key = repr(event.char)
        self.move(key)

    def add_value(self, mat):
        row = randint(0, len(mat)-1)
        col = randint(0, len(mat)-1)

        while(mat[row][col] != 0):
            row = randint(0, len(mat)-1)
            col = randint(0, len(mat)-1)

        if randint(0, 9) == 0:
            mat[row][col] = 4
        else:
            mat[row][col] = 2

        return mat

    def game_state(self, mat):
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                if mat[i][j] == 2048:
                    return WIN

        for i in range(len(mat)-1):
            for j in range(len(mat[0])-1):
                if mat[i][j] == mat[i+1][j] or mat[i][j+1] == mat[i][j]:
                    return NOT_OVER

        for i in range(len(mat)):
            for j in range(len(mat[0])):
                if mat[i][j] == 0:
                    return NOT_OVER

        for i in range(len(mat)-1):  # check entries on the last row
            if mat[len(mat)-1][i] == mat[len(mat)-1][i+1]:  # left/right
                return NOT_OVER
            if mat[i][len(mat)-1] == mat[i+1][len(mat)-1]:  # up/down
                return NOT_OVER
        return LOST

    def reverse(self, mat):
        new = [[], [], [], []]
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                new[i].append(mat[i][len(mat[0])-j-1])
        return new

    def transpose(self, mat):
        new = [[], [], [], []]
        for i in range(len(mat[0])):
            for j in range(len(mat)):
                new[i].append(mat[j][i])
        return new

    def cover_up(self, mat):
        new = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        done = False
        for i in range(len(mat)):
            count = 0
            for j in range(len(mat)):
                if mat[i][j] != 0:
                    new[i][count] = mat[i][j]
                    if j != count:
                        done = True
                    count += 1
        return (new, done)

    def merge(self, mat):
        done = False
        for i in range(len(mat)):
            for j in range(len(mat) - 1):
                if mat[i][j] == mat[i][j+1] and mat[i][j] != 0:
                    mat[i][j] *= 2
                    self.score += mat[i][j]
                    mat[i][j+1] = 0
                    done = True
        return (mat, done)

    def up(self, game):
        game, done = self.cover_up(self.transpose(game))
        game, temp = self.merge(game)
        return (self.transpose(self.cover_up(game)[0]), done or temp)

    def down(self, game):
        game, done = self.cover_up(self.reverse(self.transpose(game)))
        game, temp = self.merge(game)
        game = self.transpose(self.reverse(self.cover_up(game)[0]))
        return (game, done or temp)

    def left(self, game):
        game, done = self.cover_up(game)
        game, temp = self.merge(game)
        return (self.cover_up(game)[0], done or temp)

    def right(self, game):
        game, done = self.cover_up(self.reverse(game))
        game, temp = self.merge(game)
        return (self.reverse(self.cover_up(game)[0]), done or temp)

    def get_largest_tile(self):
        largest = 0
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[i])):
                if self.matrix[i][j] > largest:
                    largest = self.matrix[i][j]
        return largest
