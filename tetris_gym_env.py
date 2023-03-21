from turtle import pos
import pygame
from pygame import locals
import numpy as np
import random as rnd
import time 
from DQ_Agent import DQNAgent
import torch
import matplotlib.pyplot as plt
import math

import gym
from gym import spaces

MAX_EPISODE_LENGTH = 100

# EPSILON DECAY starts at 0.5 and at around 50,000 timesteps it goes to around 0.01

class Tetris_env(gym.Env):
    metadata = {'render_modes' : ['human']}

    
    # SHAPE FORMATS
    
    S = [['.....',
        '......',
        '..00..',
        '.00...',
        '.....'],
        ['.....',
        '..0..',
        '..00.',
        '...0.',
        '.....']]
    
    Z = [['.....',
        '.....',
        '.00..',
        '..00.',
        '.....'],
        ['.....',
        '..0..',
        '.00..',
        '.0...',
        '.....']]
    
    I = [['..0..',
        '..0..',
        '..0..',
        '..0..',
        '.....'],
        ['.....',
        '0000.',
        '.....',
        '.....',
        '.....']]
    
    O = [['.....',
        '.....',
        '.00..',
        '.00..',
        '.....']]
    
    J = [['.....',
        '.0...',
        '.000.',
        '.....',
        '.....'],
        ['.....',
        '..00.',
        '..0..',
        '..0..',
        '.....'],
        ['.....',
        '.....',
        '.000.',
        '...0.',
        '.....'],
        ['.....',
        '..0..',
        '..0..',
        '.00..',
        '.....']]
    
    L = [['.....',
        '...0.',
        '.000.',
        '.....',
        '.....'],
        ['.....',
        '..0..',
        '..0..',
        '..00.',
        '.....'],
        ['.....',
        '.....',
        '.000.',
        '.0...',
        '.....'],
        ['.....',
        '.00..',
        '..0..',
        '..0..',
        '.....']]
    
    T = [['.....',
        '..0..',
        '.000.',
        '.....',
        '.....'],
        ['.....',
        '..0..',
        '..00.',
        '..0..',
        '.....'],
        ['.....',
        '.....',
        '.000.',
        '..0..',
        '.....'],
        ['.....',
        '..0..',
        '.00..',
        '..0..',
        '.....']]

    
    
    def __init__(self, play_height=600, play_width=300, shapes=[S, Z, I, O, J, L, T],):
        super().__init__()

        self.s_width = 800
        self.s_height = 700
        # Play dimentions
        self.play_width = play_width  # meaning 300 // 10 = 30 width per block
        self.play_height = play_height  # meaning 600 // 20 = 20 height per block
        self.top_left_x = 250
        self.top_left_y = 100
        self.block_size = 30

        self.score = 0
        self.level = 1
        self.lines_cleared = 0
        self.shapes = shapes
        self.shape_colours = [(0, 255, 0), (255, 0, 0), 
                              (0, 255, 255), (255, 255, 0), 
                              (255, 165, 0), (0, 0, 255), 
                              (128, 0, 128)]
        self.locked_positions = {}
        self.current_piece = self.get_shape()
        self.next_piece = self.get_shape()
        self.screen = None
        self.grid = self.create_grid()
        self.current_piece_grid = np.zeros((1, 20, 10))

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=1, high=256, shape=(10,20)),
            "score": spaces.Discrete(1),
            "next_piece": spaces.Discrete(1),
            "total_lines_cleared": spaces.Discrete(1)
        })
        self.episode_length = 0
        self.cur_game_length = 0
    
    def step(self, action: int):
        assert self.action_space.contains(action), "invalid action"

        terminated = False
        done = False
        reference_score = self.score
        self.grid = self.create_grid()

        # Make the action
        change_piece = self.make_action(action)
        #print(self.current_piece.y)
        #print(self.current_piece.x)

        shape_pos = self.convert_shape_format(self.current_piece)

        self.current_piece_grid = np.zeros((1, 20, 10))

        for i in range(len(shape_pos)):
            x, y = shape_pos[i]
            if y > -1:
                self.grid[y][x] = self.current_piece.color

                self.current_piece_grid[0][y][x] = 1


        # our change_piece variable will determine if our piece has hit the bottom or not. 
        if change_piece:
            for pos in shape_pos:
                p = (pos[0], pos[1])
                self.locked_positions[p] = self.current_piece.color
                #print(self.current_piece.color)
            self.next_piece = self.get_shape()
            self.current_piece = self.next_piece
            change_piece = False
        
        #print(self.grid)
        # we call rendering, then we calculate all the things we need for rewards and whatnaught. 
        self.render()
        # Create the new observation state
        observation = {"grid": np.array(self.grid),
                       "score": self.score,
                       "next_piece": self.next_piece,
                       "total_lines_cleared": self.lines_cleared,
                       "current_piece_grid": np.array(self.current_piece_grid)}
        

        # Calculate reward. 


        current_state = np.array(self.grid)
        current_state = (current_state != 0)
        current_state = np.any(current_state, axis=2)

        #reward = 1 + (self.score - reference_score) - peak

        # Check if terminated. 
        terminated = self.check_lost()

        try:
            # We want to count how many 'holes' there are in each row, starting from the bottom
            count = 0
            rows_in_play = 1
            for row in reversed(current_state):
                if 1 in row:
                    count += np.count_nonzero(row==0)
                    rows_in_play += 1
                else: # If there is no 1's in the whole row, we can see that there is no holes above it. So we can break the loop
                    break
            #peak = min([np.argmax(current_state[:, col]) for col in range(current_state.shape[1]) if 1 in current_state[:, col]])

            #print('COUNT', count)
            #print("ROWS IN PLAY" , rows_in_play)

            # Reward = + 1 for empty board state / staying alive 
            # + 10, 20, 30, 80 for 1, 2, 3, 4 line clears in 1 move
            # - how many holes there are, averaged out by how many rows there are in play
            # - height of the board

            # print(count)
            # reward =  1 + (self.score/10 - reference_score/10) - count/rows_in_play - (rows_in_play-11)/20
            reward_bonus = 0
            if (terminated == True):
                reward = -1
            else:
                
                if (self.score - reference_score) > 0:
                    print('!!!!line clear!!!!')
                    reward_bonus = 1
                    
                if rows_in_play < 10:
                    reward = 1 - (rows_in_play/10)**0.3 + reward_bonus
                else: 
                    reward = 0
                # reward = -(1/2)*rows_in_play + 5

            #print(reward)
        except ValueError:
            reward = 1
        
        return observation, reward, terminated, {}

    def render(self):
        """
        The render function takes care of all things rendering, from making the screen, drawing the grid, etc.

        We will make it so that this agent can only be run when the game is rendered. To do that, we can put the render function in the 
        step function. 
        """
        # Checks if the actual screen has been made.
        if self.screen == None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.s_width, self.s_height))

        # Changes the grid based on the locked positions we have. 
        # Checks if there are any rows which need to be cleared based on the locked positions. 
        self.clear_rows(self.grid, self.locked_positions)

        # We can then focus on drawing the game.

        self.draw_window()
        self.draw_next_shape(self.next_piece, self.screen)
        self.draw_score(self.score)
        pygame.display.update()
        return

    def reset(self):
        self.locked_positions = {}
        self.current_piece = self.get_shape()
        self.next_piece = self.get_shape()
        self.screen = None
        self.score = 0
        self.grid = self.create_grid()
        return

    ############# GAME PROCESSING HELPER FUNCTIONS ################

    def valid_space(self, shape, grid):
        accepted_pos = [ [(j, i) for j in range(10) if grid[i][j] == (0,0,0)] for i in range(20) ]
        accepted_pos = [j for sub in accepted_pos for j in sub]

        formatted = self.convert_shape_format(shape)

        for position in formatted:
            if position not in accepted_pos:
                if position[1] > -1:
                    return False

        return True
    
    def get_shape(self):
        return Piece(5, 0, rnd.choice(self.shapes))
    
    def convert_shape_format(self, shape):
        positions = []
        format = shape.shape[shape.rotation % len(shape.shape)]

        for i, row in enumerate(format):
            row = list(row)
            for j, column in enumerate(row):
                if column == '0':
                    positions.append((shape.x + j, shape.y + i))

        for i, pos in enumerate(positions):
            positions[i] = (pos[0] - 2, pos[1] - 4)

        return positions
    
    def create_grid(self):
        # makes 10 black blocks for every row, and loops through for every column
        grid = [[(0,0,0) for x in range(10)] for x in range(20)]

        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if (j, i) in self.locked_positions:
                    c = self.locked_positions[(j,i)]
                    grid[i][j] = c 

        return grid

    def draw_grid(self, surface, grid):
        sx = self.top_left_x
        sy = self.top_left_y

        for i in range(len(grid)):
            pygame.draw.line(surface, (128,128,128), (sx, sy+i*self.block_size), (sx+self.play_width, sy+i*self.block_size))
            for j in range(len(grid[i])):
                pygame.draw.line(surface, (128,128,128), (sx + j*self.block_size, sy), (sx+j*self.block_size, sy+self.play_height))
    

    def make_action(self, action) -> bool:
        '''
        This is a helper function which moves the piece in a predetermined way. 
        Checks if the piece can move in that direction as well. 
        Returns a boolean value which determines if the piece needs to be changed. 
        '''

        # move left
        if action == 0: 
            self.current_piece.x -= 1
            if not(self.valid_space(self.current_piece, self.grid)):
                self.current_piece.x += 1

        # move right
        if action == 1:
            self.current_piece.x += 1
            if not(self.valid_space(self.current_piece, self.grid)):
                self.current_piece.x -= 1

        # rotate
        if action == 2:
            # Rotate shape
            self.current_piece.rotation += 1 
            # This disallows a rotation if the rotation will result in the shape going out of the valid space.
            if not(self.valid_space(self.current_piece, self.grid)):
                self.current_piece.rotation -=1 

        # move down
        if action == 3:
            self.current_piece.y += 1
            if not(self.valid_space(self.current_piece, self.grid)):
                self.current_piece.y -= 1

        self.current_piece.y += 1
        if not(self.valid_space(self.current_piece, self.grid)) and self.current_piece.y > 0:
            self.current_piece.y -= 1
            return True
        else:
            return False

    def clear_rows(self, grid, locked):
        # Both checks if a row is full, and is responsibile of deleting the row if it is. 
        inc = 0
        for i in range(len(grid)-1, -1, -1):
            row = grid[i]
            if (0,0,0) not in row:
                inc += 1
                ind = i
                for j in range(len(row)):
                    try:
                        del locked[(j,i)]
                    except:
                        continue

        self.increase_score(inc) 
        #print(self.lines_cleared)

        self.lines_cleared += inc
        if inc > 0:
            for key in sorted(list(locked), key = lambda x: x[1])[::-1]:
                x, y = key
                if y < ind:
                    newKey = (x, y + inc)
                    locked[newKey] = locked.pop(key)
    
    def increase_score(self, rows_cleared) -> None:
        if rows_cleared == 4:
            self.score = int(self.score+800)
        elif rows_cleared != 0: 
            self.score = int(self.score+(100 + 200*(rows_cleared-1))*self.level)

    def get_shape(self):
        return Piece(5, 0, rnd.choice(self.shapes), self.shapes, self.shape_colours)
    
    ########## PYGAME DRAWING FUNCTIONS #################

    def draw_next_shape(self, shape, surface):
        font = pygame.font.SysFont('Britannic', 30)
        label = font.render('Next Shape: ', 1, (255, 255, 255))

        sx = self.top_left_x + self.play_width + 50
        sy = self.top_left_y + self.play_height/2 - 100

        format = shape.shape[shape.rotation % len(shape.shape)]

        for i, line in enumerate(format):
            row = list(line)
            for j, column in enumerate(row):
                if column == '0': 
                    pygame.draw.rect(surface, shape.color, (sx + j*30, sy + i*self.block_size, self.block_size, self.block_size), 0)

        surface.blit(label, (sx+10, sy-30))


    def draw_score(self, score):
        font = pygame.font.SysFont('Britannic', 30)
        sx = self.top_left_x + self.play_width + 10
        sy = self.top_left_y + self.play_height/2 - 350

        score = font.render('SCORE: ' + str(score), True, (0, 0, 255))
        self.screen.blit(score, (sx, sy))

    def draw_level(self, level):
        font = pygame.font.SysFont('Britannic', 30)
        sx = self.top_left_x + self.play_width - 400
        sy = self.top_left_y + self.play_height/2 - 350

        level = font.render('LEVEL: ' + str(level), True, (0, 0, 255))
        self.screen.blit(level, (sx, sy))


    def draw_window(self):
        #print(self.grid)
        self.screen.fill((0,0,0))

        pygame.font.init()
        font = pygame.font.SysFont('Britannic', 60)
        label = font.render('TETRIS', 1, (255,255,255))

        self.screen.blit(label, (self.top_left_x + self.play_width/2 - label.get_width()/2, 30))

        
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                pygame.draw.rect(self.screen, self.grid[i][j], (self.top_left_x + j*self.block_size, self.top_left_y + i*self.block_size, self.block_size, self.block_size), 0)

        pygame.draw.rect(self.screen, (255, 0, 0), (self.top_left_x, self.top_left_y, self.play_width, self.play_height), 4)

        self.draw_grid(self.screen, self.grid)

    def draw_grid(self, surface, grid):
        sx = self.top_left_x
        sy = self.top_left_y

        for i in range(len(grid)):
            pygame.draw.line(surface, (128,128,128), (sx, sy+i*self.block_size), (sx+self.play_width, sy+i*self.block_size))
            for j in range(len(grid[i])):
                pygame.draw.line(surface, (128,128,128), (sx + j*self.block_size, sy), (sx+j*self.block_size, sy+self.play_height))

    def check_lost(self):
        for pos in self.locked_positions:
            x, y = pos
            if y < 1: 
                return True
            
        return False

class Piece():
    # Holds all the information for the pieces
    def __init__(self, x, y, shape, shapes, shape_color) -> None:
        self.x = x
        self.y = y
        self.shape = shape
        self.shapes = shapes
        self.shape_colors = shape_color
        self.rotation = 0
        self.color = shape_color[shapes.index(shape)]

MAX_TIMESTEPS = 5000

if __name__ == "__main__":
    new_game = Tetris_env()
    new_game.reset()
    new_game.render()
    agent = DQNAgent((20, 10), 4)

    #Do we load off a prev dict? 
    agent.load()

    # Make the main game loop. 
    
    episodes = 0

    sum_rewards = []

    while episodes < MAX_EPISODE_LENGTH:
        time_step = 0
        rewards = []
        new_game.reset()
        agent.replay_memory.erase_memory()

        # We explicitly run one base case. 
        observation, reward, terminated, info = new_game.step(3)

        while time_step < MAX_TIMESTEPS:
            # Observation state consists of colours. Since we're not using conv arrays
            # reducing the obs space dimentions by 1 reduces computation. 
            # In this case, that results in not considering the colour of the pieces.

            #old_observation = observation

            current_state = observation['grid']
            #if time_step == 0:

            # gray scales
            current_state = np.array(current_state)
            current_state = (current_state != 0)
            current_state = np.any(current_state, axis=2)
            current_grid = current_state
            current_state = current_state.reshape((1, 20, 10))

            # Current Piece Grid
            # print(current_state.shape)
            old_piece_board = observation["current_piece_grid"].reshape((1, 20, 10))
            # input = np.transpose(current_state, (2, 0, 1))
            #print(old_piece_board.shape)
            #print(current_state.shape)
            input = np.concatenate((current_state, old_piece_board))


            #print(input.shape)
            #print(input)


            action = agent.get_action(input)
            observation, reward, terminated, info = new_game.step(action)

            # Check if we lost. If so, we'll just start the game again if max timesteps has not run out
            if terminated:
                agent.actions_taken = 0
                new_game.reset()
                observation, reward, terminated, info = new_game.step(3)

            # total board
            observation_grid = observation['grid']
            observation_grid = (observation_grid != 0)
            observation_grid  = np.any(observation_grid, axis=2)
            observation_grid = observation_grid.reshape((1, 20, 10))

            #print(observation_grid)

            # current piece
            current_piece_board = observation["current_piece_grid"].reshape((1, 20, 10))


            # print(current_state.shape)
            rewards.append(reward)

            # Store our memory
            input_memory = (input, action, reward, np.concatenate((observation_grid, current_piece_board)))
            agent.replay_memory.store_memory(input_memory)

            agent.learn()
            time_step += 1
        
        episodes += 1
        print(f"============================== EPISODE {episodes} ==============================")
        print("average rewards per episode: ", np.sum(rewards))
        sum_rewards.append(np.sum(rewards))
        
        if episodes % 5 == 0:
            agent.save()

    plt.plot(range(1, len(sum_rewards)+1), sum_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Rewards')

    plt.title('Average rewards per episodes')
    plt.show()

    # TODO: Check if reward normalization makes sense!
    agent.save()

