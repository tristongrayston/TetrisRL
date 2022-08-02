from pdb import runcall
from turtle import pos
import pygame
import random as rnd
 
# creating the data structure for pieces
# setting up global vars
# functions
# - create_grid
# - draw_grid
# - draw_window
# - rotating shape in main
# - setting up the main
 
"""
10 x 20 square grid
shapes: S, Z, I, O, J, L, T
represented in order by 0 - 6
"""
 
pygame.font.init()
 
# GLOBALS VARS
class Tetris(object):

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
    
    #shapes = [S, Z, I, O, J, L, T]
    #shape_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 165, 0), (0, 0, 255), (128, 0, 128)]
    # index 0 - 6 represent shape
    
    def __init__(self, s_width = 800, s_height=700, play_width =300, play_height = 600, SCORE = 0, LEVEL = 1, level_lines_cleared = 0, 
                shapes = [S, Z, I, O, J, L, T], shape_colours = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 165, 0), (0, 0, 255), (128, 0, 128)], 
                top_left_x = 250, top_left_y = 100, win = pygame.display.set_mode((800, 700))):
                
        # Application dimentions
        self.s_width = s_width
        self.s_height = s_height
        # Play dimentions
        self.play_width = play_width  # meaning 300 // 10 = 30 width per block
        self.play_height = play_height  # meaning 600 // 20 = 20 height per block
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y

        self.block_size = 30

        self.score = SCORE
        self.level = LEVEL
        self.lines_cleared = level_lines_cleared

        self.shapes = shapes
        self.shape_colors = shape_colours
        self.win = win
        #self.Piece = self.Piece(5, 0, rnd.choice(self.shapes), self.shapes, self.shape_colors)
    
    '''
    # Application dimentions
    s_width = 800
    s_height = 700
    # Play dimentions
    play_width = 300  # meaning 300 // 10 = 30 width per block
    play_height = 600  # meaning 600 // 20 = 20 height per block

    block_size = 30
    global SCORE
    global LEVEL
    global level_lines_cleared
    
    top_left_x = (s_width - play_width) // 2
    top_left_y = s_height - play_height

    level_lines_cleared = 0    
    SCORE = 0
    LEVEL = 1
    '''
    def increase_level(self, fall_speed):
        self.level += 1
        level_lines_cleared = 0
        fall_speed /= 1.5
        print(level_lines_cleared)
        return fall_speed



    def increase_score(self, rows_cleared) -> None:
        self.score
        print('score increase')
        if rows_cleared == 4:
            self.score = int(self.score+800)
        elif rows_cleared != 0: 
            self.score = int(self.score+(100 + 200*(rows_cleared-1))*self.level)

    def get_shapes(self):
        return self.shapes
    
    def create_grid(self, locked_positions={}):
        # makes 10 black blocks for every row, and loops through for every column
        grid = [[(0,0,0) for x in range(10)] for x in range(20)]

        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if (j, i) in locked_positions:
                    c = locked_positions[(j,i)]
                    grid[i][j] = c 

        return grid
    
    def convert_shape_format(self, shape):
        positions = []
        format = shape.shape[shape.rotation % len(shape.shape)]

        for i, row in enumerate(format):
            row = list(row)
            for j, column in enumerate(row):
                if column == '0':
                    positions. append((shape.x + j, shape.y + i))

        for i, pos in enumerate(positions):
            positions[i] = (pos[0] - 2, pos[1] - 4)

        return positions

    
    def valid_space(self, shape, grid):
        accepted_pos = [ [ (j, i) for j in range(10) if grid[i][j] == (0,0,0) ] for i in range(20) ]
        accepted_pos = [j for sub in accepted_pos for j in sub]

        formatted = self.convert_shape_format(shape)

        for position in formatted:
            if position not in accepted_pos:
                if position[1] > -1:
                    return False

        return True
    
    def check_lost(self, positions):
        for pos in positions:
            x, y = pos
            if y < 1: 
                return True
            
        return False
    
    def get_shape(self):
        return Piece(5, 0, rnd.choice(self.shapes), self.shapes, self.shape_colors)
    
    
    def draw_text_middle(self, text, size, color, surface):
        pass
    
    def draw_grid(self, surface, grid):
        sx = self.top_left_x
        sy = self.top_left_y

        for i in range(len(grid)):
            pygame.draw.line(surface, (128,128,128), (sx, sy+i*self.block_size), (sx+self.play_width, sy+i*self.block_size))
            for j in range(len(grid[i])):
                pygame.draw.line(surface, (128,128,128), (sx + j*self.block_size, sy), (sx+j*self.block_size, sy+self.play_height))


        
    def clear_rows(self, grid, locked, score, level):
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
        print(self.lines_cleared)


        self.lines_cleared += inc
        if inc > 0:
            for key in sorted(list(locked), key = lambda x: x[1])[::-1]:
                x, y = key
                if y < ind:
                    newKey = (x, y + inc)
                    locked[newKey] = locked.pop(key)

    
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
        self.win.blit(score, (sx, sy))

    def draw_level(self, level):
        font = pygame.font.SysFont('Britannic', 30)
        sx = self.top_left_x + self.play_width - 400
        sy = self.top_left_y + self.play_height/2 - 350

        level = font.render('LEVEL: ' + str(level), True, (0, 0, 255))
        self.win.blit(level, (sx, sy))


    def draw_window(self, surface, grid):
        surface.fill((0,0,0))

        pygame.font.init()
        font = pygame.font.SysFont('Britannic', 60)
        label = font.render('TETRIS', 1, (255,255,255))

        surface.blit(label, (self.top_left_x + self.play_width/2 - label.get_width()/2, 30))

        
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                pygame.draw.rect(surface, grid[i][j], (self.top_left_x + j*self.block_size, self.top_left_y + i*self.block_size, self.block_size, self.block_size), 0)

        pygame.draw.rect(surface, (255, 0, 0), (self.top_left_x, self.top_left_y, self.play_width, self.play_height), 4)

        self.draw_grid(surface, grid)

    
    def main(self, win):
        locked_positions = {}
        grid = self.create_grid(locked_positions)
        change_piece = False
        run = True
        current_piece = self.get_shape()
        next_piece = self.get_shape()
        clock = pygame.time.Clock()
        fall_time = 0
        fall_speed = 0.27
        stop_time = 0


        while run:
            grid = self.create_grid(locked_positions)
            fall_time += clock.get_rawtime()
            stop_time = clock.get_rawtime()
            clock.tick()
            
            if stop_time > 60:
                run = False

            if fall_time/1000 > fall_speed:
                fall_time = 0
                current_piece.y += 1
                if not(self.valid_space(current_piece, grid)) and current_piece.y > 0:
                    current_piece.y -= 1
                    change_piece = True


            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        current_piece.x -= 1
                        if not(self.valid_space(current_piece, grid)):
                            current_piece.x += 1
                    if event.key == pygame.K_RIGHT:
                        current_piece.x += 1
                        if not(self.valid_space(current_piece, grid)):
                            current_piece.x -= 1
                    if event.key == pygame.K_UP:
                        # Rotate shape
                        current_piece.rotation += 1 
                        # This disallows a rotation if the rotation will result in the shape going out of the valid space.
                        if not(self.valid_space(current_piece, grid)):
                            current_piece.rotation -=1 
                    if event.key == pygame.K_DOWN:
                        current_piece.y += 1
                        if not(self.valid_space(current_piece, grid)):
                            current_piece.y -= 1
            
            shape_pos = self.convert_shape_format(current_piece)

            for i in range(len(shape_pos)):
                x, y = shape_pos[i]
                if y > -1:
                    grid[y][x] = current_piece.color

            # For when a piece hits the ground
            if change_piece:
                for pos in shape_pos:
                    p = (pos[0], pos[1])
                    locked_positions[p] = current_piece.color
                current_piece = next_piece
                next_piece = self.get_shape()
                change_piece = False
                self.clear_rows(grid, locked_positions, self.score, self.level)

            if self.lines_cleared >= self.level:
                fall_speed = self.increase_level(fall_speed)

            self.draw_window(self.win, grid)
            self.draw_next_shape(next_piece, win)
            self.draw_score(self.score)
            self.draw_level(self.level)
            pygame.display.update()


            if self.check_lost(locked_positions):
                run = False

        print('FINAL SCORE: {}'.format(self.score))
        pygame.display.quit()



    
    def main_menu(self):
        self.main(self.win)
        pass
    

    #win = pygame.display.set_mode((800, 700))
    pygame.display.set_caption("Tetris")


class Piece(Tetris):
    # Holds all the information for the pieces
    def __init__(self, x, y, shape, shapes, shape_color) -> None:
        self.x = x
        self.y = y
        self.shape = shape
        self.shapes = shapes
        self.shape_colors = shape_color
        self.rotation = 0
        self.color = shape_color[shapes.index(shape)]

if __name__ == "__main__":
    win = pygame.display.set_mode((800, 700))
    play_tetris = Tetris()
    play_tetris.main_menu()  # start game

