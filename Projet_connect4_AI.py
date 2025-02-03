import tkinter as tk
from tkinter import ttk
import numpy as np
from threading import Thread
from queue import Queue


disk_color = ['white', 'red', 'orange']
disks = list()

player_type = ['human']
for i in range(42):
    player_type.append('AI: alpha-beta level '+str(i+1))


def alpha_beta_decision(board, turn, ai_level, queue, max_player):
    """
    Chooses the best move for the maximizing player based on the depth
    of AI's calculation (ai_level) and places the move in the queue.
    :param board: The game board in its current state
    :param turn: Current turn number
    :param ai_level: Maximum depth for the AI search
    :param queue: Queue to hold the AI's next move
    :param max_player: The player trying to maximize their score
    :return: None
    """
    _, best_move = alpha_beta(board, turn, ai_level, float('-inf'),
                              float('inf'), max_player, True)
    queue.put(best_move)


def alpha_beta(board, turn, depth, alpha, beta, current_player,
               maximizing_player):
    """
    Implements the minimax algorithm with alpha-beta pruning.
    Evaluates possible moves to find the optimal move for the current player.
    :param board: The game board
    :param turn: Current turn number
    :param depth: Depth of the search
    :param alpha: Alpha value for pruning
    :param beta: Beta value for pruning
    :param current_player: The player whose move is being calculated
    :param maximizing_player: Boolean indicating if it's the maximizing player
    :return: A tuple of the best score and the corresponding move
    """
    if len(board.get_possible_moves()) == 1:  # Full board -> we evaluate
        # the board and return the only possible move
        return board.eval(current_player), board.get_possible_moves()[0]

    if depth == 0 or board.check_victory():  # Victory or max
        # depth reached -> we evaluate the board
        return board.eval(current_player), None

    possible_moves = board.get_possible_moves()
    best_move = None

    if maximizing_player:
        max_value = float('-inf')
        for move in possible_moves:
            child_board = board.copy()
            child_board.add_disk(move, current_player, update_display=False)
            value, _ = alpha_beta(child_board, turn, depth - 1, alpha, beta,
                                  current_player, False)
            if value > max_value:
                max_value = value
                best_move = move
            alpha = max(alpha, value)
            if beta <= alpha:
                break  # Beta cutoff
        return max_value, best_move
    else:
        min_value = float('inf')
        for move in possible_moves:
            child_board = board.copy()
            opponent_player = 3 - current_player
            child_board.add_disk(move, opponent_player, update_display=False)
            value, _ = alpha_beta(child_board, turn, depth - 1, alpha, beta,
                                  current_player, True)
            if value < min_value:
                min_value = value
                best_move = move
            beta = min(beta, value)
            if beta <= alpha:
                break  # Alpha cutoff
        return min_value, best_move


class Board:
    """
    Class representing the Connect 4 board. Handles game logic, board state,
    and evaluation functions.
    """
    grid = np.zeros((7, 6), dtype=int)

    def eval_window(self, window, player):
        """
        Gives a score for a set of 4 boxs (based on the number of pawns
        of the current player, of the opponent and the number of free boxs)
        :param window: A set of 4 boxs
        :param player: The player for whom the board is being evaluated
        :return: The score of the set of boxs
        """
        opponent = 3 - player
        if window.count(player) == 4:
            return 10000  # Victory -> must be high so that it cannot be
            # overtaken by another no-win combination
        if window.count(opponent) == 4:
            return -10000  # Defeat -> must be high so that it cannot be
            # overtaken by another no-win combination
        if window.count(player) == 3 and window.count(0) == 1:
            return 20  # One pawn until victory
        if window.count(opponent) == 3 and window.count(0) == 1:
            return -20  # One pawn until opponent's victory
        return 0

    def eval(self, player):
        """
        Evaluates the score of the current board state for the specified
        player.
        :param player: The player for whom the board is being evaluated
        :return: An integer score representing the board state
        """
        # Evaluation table for the central positioning of pawns
        evaluation_table = np.array([
            [3, 4, 5, 7, 5, 4, 3],
            [4, 6, 8, 10, 8, 6, 4],
            [5, 8, 11, 13, 11, 8, 5],
            [5, 8, 11, 13, 11, 8, 5],
            [4, 6, 8, 10, 8, 6, 4],
            [3, 4, 5, 7, 5, 4, 3]
        ])

        score = 0
        opponent = 3 - player

        # -> Main evaluation criteria : the AI is encouraged to line up 4
        # pawns or 3 pawns and an empty square (and prevent the opponent from
        # doing the same)

        # Horizontal alignment
        for row in range(6):
            for col in range(4):
                window = list(self.grid[col:col + 4, row])
                score += self.eval_window(window, player)

        # Vertical alignment
        for col in range(7):
            for row in range(3):
                window = list(self.grid[col, row:row + 4])
                score += self.eval_window(window, player)

        # First diagonal alignment
        for col in range(4):
            for row in range(3):
                window = [self.grid[col + i][row + i] for i in range(4)]
                score += self.eval_window(window, player)

        # Second diagonal alignment
        for col in range(4):
            for row in range(3, 6):
                window = [self.grid[col + i][row - i] for i in range(4)]
                score += self.eval_window(window, player)

        # -> Then, to help decide between 2 moves who give the same number of
        # alignments, we encourage central positioning (more chance to align
        # pawns)

        # Central positioning
        for row in range(6):
            for col in range(7):
                if self.grid[col][row] == player:
                    score += evaluation_table[row][col]
                elif self.grid[col][row] == opponent:
                    score -= evaluation_table[row][col]

        return score

    def copy(self):
        """
        Creates a copy of the board.
        :return: A new Board instance with the same state
        """
        new_board = Board()
        new_board.grid = self.grid.copy()
        return new_board

    def reinit(self):
        """
        Resets the board to its initial state and updates the display.
        """
        self.grid.fill(0)
        for i in range(7):
            for j in range(6):
                canvas1.itemconfig(disks[i][j], fill=disk_color[0])

    def get_possible_moves(self):
        """
        Determines the list of valid columns where a disk can be added.
        :return: List of column indices
        """
        possible_moves = list()
        if self.grid[3][5] == 0:
            possible_moves.append(3)
        for shift_from_center in range(1, 4):
            if self.grid[3 + shift_from_center][5] == 0:
                possible_moves.append(3 + shift_from_center)
            if self.grid[3 - shift_from_center][5] == 0:
                possible_moves.append(3 - shift_from_center)
        return possible_moves

    def add_disk(self, column, player, update_display=True):
        """
        Adds a disk to the specified column for the given player.
        :param column: Column index where the disk should be placed
        :param player: The player making the move
        :param update_display: Boolean to update the graphical display
        """
        for j in range(6):
            if self.grid[column][j] == 0:
                self.grid[column][j] = player
                break
        if update_display:
            canvas1.itemconfig(disks[column][j], fill=disk_color[player])

    def column_filled(self, column):
        """
        Checks if a column is completely filled.
        :param column: Column index
        :return: True if the column is full, False otherwise
        """
        return self.grid[column][5] != 0

    def check_victory(self):
        """
        Checks if there is a victory condition on the board.
        :return: True if a player has won, False otherwise
        """
        # Horizontal alignment check
        for line in range(6):
            for horizontal_shift in range(4):
                if (self.grid[horizontal_shift][line]
                        == self.grid[horizontal_shift + 1][line]
                        == self.grid[horizontal_shift + 2][line]
                        == self.grid[horizontal_shift + 3][line]
                        != 0):
                    return True
        # Vertical alignment check
        for column in range(7):
            for vertical_shift in range(3):
                if (self.grid[column][vertical_shift]
                        == self.grid[column][vertical_shift + 1]
                        == self.grid[column][vertical_shift + 2]
                        == self.grid[column][vertical_shift + 3]
                        != 0):
                    return True
        # Diagonal alignment check
        for horizontal_shift in range(4):
            for vertical_shift in range(3):
                if (self.grid[horizontal_shift][vertical_shift]
                        == self.grid[horizontal_shift + 1][vertical_shift + 1]
                        == self.grid[horizontal_shift + 2][vertical_shift + 2]
                        == self.grid[horizontal_shift + 3][vertical_shift + 3]
                        != 0):
                    return True
                elif (self.grid[horizontal_shift][5 - vertical_shift]
                      == self.grid[horizontal_shift + 1][4 - vertical_shift]
                      == self.grid[horizontal_shift + 2][3 - vertical_shift]
                      == self.grid[horizontal_shift + 3][2 - vertical_shift]
                      != 0):
                    return True
        return False


class Connect4:
    """
    Class managing the Connect 4 game, with AI.
    """
    def __init__(self):
        self.board = Board()
        self.human_turn = False
        self.turn = 0
        self.players = (0, 0)
        self.ai_move = Queue()

    def current_player(self):
        """
        Determines the current player's number (1 or 2).
        :return: The current player's number
        """
        return (self.turn - 1) % 2 + 1

    def launch(self):
        """
        Starts a new game and initializes the board.
        """
        self.board.reinit()
        self.turn = 0
        information['fg'] = 'black'
        information['text'] = "Turn " + str(self.turn) + " - Player " + str(
            self.current_player()) + " is playing"
        self.human_turn = False
        self.players = (combobox_player1.current(), combobox_player2.current())
        self.handle_turn()

    def move(self, column):
        """
        Executes a move in the specified column.
        :param column: The column index for the move
        """
        if not self.board.column_filled(column):
            self.board.add_disk(column, self.current_player())
            self.handle_turn()

    def click(self, event):
        """
        Handles a mouse click event on the game board.
        :param event: The mouse click event
        """
        if self.human_turn:
            column = event.x // row_width
            self.move(column)

    def ai_turn(self, ai_level):
        """
        Starts the AI's turn by running the alpha-beta decision function in a
        separate thread.
        :param ai_level: The AI's calculation depth
        """
        Thread(target=alpha_beta_decision,
               args=(self.board, self.turn, ai_level, self.ai_move,
                     self.current_player(),)).start()
        self.ai_wait_for_move()

    def ai_wait_for_move(self):
        """
        Waits for AI's move then places its play on board.
        """
        if not self.ai_move.empty():
            self.move(self.ai_move.get())
        else:
            window.after(100, self.ai_wait_for_move)

    def handle_turn(self):
        """
        Handles the actions for the current turn.
        """
        self.human_turn = False
        if self.board.check_victory():
            information['fg'] = 'red'
            information['text'] = ("Player " + str(self.current_player())
                                   + " wins !")
            return
        elif self.turn >= 42:
            information['fg'] = 'red'
            information['text'] = "This a draw !"
            return
        self.turn += 1
        information['text'] = ("Turn " + str(self.turn) + " - Player "
                               + str(self.current_player()) + " is playing")
        if self.players[self.current_player() - 1] != 0:
            self.human_turn = False
            self.ai_turn(self.players[self.current_player() - 1])
        else:
            self.human_turn = True


game = Connect4()

# Graphical settings
width = 700
row_width = width // 7
row_height = row_width
height = row_width * 6
row_margin = row_height // 10

window = tk.Tk()
window.title("Connect 4")
canvas1 = tk.Canvas(window, bg="blue", width=width, height=height)

# Drawing the grid
for i in range(7):
    disks.append(list())
    for j in range(5, -1, -1):
        disks[i].append(canvas1.create_oval(row_margin + i * row_width,
                                            row_margin + j * row_height,
                                            (i + 1) * row_width - row_margin,
                                            (j + 1) * row_height - row_margin,
                                            fill='white'))


canvas1.grid(row=0, column=0, columnspan=2)

information = tk.Label(window, text="")
information.grid(row=1, column=0, columnspan=2)

label_player1 = tk.Label(window, text="Player 1: ")
label_player1.grid(row=2, column=0)
combobox_player1 = ttk.Combobox(window, state='readonly')
combobox_player1.grid(row=2, column=1)

label_player2 = tk.Label(window, text="Player 2: ")
label_player2.grid(row=3, column=0)
combobox_player2 = ttk.Combobox(window, state='readonly')
combobox_player2.grid(row=3, column=1)

combobox_player1['values'] = player_type
combobox_player1.current(0)
combobox_player2['values'] = player_type
combobox_player2.current(6)

button2 = tk.Button(window, text='New game', command=game.launch)
button2.grid(row=4, column=0)

button = tk.Button(window, text='Quit', command=window.destroy)
button.grid(row=4, column=1)

# Mouse handling
canvas1.bind('<Button-1>', game.click)

window.mainloop()
