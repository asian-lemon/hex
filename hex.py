import sys

class HexGraph:
    def __init__(self, size=11):
        """Initializes the Hex board as a graph with vertices and edges."""
        self.size = size
        self.graph = {}
        self.initialize_graph()

    def initialize_graph(self):
        """Initializes the vertices and edges for the Hex board."""
        for row in range(self.size):
            for col in range(self.size):
                node = (row, col)
                self.graph[node] = []
                # Define the six possible neighbors for a hex cell
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
                for dr, dc in directions:
                    neighbor = (row + dr, col + dc)
                    if 0 <= neighbor[0] < self.size and 0 <= neighbor[1] < self.size:
                        self.graph[node].append(neighbor)

    def get_possible_moves(self, board_state):
        """Returns a list of available cells for making a move."""
        return [pos for pos, val in board_state.items() if val is None]

    def make_move(self, move, player, board_state):
        """Places a player's move on the board."""
        board_state[move] = player

    def undo_move(self, move, board_state):
        """Removes a move from the board."""
        board_state[move] = None

    def display(self, board_state):
        """Displays the current state of the Hex board in the terminal with boundary markers."""
        # Print the top boundary (row of 'X')
        print('  ' * 1, end='')  # No indentation for the top row
        print('X ' * self.size)

        for row in range(self.size):
            # Print the left boundary ('O') for each row
            print(' ' * row, end='')  # Indentation for hex structure
            print('O ', end='')  # Print the left-side boundary

            # Print the board state
            for col in range(self.size):
                print(board_state.get((row, col), '.'), end=' ')

            # Print the right boundary ('O') at the end of each row
            print('O')
        print(' ' * self.size, end=' ')  # No indentation for the top row
        print('X ' * self.size)

    def place_move(self, row, col, player, board_state):
        """Places a move on the board and updates the state."""
        if 0 <= row < self.size and 0 <= col < self.size:
            if (row, col) not in board_state:
                board_state[(row, col)] = player
                return True
            else:
                print("Cell is already occupied.")
                return False
        else:
            print("Invalid move.")
            return False
    def bfs(self, start_nodes, player, board_state):
        """Breadth-First Search to find a connected path for a player."""
        visited = set()
        queue = start_nodes[:]

        while queue:
            node = queue.pop(0)
            if node in visited:
                continue

            visited.add(node)
            # Check if this node reaches the target side (bottom or right edge)
            if player == 'X' and node[0] == self.size - 1:  # Reaching bottom edge
                return True
            elif player == 'O' and node[1] == self.size - 1:  # Reaching right edge
                return True

            for neighbor in self.graph[node]:
                if neighbor not in visited and board_state.get(neighbor) == player:
                    queue.append(neighbor)

        return False

    def dfs(self, node, player, board_state, visited=None):
        """Depth-First Search to find a connected path for a player."""
        if visited is None:
            visited = set()

        if node in visited:
            return False

        visited.add(node)

        # Check if this node reaches the target side (bottom or right edge)
        if player == 'X' and node[0] == self.size - 1:  # Reaching bottom edge
            return True
        elif player == 'O' and node[1] == self.size - 1:  # Reaching right edge
            return True

        for neighbor in self.graph[node]:
            if neighbor not in visited and board_state.get(neighbor) == player:
                if self.dfs(neighbor, player, board_state, visited):
                    return True

        return False

    def check_win(self, player, board_state):
        """Checks if the given player has won by connecting the required sides."""
        start_nodes = []

        # Collect starting nodes based on the player's side
        if player == 'X':  # Top side
            start_nodes = [(0, col) for col in range(self.size) if board_state.get((0, col)) == player]
        elif player == 'O':  # Left side
            start_nodes = [(row, 0) for row in range(self.size) if board_state.get((row, 0)) == player]

        # Run BFS or DFS from each starting node
        for start_node in start_nodes:
            if self.dfs(start_node, player, board_state):  # You can also use `bfs(start_nodes, player, board_state)`
                return True

        return False
def minimax(board, board_state, depth, alpha, beta, maximizing_player, current_player):
    """Minimax algorithm with alpha-beta pruning."""
    opponent = 'X' if current_player == 'O' else 'O'

    # Terminal condition: Check for a win or a full board
    if board.check_win(current_player, board_state):
        return (1 if maximizing_player else -1), board_state
    elif all(value is not None for value in board_state.values()):
        return 0, board_state  # Draw

    if depth == 0:
        return 0, board_state  # Depth limit reached for simplicity

    best_move = None
    if maximizing_player:
        max_eval = -sys.maxsize
        for move in board.get_possible_moves(board_state):
            board.make_move(move, current_player, board_state)
            eval, _ = minimax(board, board_state, depth - 1, alpha, beta, False, opponent)
            board.undo_move(move, board_state)

            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cut-off
        if best_move:
            board.make_move(best_move, current_player, board_state)
        return max_eval, board_state
    else:
        min_eval = sys.maxsize
        for move in board.get_possible_moves(board_state):
            board.make_move(move, current_player, board_state)
            eval, _ = minimax(board, board_state, depth - 1, alpha, beta, True, opponent)
            board.undo_move(move, board_state)

            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cut-off
        if best_move:
            board.make_move(best_move, current_player, board_state)
        return min_eval, board_state
# Example usage:
size = 5  # You can change this to test different board sizes
hex_graph = HexGraph(size)
board_state = {}  # Dictionary to track the board state

# Place a few moves and display the board
#hex_graph.place_move(0, 0, 'X', board_state)
#hex_graph.place_move(1, 0, 'X', board_state)
#hex_graph.place_move(2, 0, 'X', board_state)
#hex_graph.place_move(3, 0, 'X', board_state)
#hex_graph.place_move(4, 0, 'X', board_state)

hex_graph.display(board_state)

# Check if player 'X' has won
if hex_graph.check_win('X', board_state):
    print("Player 'X' wins!")
else:
    print("Player 'X' has not won yet.")

best_score, best_board_state = minimax(hex_graph, board_state, depth=3, alpha=-sys.maxsize, beta=sys.maxsize, maximizing_player=True, current_player='X')

# Display the board after the best move is made
hex_graph.display(best_board_state)
print("Best evaluation score for 'X':", best_score)