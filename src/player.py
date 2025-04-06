from shutil import move
from turtle import distance
from hex_board import HexBoard
import heapq
from math import inf

class Player:
    def __init__(self, player_id: int):
        self.player_id = player_id  # Tu identificador (1 o 2)

    def play(self, board: HexBoard) -> tuple:
        raise NotImplementedError("¡Implementa este método!")
    
# Herencia de Player
class IAPlayer(Player):
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.rounds = 0

    def play(self, board: HexBoard) -> tuple:
        depth = (board.size ** 2 // (board.size ** 2 - 2 * self.rounds)) + 3
        moves = self.minimax(board, depth, True)[1]
        return moves
    
    def minimax(self, board: HexBoard, depth: int, maximizing_player: bool) -> tuple[int, tuple[int, int]]:
        other = 1 if self.player_id == 2 else 2
        
        if depth == 0: # para nodos intermedios
            return (self.evaluate_game_state(board), (-1, -1))
        
        if board.check_connection(self.player_id):
            return (1, (-1, -1))
        
        if board.check_connection(other):
            return (-1, (-1, -1))
        
        if maximizing_player:
            max_eval = -float('inf')
            better_mov = (-1, -1)
            for i, j in board.get_possible_moves():
                board.board[i][j] = self.player_id
                (eval, _) = self.minimax(board, depth-1, False)
                if eval > max_eval:
                    max_eval = eval
                    better_mov = (i, j)
                board.board[i][j] = 0
            return (max_eval, better_mov)
        else:
            min_eval = float('inf')
            better_mov = (-1, -1)
            for i, j in board.get_possible_moves():
                board.board[i][j] = other
                (eval, new_mov) = self.minimax(board, depth-1, True)
                if eval < min_eval:
                    min_eval = eval
                    better_mov = (i, j)
                board.board[i][j] = 0
            return (min_eval, better_mov)
        
    def get_adjacent_hexes(self, row, col, board) -> list[tuple[int, int]]:
        adjacents = []
        rows = [0, 1, -1, 1, -1, 0]
        cols = [-1, -1, 0, 0, 1, 1]
        
        for i in range(len(rows)):
            new_row = row + rows[i]
            new_col = col + cols[i]
            if new_row < len(board) and new_col < len(board) and new_row >= 0 and new_col >= 0:
                adjacents.append((new_row, new_col))
                    
        return adjacents

    def heuristic(self, player, board: HexBoard):
        if player == 1:
            set_left = []
            set_right = []
            for i in range(board.size):
                if (board.board[i][0] != 2):
                    set_left.append((i, 0))
                    
                if (board.board[i][board.size - 1] != 2):
                    set_right.append((i, board.size - 1))
                    
            distance = self.optimized_shortest_path_between_sets(board.board, set_left, set_right, 1)
            return distance
        else:
            set_top = []
            set_bottom = []
            for i in range(board.size):
                if (board.board[0][i] != 1):
                    set_top.append((0, i))
                    
                if (board.board[board.size - 1][i] != 1):
                    set_bottom.append((board.size - 1, i))

            distance = self.optimized_shortest_path_between_sets(board.board, set_top, set_bottom, 2)
            return distance

    def evaluate_game_state(self, board: HexBoard):
        h1 = self.heuristic(1, board)
        h2 = self.heuristic(2, board)
        
        if not h1:
            h1 = 0
            
        if not h2:
            h2 = float("inf")

        return h1 - h2
    
    def optimized_shortest_path_between_sets(self, graph, set_A, set_B, player_id):
        min_distance = inf
        best_path = inf
        
        for start_node in set_A:
            if min_distance == 0:
                break
                
            distances = {node: inf for node in range(len(graph) ** 2)}
            distances[start_node[0] * len(graph) + start_node[1]] = 0
            heap = [(0, start_node)]
            visited = set()
            
            while heap:
                current_dist, current_node = heapq.heappop(heap)
                
                if current_node in visited:
                    continue
                visited.add(current_node)
                
                if current_node in set_B:
                    if current_dist < min_distance:
                        min_distance = current_dist
                        best_path = current_dist
                    break
                
                if current_dist > min_distance:
                    break
                
                for neighbor in self.get_adjacent_hexes(current_node[0], current_node[1], graph):
                    if neighbor in visited:
                        continue
                    weight = 0 if graph[neighbor[0]][neighbor[1]] == player_id else 1
                    weight = float("inf") if graph[neighbor[0]][neighbor[1]] == 3 - player_id else 1
                    weight = 2 if graph[neighbor[0]][neighbor[1]] == graph[current_node[0]][current_node[1]] == 0 else weight
                    distance = current_dist + weight
                    if distance < distances[neighbor[0] * len(graph) + neighbor[1]]:
                        distances[neighbor[0] * len(graph) + neighbor[1]] = distance
                        heapq.heappush(heap, (distance, neighbor))
        
        return best_path if min_distance != inf else None