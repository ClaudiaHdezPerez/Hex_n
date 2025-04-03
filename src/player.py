from hex_board import HexBoard

class Player:
    def __init__(self, player_id: int):
        self.player_id = player_id  # Tu identificador (1 o 2)

    def play(self, board: HexBoard) -> tuple:
        raise NotImplementedError("¡Implementa este método!")
    
# Herencia de Player
class IAPlayer(Player):
    def __init__(self, player_id: int):
        self.player_id = player_id

    def play(self, board: HexBoard) -> tuple:
        raise NotImplementedError("Implement this method!")
    
    def minimax(self, board: HexBoard, depth: int, maximizing_player: bool) -> int:
        other = 1 if self.player_id == 2 else 2
        
        if depth == 0:
            return self.evaluate(board)
        
        if board.check_connection(self.player_id):
            return 1
        
        if board.check_connection(other):
            return -1
        
        if maximizing_player:
            max_eval = -float('inf')
            for i, j in board.get_possible_moves():
                board[i][j] = self.player_id
                eval = self.minimax(child, depth-1, False)
                max_eval = max(max_eval, eval)
                board[i][j] = 0
            return max_eval
        else:
            min_eval = float('inf')
            for child in board.get_possible_moves():
                board[i][j] = other
                eval = self.minimax(child, depth-1, True)
                min_eval = min(min_eval, eval)
                board[i][j] = 0
            return min_eval
        
    def evaluate(self, state):
        pass