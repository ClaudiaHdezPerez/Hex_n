import random
from hex_board import HexBoard
import heapq
import math

# Métodos básicos
def get_possible_moves(board: HexBoard) -> list:
    """Devuelve todas las casillas vacías como tuplas (fila, columna)."""
    possible_moves: list = []
    for i in range(board.size):
        for j in range(board.size):
            if (not board.board[i][j]):
                possible_moves.append((i, j))
    
    return possible_moves

def check_connection(board: HexBoard, player_id: int) -> bool:
    """Verifica si el jugador ha conectado sus dos lados"""
    player_positions = []
    for i in range(board.size):
        for j in range(board.size):
            if (board.board[i][j] == player_id):
                player_positions.append((i, j))
    
    return dfs(player_positions, player_id, board.size)

adj = [(0,1),(0,-1),(1,-1),(1,0),(-1,1),(-1,0)]

def dfs(g,player_id,size):
    visited = set()
    p = {}
    for u in g:
        if player_id == 1 and u[1] != 0:
            continue
        elif player_id == 2 and u[0] != 0:
            continue

        if u not in visited:
            p[(u[0],u[1])] = None
            if dfs_visit(g,u,visited,p,size,player_id):
                return True
    return False

def dfs_visit(g,u,visited,p,size,player_id):
    visited.add((u))
    for dir in adj:
        v = (u[0]+dir[0],u[1]+dir[1])
        if v not in g:
            continue
        if player_id == 1 and v[1] == size - 1:
            return True
        elif player_id == 2 and v[0] == size - 1:
            return True

        if v not in visited:
            p[v] = u
            if dfs_visit(g,v,visited,p,size,player_id):
                return True
    return False

# Clase base de jugador.
class Player:
    def __init__(self, player_id: int):
        self.player_id = player_id  # Identificador: 1 o 2

    def play(self, board: HexBoard) -> tuple:
        raise NotImplementedError


# IAPlayer utiliza Monte Carlo, Minimax con poda alfa-beta y
# la evaluación de conexión basada en A* (clase HSearch).
class IAPlayer(Player):
    def __init__(self, player_id: int):
        super().__init__(player_id)
        self.rounds = 0
        self.h_search = None

    def play(self, board: HexBoard) -> tuple:
        if not self.h_search:
            self.h_search = HSearch(board.size)
        
        # Primer movimiento: si el centro está libre, juega allí.
        center = board.size // 2
        if board.board[center][center] == 0:
            return (center, center)
        
        winner, play_winner = self.one_move_to_win(board, self.player_id)
        
        if winner != 0:
            return play_winner
        
        # Selecciona la estrategia según el progreso de la partida.
        if board.size ** 2 - 2 * self.rounds > board.size ** 2 // 4:
            depth = 3
            move = self.monte_carlo_search(board, depth)
        else:
            depth = (board.size ** 2 // (board.size ** 2 - 2 * self.rounds)) + 1
            move = self.minimax_alpha_beta(board, depth, -float('inf'), float('inf'), True)[1]
        
        self.rounds += 1
        return move
    
    def one_move_to_win(self, board: HexBoard, player_id: int) -> tuple[int, tuple[int, int]]:
        possible_moves = get_possible_moves(board)
        move = (0, (-1, -1))
        
        for i, j in possible_moves:
            board.board[i][j] = player_id
            if check_connection(board, player_id):
                return (1, (i, j))
            
            board.board[i][j] = 0
            board.board[i][j] = 3 - player_id
            if check_connection(board, 3 - player_id):
                move = (-1, (i, j))
                
            board.board[i][j] = 0
            
        return move

    def monte_carlo_search(self, board: HexBoard, depth: int) -> tuple:
        possible_moves = get_possible_moves(board)
        if not possible_moves:
            return (-1, -1)
        
        # Estadísticas para cada movimiento.
        move_stats = {move: {'wins': 0, 'simulations': 0} for move in possible_moves}
        simulations = 0
        
        while simulations < 1000:
            move = random.choice(possible_moves)
            board.board[move[0]][move[1]] = self.player_id

            # Simulación a profundidad 'depth'.
            result = self.simulate(board, depth, 3 - self.player_id)
            board.board[move[0]][move[1]] = 0
            
            move_stats[move]['simulations'] += 1
            if result >= 0:
                move_stats[move]['wins'] += result
            simulations += 1
        
        best_move = max(
            move_stats.keys(), 
            key=lambda m: (move_stats[m]['wins'] / move_stats[m]['simulations'] 
                           if move_stats[m]['simulations'] > 0 else 0)
        )
        return best_move

    def simulate(self, board: HexBoard, depth: int, current_player: int) -> int:
        # Termina la simulación si alguien conecta.
        if check_connection(board, self.player_id):
            return float("inf")
        if check_connection(board, 3 - self.player_id):
            return -float("inf")
        
        # Si llega a la profundidad máxima, usa la heurística.
        if depth == 0:
            return self.evaluate_game_state(board)
        
        possible_moves = get_possible_moves(board)
        
        move = random.choice(possible_moves)
        board.board[move[0]][move[1]] = current_player
        result = self.simulate(board, depth - 1, 3 - current_player)
        board.board[move[0]][move[1]] = 0
        return result

    def minimax_alpha_beta(self, board: HexBoard, depth: int, alpha: float, beta: float, maximizing_player: bool) -> tuple:
        other = 3 - self.player_id
        
        if check_connection(board, self.player_id):
            return (float('inf'), (-1, -1))
        if check_connection(board, other):
            return (-float('inf'), (-1, -1))
        if depth == 0:
            return (self.evaluate_game_state(board), (-1, -1))
        
        if maximizing_player:
            max_eval = -float('inf')
            best_move = (-1, -1)
            for i, j in get_possible_moves(board):
                board.board[i][j] = self.player_id
                eval, _ = self.minimax_alpha_beta(board, depth - 1, alpha, beta, False)
                board.board[i][j] = 0
                if eval > max_eval:
                    max_eval = eval
                    best_move = (i, j)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break 
            return (max_eval, best_move)
        else:
            min_eval = float('inf')
            best_move = (-1, -1)
            for i, j in board.get_possible_moves(board):
                board.board[i][j] = other
                eval, _ = self.minimax_alpha_beta(board, depth - 1, alpha, beta, True)
                board.board[i][j] = 0
                if eval < min_eval:
                    min_eval = eval
                    best_move = (i, j)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return (min_eval, best_move)

    def evaluate_game_state(self, board: HexBoard) -> int:
        """
        Se usa el método A* implementado en HSearch para obtener el costo (es decir,
        el esfuerzo requerido para conectar) para el jugador y el oponente.
        Un menor costo para el jugador y un mayor costo para el adversario indican
        una posición ventajosa.
        """
        my_cost = self.h_search.astar_search(board.board, self.player_id)
        opp_cost = self.h_search.astar_search(board.board, 3 - self.player_id)
        if my_cost == 0:
            return float('inf')
        if opp_cost == 0:
            return -float('inf')
        return 1 / my_cost + opp_cost
    
class HSearch:
    def __init__(self, board_size):
        self.board_size = board_size
        # Se definen las 6 direcciones de movimiento en el tablero Hex.
        self.neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    
    def _heuristic(self, x, y, player):
        """
        Función heurística que estima la "distancia" restante hasta la meta.
        Para el jugador 1 (conexión izquierda a derecha): diferencia entre la columna actual y la última.
        Para el jugador 2 (conexión arriba a abajo): diferencia entre la fila actual y la última.
        """
        n = self.board_size
        if player == 1:
            return abs(n // 2 - y)
        else:
            return abs(n // 2 - x)

    def astar_search(self, board, player):
        """
        Realiza una búsqueda A* para determinar el costo mínimo de conexión.
        - A cada celda se le asigna:
            • Costo 0 si ya está ocupada por el jugador.
            • Costo 1 si la celda está vacía.
            • Costo infinito (math.inf) si la celda está ocupada por el adversario.
        Se utiliza la función heurística para dirigir la búsqueda hacia el borde meta.
        """
        n = self.board_size
        # Se inicializa una matriz de costos con valor infinito
        distances = [[math.inf] * n for _ in range(n)]
        heap = []
        
        if player == 1:
            # Para el jugador 1: la conexión va de la columna 0 a la columna n-1.
            for i in range(n):
                if board[i][0] == player:
                    cost = 0
                elif board[i][0] == 0:
                    cost = 1
                else:
                    continue  # Celdas ocupadas por el adversario se ignoran.
                distances[i][0] = cost
                h = self._heuristic(i, 0, player)
                heapq.heappush(heap, (cost + h, cost, i, 0))
            
            best = math.inf
            target_col = n - 1
            while heap:
                priority, cost, x, y = heapq.heappop(heap)
                # Si alcanzamos la última columna, actualizamos el mejor costo encontrado.
                if y == target_col:
                    best = min(best, cost)
                    if best == 0:
                        return best
                if cost > distances[x][y]:
                    continue
                for dx, dy in self.neighbors:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < n and 0 <= ny < n:
                        # Se asigna costo según el contenido de la celda:
                        if board[nx][ny] == player:
                            new_cost = cost
                        elif board[nx][ny] == 0:
                            new_cost = cost + 1
                        else:
                            new_cost = math.inf  # Bloqueada por el adversario.
                        if new_cost < distances[nx][ny]:
                            distances[nx][ny] = new_cost
                            h = self._heuristic(nx, ny, player)
                            heapq.heappush(heap, (new_cost + h, new_cost, nx, ny))
            return best
        else:
            # Para el jugador 2: la conexión va de la fila 0 a la fila n-1.
            for j in range(n):
                if board[0][j] == player:
                    cost = 0
                elif board[0][j] == 0:
                    cost = 1
                else:
                    continue
                distances[0][j] = cost
                h = self._heuristic(0, j, player)
                heapq.heappush(heap, (cost + h, cost, 0, j))
            
            best = math.inf
            target_row = n - 1
            while heap:
                priority, cost, x, y = heapq.heappop(heap)
                if x == target_row:
                    best = min(best, cost)
                    if best == 0:
                        return best
                if cost > distances[x][y]:
                    continue
                for dx, dy in self.neighbors:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < n and 0 <= ny < n:
                        if board[nx][ny] == player:
                            new_cost = cost
                        elif board[nx][ny] == 0:
                            new_cost = cost + 1
                        else:
                            new_cost = math.inf
                        if new_cost < distances[nx][ny]:
                            distances[nx][ny] = new_cost
                            h = self._heuristic(nx, ny, player)
                            heapq.heappush(heap, (new_cost + h, new_cost, nx, ny))
            return best

    def evaluate_connection_strength(self, board, player):
        """
        Evalúa la fortaleza de conexión de un jugador.
        
        Se utiliza la búsqueda A* para determinar el costo mínimo de conectar los bordes.
        Un menor costo indica una conexión más fuerte y, por ende, una posición ventajosa.
        """
        return self.astar_search(board, player)