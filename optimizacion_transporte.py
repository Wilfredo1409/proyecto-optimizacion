import os
import requests
import networkx as nx
import matplotlib.pyplot as plt

class TransportationProblem:
    """
    Clase principal que almacena los datos del problema de transporte de 'Rutas y Suministros C.A.'[cite: 1].
    Incluye validaciones de datos y valores por defecto.
    """
    def __init__(self, plants=None, destinations=None, supply=None, demand=None, costs=None):
        # Datos por defecto según el enunciado de la tarea[cite: 1]
        self.plants = plants or ["Planta San Diego (01)", "Planta Maracay (02)", "Planta Barquisimeto (03)"]
        self.destinations = destinations or ["C.D. Caracas (D1)", "C.D. Maracaibo (D2)", "C.D. Pto. La Cruz (D3)", "C.D. San Cristóbal (D4)"]
        self.supply = supply or [300, 400, 500]
        self.demand = demand or [250, 350, 400, 200]
        self.costs = costs or [
            [15, 35, 40, 45],
            [10, 45, 30, 50],
            [25, 20, 55, 20]
        ]
        self.validate_data()

    def validate_data(self):
        """Valida que el problema esté balanceado y los datos sean consistentes[cite: 1, 4]."""
        total_supply = sum(self.supply)
        total_demand = sum(self.demand)
        if total_supply != total_demand:
            raise ValueError(f"El problema no está balanceado. Oferta total ({total_supply}) != Demanda total ({total_demand})[cite: 1].")
        if len(self.costs) != len(self.supply) or any(len(row) != len(self.demand) for row in self.costs):
            raise ValueError("Las dimensiones de la matriz de costos no coinciden con la oferta y la demanda.")
        print("[Validación] Datos cargados y validados exitosamente. El sistema está balanceado (Oferta = Demanda = 1200)[cite: 1, 4].\n")


class AlgorithmicEngine:
    """
    Módulo A: Motor Algorítmico desde cero utilizando estructuras de datos nativas (listas/diccionarios).
    """
    def __init__(self, problem: TransportationProblem):
        self.problem = problem

    def calculate_total_cost(self, allocation):
        """Calcula el costo total de transporte basado en la matriz de asignación y costos[cite: 2]."""
        total_cost = 0
        for i in range(len(self.problem.supply)):
            for j in range(len(self.problem.demand)):
                total_cost += allocation[i][j] * self.problem.costs[i][j]
        return total_cost

    def northwest_corner(self):
        """1. Método de la Esquina Noroeste[cite: 2]."""
        supply = list(self.problem.supply)
        demand = list(self.problem.demand)
        rows = len(supply)
        cols = len(demand)
        allocation = [[0] * cols for _ in range(rows)]
        
        i, j = 0, 0
        while i < rows and j < cols:
            qty = min(supply[i], demand[j])
            allocation[i][j] = qty
            supply[i] -= qty
            demand[j] -= qty
            
            if supply[i] == 0:
                i += 1
            if j < cols and demand[j] == 0:
                j += 1
                
        cost = self.calculate_total_cost(allocation)
        return allocation, cost

    def least_cost(self):
        """2. Método del Costo Mínimo[cite: 2]."""
        supply = list(self.problem.supply)
        demand = list(self.problem.demand)
        rows = len(supply)
        cols = len(demand)
        allocation = [[0] * cols for _ in range(rows)]
        
        # Crear lista de celdas con sus costos y coordenadas
        cells = []
        for i in range(rows):
            for j in range(cols):
                cells.append((self.problem.costs[i][j], i, j))
        cells.sort(key=lambda x: x[0]) # Ordenar por menor costo
        
        for cost_val, i, j in cells:
            if supply[i] > 0 and demand[j] > 0:
                qty = min(supply[i], demand[j])
                allocation[i][j] = qty
                supply[i] -= qty
                demand[j] -= qty
                
        total_cost = self.calculate_total_cost(allocation)
        return allocation, total_cost

    def vogel_approximation(self):
        """3. Método de Aproximación de Vogel (VAM)[cite: 2]."""
        supply = list(self.problem.supply)
        demand = list(self.problem.demand)
        rows = len(supply)
        cols = len(demand)
        allocation = [[0] * cols for _ in range(rows)]
        
        active_rows = [True] * rows
        active_cols = [True] * cols
        
        while any(active_rows) and any(active_cols):
            row_penalties = []
            for i in range(rows):
                if not active_rows[i]:
                    row_penalties.append(-1)
                    continue
                row_costs = [self.problem.costs[i][j] for j in range(cols) if active_cols[j]]
                if len(row_costs) == 0:
                    row_penalties.append(-1)
                elif len(row_costs) == 1:
                    row_penalties.append(row_costs[0])
                else:
                    row_costs.sort()
                    row_penalties.append(row_costs[1] - row_costs[0])
            
            col_penalties = []
            for j in range(cols):
                if not active_cols[j]:
                    col_penalties.append(-1)
                    continue
                col_costs = [self.problem.costs[i][j] for i in range(rows) if active_rows[i]]
                if len(col_costs) == 0:
                    col_penalties.append(-1)
                elif len(col_costs) == 1:
                    col_penalties.append(col_costs[0])
                else:
                    col_costs.sort()
                    col_penalties.append(col_costs[1] - col_costs[0])
            
            max_row_pen = max(row_penalties) if row_penalties else -1
            max_col_pen = max(col_penalties) if col_penalties else -1
            
            if max_row_pen >= max_col_pen and max_row_pen != -1:
                i = row_penalties.index(max_row_pen)
                min_cost = float('inf')
                chosen_j = -1
                for j in range(cols):
                    if active_cols[j] and self.problem.costs[i][j] < min_cost:
                        min_cost = self.problem.costs[i][j]
                        chosen_j = j
                j = chosen_j
            elif max_col_pen != -1:
                j = col_penalties.index(max_col_pen)
                min_cost = float('inf')
                chosen_i = -1
                for i in range(rows):
                    if active_rows[i] and self.problem.costs[i][j] < min_cost:
                        min_cost = self.problem.costs[i][j]
                        chosen_i = i
                i = chosen_i
            else:
                break
                
            qty = min(supply[i], demand[j])
            allocation[i][j] = qty
            supply[i] -= qty
            demand[j] -= qty
            
            if supply[i] == 0:
                active_rows[i] = False
            if demand[j] == 0:
                active_cols[j] = False
                
        total_cost = self.calculate_total_cost(allocation)
        return allocation, total_cost


class NetworkVisualizer:
    """
    Módulo B: Representación en Grafo (Modelo de Red) utilizando NetworkX + Matplotlib[cite: 2].
    """
    def __init__(self, problem: TransportationProblem):
        self.problem = problem

    def generate_graph(self, allocation, method_name, filename):
        """Genera y guarda una imagen del grafo para el método evaluado[cite: 2, 3]."""
        G = nx.DiGraph()
        
        # Añadir nodos y posiciones (Plantas a la izquierda, Destinos a la derecha)
        pos = {}
        for i, plant in enumerate(self.problem.plants):
            G.add_node(plant, bipartite=0)
            pos[plant] = (0, -i * 2)
            
        for j, dest in enumerate(self.problem.destinations):
            G.add_node(dest, bipartite=1)
            pos[dest] = (3, -j * 1.5)
            
        # Añadir aristas con peso > 0[cite: 2]
        edge_labels = {}
        for i in range(len(self.problem.plants)):
            for j in range(len(self.problem.destinations)):
                qty = allocation[i][j]
                if qty > 0:
                    u = self.problem.plants[i]
                    v = self.problem.destinations[j]
                    G.add_edge(u, v, weight=qty)
                    edge_labels[(u, v)] = f"{qty} u"
                    
        plt.figure(figsize=(10, 6))
        nx.draw_networkx_nodes(G, pos, nodelist=self.problem.plants, node_color='lightblue', node_shape='s', node_size=2000)
        nx.draw_networkx_nodes(G, pos, nodelist=self.problem.destinations, node_color='lightgreen', node_shape='o', node_size=2000)
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowsize=20, width=2, edge_color='gray')
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title(f"Modelo de Red - {method_name}\nCosto Total de Distribución", fontsize=12)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"[Grafo Generado] Imagen guardada exitosamente como: {filename}.")


class AIAnalystAgent:
    """
    Módulo C: Integración con API de IA (Agente Analista) vía peticiones HTTP REST[cite: 3].
    """
    def __init__(self, api_key: str = None):
        # Puede configurar su API Key de Gemini u otro proveedor compatible
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "TU_API_KEY_AQUI")
        self.url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

    def analyze_results(self, costs_summary: dict, best_method_name: str, best_allocation: list, problem: TransportationProblem):
        """Construye dinámicamente el prompt y realiza la petición HTTP REST a la API de IA[cite: 3]."""
        
        allocation_str = ""
        for i, plant in enumerate(problem.plants):
            for j, dest in enumerate(problem.destinations):
                if best_allocation[i][j] > 0:
                    allocation_str += f"- {plant} hacia {dest}: {best_allocation[i][j]} unidades\n"

        prompt = (
            "Asume el rol de 'Consultor Logístico' experto en investigación de operaciones y optimización de cadenas de suministro. "
            "A continuación se presentan los resultados de la optimización de envíos de la empresa 'Rutas y Suministros C.A.'[cite: 1, 3]:\n\n"
            f"Costos totales obtenidos por método[cite: 3]:\n"
            f"1. Esquina Noroeste: ${costs_summary['Esquina Noroeste']}\n"
            f"2. Costo Mínimo: ${costs_summary['Costo Mínimo']}\n"
            f"3. Aproximación de Vogel (VAM): ${costs_summary['Vogel']}\n\n"
            f"El mejor método fue: {best_method_name}.\n"
            f"Matriz de asignación detallada del mejor método ({best_method_name}):\n{allocation_str}\n"
            "Por favor, devuelve una conclusión ejecutiva breve (máximo 2 párrafos) donde analices por qué un método superó a los demás matemáticamente "
            "y valide si la distribución final tiene sentido de negocio para una red de transporte[cite: 3]."
        )

        headers = {'Content-Type': 'application/json'}
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }

        # Si no se provee una API Key real, se simula la respuesta ejecutiva del Agente Analista para fines de demostración de la tarea
        if self.api_key == "TU_API_KEY_AQUI":
            return (
                "[Simulación de Respuesta del Agente IA - Consultor Logístico]:\n"
                "El método de Aproximación de Vogel (VAM) superó matemáticamente a los métodos de Esquina Noroeste y Costo Mínimo "
                "al priorizar estratégicamente las penalizaciones de costo más altas en cada iteración, evitando asignaciones ineficientes en rutas costosas. "
                "La distribución resultante tiene un sólido sentido de negocio, ya que asigna eficientemente la mayor parte de la capacidad de las plantas hacia los "
                "centros de demanda cercanos, minimizando el impacto financiero total en la cadena de suministros de 'Rutas y Suministros C.A.'"
            )

        try:
            response = requests.post(f"{self.url}?key={self.api_key}", json=payload, headers=headers)
            if response.status_code == 200:
                data = response.json()
                return data['candidates'][0]['content']['parts'][0]['text']
            else:
                return f"Error en la petición HTTP a la API de IA: Código {response.status_code} - {response.text}"
        except Exception as e:
            return f"Excepción de conexión con la API de IA: {str(e)}"


# --- Ejecución Principal del Programa ---
if __name__ == "__main__":
    print("=== INICIANDO SISTEMA DE OPTIMIZACIÓN LOGÍSTICA (UJAP) ===")
    
    # Instanciar el problema con los datos por defecto[cite: 1, 4]
    problem = TransportationProblem()
    
    # Módulo A: Motor Algorítmico[cite: 2]
    engine = AlgorithmicEngine(problem)
    
    alloc_nw, cost_nw = engine.northwest_corner()
    alloc_lc, cost_lc = engine.least_cost()
    alloc_v, cost_v = engine.vogel_approximation()
    
    costs_summary = {
        "Esquina Noroeste": cost_nw,
        "Costo Mínimo": cost_lc,
        "Vogel": cost_v
    }
    
    print("--- RESULTADOS MÓDULO A: COSTOS TOTALES ---")
    print(f"• Costo Total (Esquina Noroeste): ${cost_nw}[cite: 3]")
    print(f"• Costo Total (Costo Mínimo): ${cost_lc}[cite: 3]")
    print(f"• Costo Total (Vogel - VAM): ${cost_v}[cite: 3]\n")
    
    # Determinar el mejor método (menor costo)
    best_method = min(costs_summary, key=costs_summary.get)
    if best_method == "Esquina Noroeste":
        best_alloc = alloc_nw
    elif best_method == "Costo Mínimo":
        best_alloc = alloc_lc
    else:
        best_alloc = alloc_v
        
    print(f"Mejor método seleccionado: {best_method} con un costo de ${costs_summary[best_method]}\n")
    
    # Módulo B: Representación en Grafo[cite: 2]
    visualizer = NetworkVisualizer(problem)
    visualizer.generate_graph(alloc_nw, "Esquina Noroeste", "grafo_esquina_noroeste.png")
    visualizer.generate_graph(alloc_lc, "Costo Minimo", "grafo_costo_minimo.png")
    visualizer.generate_graph(alloc_v, "Aproximacion de Vogel", "grafo_vogel.png")
    print()
    
    # Módulo C: Integración con API de IA (Agente Analista)[cite: 3]
    ai_agent = AIAnalystAgent()
    ai_conclusion = ai_agent.analyze_results(costs_summary, best_method, best_alloc, problem)
    
    print("--- RESULTADOS MÓDULO C: CONCLUSIÓN DEL AGENTE IA ---")
    print(ai_conclusion)
    print("\n=== TAREA COMPLETADA EXITOSAMENTE ===")