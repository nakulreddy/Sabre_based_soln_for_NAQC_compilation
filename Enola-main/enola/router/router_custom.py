import math
import time
from typing import Any, Dict, List, Tuple

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def distance(
    p1: Tuple[int, int], 
    p2: Tuple[int, int], 
    use_euclid: bool = True
) -> float:
    """Euclidean (if use_euclid) or Manhattan distance."""
    dx, dy = p1[0] - p2[0], p1[1] - p2[1]
    return math.hypot(dx, dy) if use_euclid else abs(dx) + abs(dy)


def compatible_2d(
    v1: Tuple[int, int, int, int], 
    v2: Tuple[int, int, int, int]
) -> bool:
    """
    Return True if two move‐vectors v1=(x0,x1,y0,y1), v2=... do NOT conflict.
    Same logic as in router_mis.compatible_2D.
    """
    x0, x1, y0, y1 = v1
    X0, X1, Y0, Y1 = v2

    # same start‐row but different end‐row
    if y0 == Y0 and y1 != Y1:
        return False
    # same start‐col but different end‐col
    if x0 == X0 and x1 != X1:
        return False
    # crossing in X
    if (x0 < X0 and x1 >= X1) or (x0 > X0 and x1 <= X1):
        return False
    # crossing in Y
    if (y0 < Y0 and y1 >= Y1) or (y0 > Y0 and y1 <= Y1):
        return False
    return True


def mapping_cost(
    mapping: List[Tuple[int,int]],
    gates: List[Tuple[int,int]],
    use_euclid: bool
) -> float:
    """
    Sum of distances over all (a,b) in gates under the given mapping.
    """
    return sum(distance(mapping[a], mapping[b], use_euclid) for a,b in gates)



# -----------------------------------------------------------------------------
# Greedy, cost‐based routing function
# -----------------------------------------------------------------------------
def route_qubit_parallel_neighbors(
    chip_dim: Tuple[int,int],
    n_qubits: int,
    stage_idx: int,
    list_gates: List[List[Tuple[int,int]]],
    mapping: List[Tuple[int,int]],
    routing_strategy: str,
    reverse_to_initial: bool,
    use_euclid: bool,
    use_window: bool
) -> Tuple[Dict[str,Any], List[Tuple[int,int]], float]:
    """
    SabreSwap‐style routing: at each step pick the single move/swap that
    most reduces the total distance of the remaining gates.
    """
    start = time.time()
    width, height = chip_dim
    neighbor_thresh = 1.0

    # unsatisfied gates in this Rydberg stage
    remaining = list_gates[stage_idx].copy()

    # build initial layer
    layers: List[Dict[str,Any]] = [{
        "qubits": [
            {"id": i,
             "x": mapping[i][0],
             "y": mapping[i][1],
             "c": mapping[i][0],
             "r": mapping[i][1],
             "a": 0}
            for i in range(n_qubits)
        ],
        "gates": []
    }]

    batch = 0
    move_remaining = False
    remaining_moved = -1
    remaining_occupied = -1
    while remaining:
        move_remaining = False
        orig_cost = mapping_cost(mapping, list_gates[stage_idx], use_euclid)

        # 1) generate **all** candidate moves/swaps that strictly reduce at least one gate
        candidates: List[Tuple] = []
        for a, b in remaining:
            pa, pb = mapping[a], mapping[b]
            d_ab = distance(pa, pb, use_euclid)
            if d_ab <= neighbor_thresh:
                continue
            for q, p_old, p_other in ((a, pa, pb), (b, pb, pa)):
                x0, y0 = p_old
                for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                    x1, y1 = x0 + dx, y0 + dy
                    if not (0 <= x1 < width and 0 <= y1 < height):
                        continue
                    p_new = (x1, y1)
                    if p_new not in mapping:
                        # free cell → "move"
                        if distance(p_new, p_other, use_euclid) < d_ab:
                            candidates.append(("move", q, p_old, p_new, (a,b)))
                    else:
                        # occupied → "swap"
                        occ = mapping.index(p_new)
                        if distance(p_new, p_other, use_euclid) < d_ab:
                            candidates.append(("swap", q, p_old, p_new, (a,b), occ))

        # no possible improvements → stop
        if not candidates:
            break

        # 2) evaluate each candidate
        best_delta = 0.0
        best_cand  = None
        for cand in candidates:
            temp_map = mapping.copy()
            typ, q, old_pos, new_pos = cand[0], cand[1], cand[2], cand[3]
            if typ == "move":
                temp_map[q] = new_pos
            else:
                occ = cand[5]
                temp_map[q], temp_map[occ] = temp_map[occ], temp_map[q]

            delta = mapping_cost(temp_map, list_gates[stage_idx], use_euclid) - orig_cost
            if delta < best_delta:
                best_delta = delta
                best_cand  = cand

        # no candidate reduces cost → done
        if best_cand is None:
            break

        # unpack best candidate
        typ, moved, old_pos, new_pos = best_cand[0], best_cand[1], best_cand[2], best_cand[3]
        occupied = best_cand[5] if typ == "swap" else None

        # 3) record new layer *before* updating mapping
        layer_qubits = []
        for i in range(n_qubits):
            layer_qubits.append({
                "id": i,
                "x": mapping[i][0],
                "y": mapping[i][1],
                "c": mapping[i][0],
                "r": mapping[i][1],
                "a": 1 if i == moved or i == occupied else 0
            })
        gate_entry: Dict[str, Any] = {
            "type": typ,
            "q0": moved, ## this was gate_entry["qubit"]
            "from": list(old_pos),
            "to":   list(new_pos)
        }
        if occupied is not None:
            gate_entry["q1"] = occupied ## this was gate_entry["occupied"]
        # layers.append({
        #     "qubits": layer_qubits,
        #     "gates":  [gate_entry]
        # })
        layers.append({
            "qubits": layer_qubits,
            "gates":  []
        })

        batch += 1

        # 4) now commit the move/swap to the mapping for the next iteration
        if typ == "move":
            mapping[moved] = new_pos
        else:
            mapping[moved], mapping[occupied] = mapping[occupied], mapping[moved]
        
        move_remaining = True
        # remaining_moved = moved
        # remaining_occupied = occupied

        # 5) drop now‐satisfied gates
        new_remaining = []
        for (a, b) in remaining:
            if distance(mapping[a], mapping[b], use_euclid) > neighbor_thresh:
                new_remaining.append((a, b))
        remaining = new_remaining
    
    exec_layer_qubits = []
    # if move_remaining:
        # add the move that may have been missed in the last iteration
    for i in range(n_qubits):
        exec_layer_qubits.append({
            'id': i,
            'x': mapping[i][0],
            'y': mapping[i][1],
            'c': mapping[i][0],
            'r': mapping[i][1],
            'a': 0
        })
    

    # --- INSERTION POINT: execution layer for the real two‐qubit gates ---
    exec_layer = {
        'qubits': exec_layer_qubits,
        "gates": [
            {"id": i, "q0": a, "q1": b}
            for i, (a, b) in enumerate(list_gates[stage_idx])
        ]
    }
    layers.append(exec_layer)

    # optional reverse‐to‐initial (unchanged) …
    # if reverse_to_initial:
    #     layers[-1]["qubits"] = [
    #         {**q, "a": 0} for q in layers[-1]["qubits"]
    #     ]
    #     rev: List[Dict[str,Any]] = []
    #     for i in range(len(layers)-2, 0, -1):
    #         prev_a_map = {q["id"]: q["a"] for q in layers[i-1]["qubits"]}
    #         rev.append({
    #             "qubits": [
    #                 {"id": q["id"],
    #                  "x": q["x"], "y": q["y"],
    #                  "c": q["x"], "r": q["y"],
    #                  "a": prev_a_map[q["a"]]}
    #                 for q in layers[i]["qubits"]
    #             ],
    #             "gates": []
    #         })
    #     layers.extend(rev)
    #     layers.append({
    #         "qubits": [
    #             {"id": i,
    #              "x": mapping[i][0], "y": mapping[i][1],
    #              "c": mapping[i][0], "r": mapping[i][1],
    #              "a": 0}
    #             for i in range(n_qubits)
    #         ],
    #         "gates": []
    #     })
    if reverse_to_initial:
        for q in range(len(layers[-1]["qubits"])):
            layers[-1]["qubits"][q]["a"] = layers[-2]["qubits"][q]["a"]
        
        reverse_layers = []
        for q in range(len(layers[-1]["qubits"])):
            layers[len(layers)-1]["qubits"][q]["a"] = layers[len(layers)-2]["qubits"][q]["a"]
        for i in range(len(layers)-2, 0, -1):
            reverse_layers.append({
                                "qubits": [{
                                    "id": j,
                                    "a": 0,
                                    "x": layers[i]["qubits"][j]["x"],
                                    "y": layers[i]["qubits"][j]["y"],
                                    "c": layers[i]["qubits"][j]["x"],
                                    "r": layers[i]["qubits"][j]["y"],
                                } for j in range(n_qubits)],
                                "gates": [],
                            })
            reverse_layers[len(reverse_layers)-1]["gates"] = []
            for q in range(len(reverse_layers[-1]["qubits"])):
                reverse_layers[len(reverse_layers)-1]["qubits"][q]["a"] = layers[i-1]["qubits"][q]["a"]

                    
        layers = layers + reverse_layers

    total_time = time.time() - start
    return ({
        "runtime":     total_time,
        "no_transfer": False,
        "layers":      layers,
        "n_q":         n_qubits,
        "g_q":         list_gates
    }, mapping, total_time)
    # return ({
    #     'n_x': 3,
    #     'n_y': 3,
    #     'n_r': 4,
    #     'n_c': 4,
    #     "layers":      layers,
    #     "n_q":         n_qubits,
    #     "g_q":         list_gates
    # }, mapping, total_time)








# --- assume route_qubit_parallel_neighbors is already imported ---

    
# -----------------------------------------------------------------------------
# Test setup: 5 qubits in a row, 3 two‐qubit gates in one Rydberg stage
# -----------------------------------------------------------------------------

import json

def run_test(chip_dim, n_qubits, initial_map, list_gates, test_name):
    data, final_map, _ = route_qubit_parallel_neighbors(
        chip_dim,
        n_qubits,
        stage_idx=0,
        list_gates=list_gates,
        mapping=initial_map.copy(),
        routing_strategy="greedy",
        reverse_to_initial=True,
        use_euclid=True,
        use_window=False
    )
    print(f"--- {test_name} ---")
    data["runtime"] = 0.0
    # print(json.dumps(data, indent=2))
    print(data)
    print("Final mapping:", final_map)
    print()


def main():
    # Original single-row test (5 qubits, 1 row)
    chip_dim = (5, 1)
    n_qubits = 5
    initial_map = [(i, 0) for i in range(n_qubits)]
    list_gates = [[
        (0, 4),  # qubit 0 ↔ 4 (span 4)
        (1, 3),  # qubit 1 ↔ 3 (span 2)
        (2, 3)   # qubit 2 ↔ 3 (already adjacent)
    ]]
    # run_test(chip_dim, n_qubits, initial_map, list_gates, "Test 5x1 single row")

    # Test case 2x2 grid (4 qubits, 2 rows)
    chip_dim = (2, 2)
    n_qubits = 4
    initial_map = [
        (0, 0), (1, 0),  # row 0
        (0, 1), (1, 1)   # row 1
    ]
    list_gates = [[
        (0, 3),  # requires vertical swap between (0,0) and (1,1)
        (1, 2)   # requires vertical swap between (1,0) and (0,1)
    ]]
    # print(initial_map)
    # run_test(chip_dim, n_qubits, initial_map, list_gates, "Test 2x2 grid")

    # Test case 3x2 grid (6 qubits, 2 rows)
    chip_dim = (3, 2)
    n_qubits = 6
    initial_map = [(x, y) for y in range(2) for x in range(3)]
    list_gates = [[
        (0, 5),  # span across both rows and full width
        (2, 3),  # horizontal neighbor on separate rows
        (4, 1)   # vertical + horizontal movement
    ]]
    # print(initial_map)
    # (0, 0) ->0; (1, 0) ->1; (2, 0) ->2; (0, 1) ->3; (1, 1) ->4; (2, 1) ->5  
    # run_test(chip_dim, n_qubits, initial_map, list_gates, "Test 3x2 grid")

    # Test case 3x3 grid (9 qubits, 3 rows)
    chip_dim = (3, 3)
    n_qubits = 9
    initial_map = [(x, y) for y in range(3) for x in range(3)]
    list_gates = [[
        (0, 8),  # corner-to-corner diagonal
        (1, 7),  # near-diagonal
        (3, 5),  # vertical neighbor
        (2, 4)   # horizontal neighbor within middle row
    ]]
    print(initial_map)
    run_test(chip_dim, n_qubits, initial_map, list_gates, "Test 3x3 grid")
    # [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)]

    chip_dim = (4, 2)
    n_qubits = 3
    initial_map = [(0, 0), (1, 0), (3, 0)]
    list_gates = [[
        (1, 2),
        (0, 2)   
    ]]
    # print(initial_map)
    # run_test(chip_dim, n_qubits, initial_map, list_gates, "Test 4x2 grid")
if __name__ == "__main__":
    main()
