import math
import random

def place_qubit_sabre_layout(chip_dim, n_qubit, list_gate, l2=True):
    """
    A dynamic, stage-by-stage qubit mapping algorithm, loosely inspired
    by SabreLayout/SabreSwap, adapted to a grid-based DPQA scenario,
    with a constraint that no two qubits occupy the same (x, y) location.

    Parameters
    ----------
    chip_dim : tuple (width, height)
        Dimensions of the 2D grid.
    n_qubit : int
        Number of qubits to place.
    list_gate : list of list of tuples
        Each sub-list represents one "Rydberg stage" of parallel gates.
        A gate is a tuple (q1, q2), meaning a two-qubit interaction.
    l2 : bool
        If True, use Euclidean distance; if False, use Manhattan distance.

    Returns
    -------
    best_global_mapping : list of (x, y) positions (length = n_qubit)
        The final assigned coordinates for each logical qubit (IDs 0..n_qubit-1),
        with no duplicates.
    """

    # ------------------------------------------------------------------
    # 1. HELPER FUNCTIONS
    # ------------------------------------------------------------------
    def distance(p1, p2):
        """Compute distance between two positions p1=(x,y) and p2=(x,y)."""
        dx, dy = p1[0] - p2[0], p1[1] - p2[1]
        return math.sqrt(dx*dx + dy*dy) if l2 else abs(dx) + abs(dy)

    def stage_feasible(stage, mapping, threshold=0.5):
        """
        Check if all gates in a given stage are 'feasible' under the current mapping.
        A gate (q1, q2) is feasible if the distance between qubits is <= threshold.
        """
        for (q1, q2) in stage:
            if distance(mapping[q1], mapping[q2]) > threshold:
                return False
        return True

    def sabre_cost(mapping, stage_idx, lookahead=1, threshold=0.5):
        """
        Compute a heuristic score for the current mapping based on:
          - Current stage: list_gate[stage_idx]
          - Up to 'lookahead' future stages.
        Higher score indicates a better arrangement.
        """
        score = 0
        current_stage = list_gate[stage_idx]
        for (q1, q2) in current_stage:
            d = distance(mapping[q1], mapping[q2])
            if d <= threshold:
                score += 10
            else:
                score -= d
        future_stages = list_gate[stage_idx+1 : stage_idx+1+lookahead]
        for stage in future_stages:
            for (q1, q2) in stage:
                d = distance(mapping[q1], mapping[q2])
                if d <= threshold:
                    score += 5
                else:
                    score -= 0.5 * d
        return score

    def generate_candidate_moves(mapping, stage_idx, max_candidates=10):
        """
        Generate candidate moves:
          - Pairwise swaps of qubits.
          - Single-qubit shifts in one of four directions.
        Only return moves that keep qubits in unique positions.
        """
        candidates = []
        used = set(mapping)
        width, height = chip_dim

        # Generate candidate swaps
        num_swaps = min(max_candidates // 2, 5)
        for _ in range(num_swaps):
            q_a, q_b = random.sample(range(n_qubit), 2)
            pos_a, pos_b = mapping[q_a], mapping[q_b]
            if pos_a != pos_b:
                candidates.append(('swap', q_a, q_b))

        # Generate candidate single-qubit moves
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        remaining = max_candidates - len(candidates)
        for _ in range(remaining):
            q = random.randrange(n_qubit)
            x, y = mapping[q]
            dx, dy = random.choice(directions)
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < width and 0 <= new_y < height:
                if (new_x, new_y) not in used:
                    candidates.append(('move', q, (new_x, new_y)))
        random.shuffle(candidates)
        return candidates

    def apply_candidate_move(mapping, move):
        """
        Return a *new* mapping with the candidate move applied.
        For a 'swap' move, swap positions between two qubits.
        For a 'move' move, relocate the qubit to the new position.
        """
        new_mapping = list(mapping)
        if move[0] == 'swap':
            _, a, b = move
            new_mapping[a], new_mapping[b] = new_mapping[b], new_mapping[a]
        elif move[0] == 'move':
            _, q, new_pos = move
            new_mapping[q] = new_pos
        return new_mapping

    # ------------------------------------------------------------------
    # 2. INITIALIZATION
    # ------------------------------------------------------------------
    width, height = chip_dim

    # First, compute a tentative near-square layout.
    tentative_length = int(math.sqrt(n_qubit)) + 4
    # Check if the tentative layout (a square of tentative_length) is sufficient:
    if tentative_length * tentative_length >= n_qubit:
        # Use the near-square (compact) layout if it fits within provided dimensions.
        compact_width = min(width, tentative_length)
        compact_height = min(height, tentative_length)
        chip_dim = (compact_width, compact_height)
    # Otherwise, use full chip_dim as provided.
    width, height = chip_dim  # update

    # Generate all possible positions in the new chip_dim
    all_positions = [(x, y) for x in range(width) for y in range(height)]
    if len(all_positions) < n_qubit:
        raise ValueError("Not enough grid positions for all qubits.")

    best_global_mapping = None
    best_global_score = -float('inf')
    NUM_SEEDS = 5
    LOOKAHEAD = 3
    FEAS_THRESHOLD = 0.5
    MAX_ITER = 200

    # Try multiple random initializations (seeds)
    for s in range(NUM_SEEDS):
        random.shuffle(all_positions)
        current_mapping = list(all_positions[:n_qubit])
        # Process mapping stage by stage
        for stage_idx, stage in enumerate(list_gate):
            iter_count = 0
            while not stage_feasible(stage, current_mapping, FEAS_THRESHOLD):
                iter_count += 1
                if iter_count > MAX_ITER:
                    break
                base_score = sabre_cost(current_mapping, stage_idx, LOOKAHEAD, FEAS_THRESHOLD)
                candidates = generate_candidate_moves(current_mapping, stage_idx, max_candidates=20)
                best_move = None
                best_improvement = 0
                for move in candidates:
                    trial_mapping = apply_candidate_move(current_mapping, move)
                    if len(set(trial_mapping)) < len(trial_mapping):
                        continue
                    trial_score = sabre_cost(trial_mapping, stage_idx, LOOKAHEAD, FEAS_THRESHOLD)
                    improvement = trial_score - base_score
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = move
                if best_move is not None and best_improvement > 0:
                    current_mapping = apply_candidate_move(current_mapping, best_move)
                else:
                    # Fallback: take a random candidate move if available
                    if candidates:
                        random_move = random.choice(candidates)
                        trial_mapping = apply_candidate_move(current_mapping, random_move)
                        if len(set(trial_mapping)) == len(trial_mapping):
                            current_mapping = trial_mapping
                    else:
                        break

        # Score the final mapping for all stages
        final_score = sum(sabre_cost(current_mapping, s_idx, 0, FEAS_THRESHOLD) for s_idx in range(len(list_gate)))
        if final_score > best_global_score:
            best_global_score = final_score
            best_global_mapping = list(current_mapping)

    return best_global_mapping



import math
import random

def place_qubit_sabre_layout_with_backward_pass(chip_dim, n_qubit, list_gate, l2=True):
    """
    A full SabreLayout-inspired qubit mapping algorithm for DPQA.
    It performs an iterative forward pass, then a backward pass on the reversed circuit,
    using the backward pass's result as the new initial layout for the next forward pass.
    This bidirectional process is repeated several times to refine the mapping.

    Parameters
    ----------
    chip_dim : tuple (width, height)
        Dimensions of the 2D grid.
    n_qubit : int
        Number of qubits to place.
    list_gate : list of list of tuples
        Each sub-list represents one "Rydberg stage" of parallel two-qubit gates.
        A gate is a tuple (q1, q2), meaning a two-qubit interaction.
    l2 : bool
        If True, use Euclidean distance; if False, use Manhattan distance.

    Returns
    -------
    best_global_mapping : list of (x, y) positions (length = n_qubit)
        The final assigned coordinates for logical qubits 0..n_qubit-1, with no duplicates.
    """
    # ------------------------------------------------------------------
    # 1. HELPER FUNCTIONS
    # ------------------------------------------------------------------
    def distance(p1, p2):
        """Compute distance between two positions p1=(x,y) and p2=(x,y)."""
        dx, dy = p1[0] - p2[0], p1[1] - p2[1]
        return math.sqrt(dx*dx + dy*dy) if l2 else abs(dx) + abs(dy)

    def stage_feasible(stage, mapping, threshold=0.5):
        """
        Check if all gates in a given stage are feasible under the current mapping.
        A gate (q1, q2) is feasible if the distance between the two qubits is <= threshold.
        """
        for (q1, q2) in stage:
            if distance(mapping[q1], mapping[q2]) > threshold:
                return False
        return True

    def sabre_cost(mapping, stage_idx, lookahead=1, threshold=0.5):
        """
        Compute a heuristic score for the current mapping from stage stage_idx,
        looking ahead up to 'lookahead' stages.
        Higher score indicates better (i.e. lower distances).
        """
        score = 0
        current_stage = list_gate[stage_idx]
        for (q1, q2) in current_stage:
            d = distance(mapping[q1], mapping[q2])
            if d <= threshold:
                score += 10
            else:
                score -= d
        future_stages = list_gate[stage_idx+1 : stage_idx+1+lookahead]
        for stage in future_stages:
            for (q1, q2) in stage:
                d = distance(mapping[q1], mapping[q2])
                if d <= threshold:
                    score += 5
                else:
                    score -= 0.5 * d
        return score

    def generate_candidate_moves(mapping, stage_idx, max_candidates=10):
        """
        Generate candidate moves:
          - Pairwise swaps of qubits.
          - Single-qubit shifts in one of four directions.
        Only returns moves that keep all qubits in unique positions.
        """
        candidates = []
        used = set(mapping)
        width, height = chip_dim

        # Candidate swaps
        num_swaps = min(max_candidates // 2, 5)
        for _ in range(num_swaps):
            q_a, q_b = random.sample(range(n_qubit), 2)
            pos_a, pos_b = mapping[q_a], mapping[q_b]
            if pos_a != pos_b:
                candidates.append(('swap', q_a, q_b))

        # Candidate single-qubit moves (shift in one of the four directions)
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        remaining = max_candidates - len(candidates)
        for _ in range(remaining):
            q = random.randrange(n_qubit)
            x, y = mapping[q]
            dx, dy = random.choice(directions)
            new_pos = (x + dx, y + dy)
            if 0 <= new_pos[0] < width and 0 <= new_pos[1] < height:
                if new_pos not in used:
                    candidates.append(('move', q, new_pos))
        random.shuffle(candidates)
        return candidates

    def apply_candidate_move(mapping, move):
        """
        Apply a candidate move to the mapping:
          - 'swap': Exchange positions of two qubits.
          - 'move': Assign a new position to a single qubit.
        Returns a new mapping list.
        """
        new_mapping = list(mapping)
        if move[0] == 'swap':
            _, a, b = move
            new_mapping[a], new_mapping[b] = new_mapping[b], new_mapping[a]
        elif move[0] == 'move':
            _, q, new_pos = move
            new_mapping[q] = new_pos
        return new_mapping

    def optimize_mapping_for_stages(stages, mapping):
        """
        Optimize the given mapping by processing each stage sequentially.
        For each stage, candidate moves are generated to improve the feasibility
        (i.e. to reduce the distance between interacting qubits).
        """
        for stage_idx, stage in enumerate(stages):
            iter_count = 0
            while not stage_feasible(stage, mapping, FEAS_THRESHOLD):
                iter_count += 1
                if iter_count > MAX_ITER:
                    break
                base_score = sabre_cost(mapping, stage_idx, LOOKAHEAD, FEAS_THRESHOLD)
                candidates = generate_candidate_moves(mapping, stage_idx, max_candidates=20)
                best_move = None
                best_improvement = 0
                for move in candidates:
                    trial_mapping = apply_candidate_move(mapping, move)
                    if len(set(trial_mapping)) < len(trial_mapping):
                        continue
                    trial_score = sabre_cost(trial_mapping, stage_idx, LOOKAHEAD, FEAS_THRESHOLD)
                    improvement = trial_score - base_score
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = move
                if best_move is not None and best_improvement > 0:
                    mapping = apply_candidate_move(mapping, best_move)
                else:
                    if candidates:
                        random_move = random.choice(candidates)
                        trial_mapping = apply_candidate_move(mapping, random_move)
                        if len(set(trial_mapping)) == len(trial_mapping):
                            mapping = trial_mapping
                    else:
                        break
        return mapping

    # ------------------------------------------------------------------
    # 2. INITIALIZATION
    # ------------------------------------------------------------------
    width, height = chip_dim
    # First, compute a tentative near-square layout.
    tentative_length = int(math.sqrt(n_qubit)) + 4
    # Use the near-square compact layout if it provides enough positions:
    if tentative_length * tentative_length >= n_qubit:
        compact_width = min(width, tentative_length)
        compact_height = min(height, tentative_length)
        chip_dim = (compact_width, compact_height)
    width, height = chip_dim  # update chip dimensions

    all_positions = [(x, y) for x in range(width) for y in range(height)]
    if len(all_positions) < n_qubit:
        raise ValueError("Not enough grid positions for all qubits.")

    NUM_SEEDS = 5
    LOOKAHEAD = 3
    FEAS_THRESHOLD = 0.5
    MAX_ITER = 200
    NUM_ITER = 3  # number of forward-backward iterations

    best_global_mapping = None
    best_global_score = float('inf')  # lower total distance is better

    # Try multiple random initial seeds:
    for s in range(NUM_SEEDS):
        random.shuffle(all_positions)
        current_mapping = list(all_positions[:n_qubit])
        # Forward pass: Process stages in original order
        mapping_forward = optimize_mapping_for_stages(list_gate, current_mapping)
        # Backward pass: Process stages in reverse order using forward mapping as seed
        mapping_backward = optimize_mapping_for_stages(list_gate[::-1], mapping_forward)
        # Iterate forward-backward several times for refinement
        for _ in range(NUM_ITER):
            mapping_forward = optimize_mapping_for_stages(list_gate, mapping_backward)
            mapping_backward = optimize_mapping_for_stages(list_gate[::-1], mapping_forward)
        final_mapping = mapping_backward
        
        # Evaluate overall mapping cost (sum of distances for all two-qubit gates)
        total_cost = sum(sabre_cost(final_mapping, idx, 0, FEAS_THRESHOLD)
                         for idx in range(len(list_gate)))
        # Lower total cost is better
        if total_cost < best_global_score:
            best_global_score = total_cost
            best_global_mapping = list(final_mapping)

    return best_global_mapping


