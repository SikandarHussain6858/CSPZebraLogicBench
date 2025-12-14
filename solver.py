import pandas as pd
import re
from enum import Enum
from typing import Tuple
import ast

# Constants
AND_SEPARATOR = " and "


class Operation(Enum):
    EQUAL = 1
    NOT_EQUAL = 2
    LEFT_OF = 3
    DIRECTLY_LEFT_OF = 4
    RIGHT_OF = 5
    DIRECTLY_RIGHT_OF = 6
    NEXT_TO = 7
    BETWEEN = 8
    N_HOUSES_BETWEEN = 9


class House:
    """A house with feature domains that narrow as constraints are applied."""

    def __init__(self, number: int, features: dict):
        self.number = number
        self.features = {k: list(v) for k, v in features.items()}

    def remove_feature(self, feature: str, *values: str):
        self.features[feature] = [
            v for v in self.features[feature] if v not in values]

        if not self.features[feature]:
            raise ValueError(
                f"No values left for feature '{feature}' in house {self.number}.")

    def remove_remaining_feature_values(self, feature: str, value: str):
        self.features[feature] = [
            v for v in self.features[feature] if v == value]

        if not self.features[feature]:
            raise ValueError(
                f"No values left for feature '{feature}' in house {self.number}.")

    def get_feature_values(self, feature: str) -> list[str]:
        return self.features.get(feature, [])

    def __repr__(self):
        return f"House({self.number}, {self.features})"


class Constraint:
    def __init__(self, left: str, right: str, operator: Operation, distance: int = 0):
        if not (left and right and operator):
            raise ValueError(
                "Constraint must have left, right, and operator defined.")
        self.left = left
        self.right = right
        self.operator = operator
        self.distance = distance

    def __repr__(self):
        if self.distance > 0:
            return f"Constraint({self.left} {self.operator.name}(dist={self.distance}) {self.right})"
        return f"Constraint({self.left} {self.operator.name} {self.right})"


def _enforce_alldiff_constraint(features: dict, houses: list[House]) -> bool | None:
    """Enforce each value appears in exactly one house per feature."""
    changes_made = False

    for feature, values in features.items():
        for value in values:
            possible_houses = [
                h for h in houses if value in h.get_feature_values(feature)]

            # If only one house can have this value, assign it
            if len(possible_houses) == 1:
                house = possible_houses[0]
                if len(house.get_feature_values(feature)) > 1:
                    try:
                        house.remove_remaining_feature_values(feature, value)
                        for other_house in houses:
                            if other_house != house:
                                other_house.remove_feature(feature, value)
                        changes_made = True
                    except ValueError:
                        return None

            # If a value has no possible house, unsolvable
            elif len(possible_houses) == 0:
                return None

    return changes_made


def _enforce_unique_assignment(features: dict, houses: list[House]) -> bool | None:
    """Assign value if no other house can have it."""
    changes_made = False

    for feature, values in features.items():
        for house in houses:
            house_values = house.get_feature_values(feature)
            if len(house_values) > 1:
                for value in house_values:
                    other_possibilities = [
                        h for h in houses if h != house and value in h.get_feature_values(feature)]
                    if len(other_possibilities) == 0 and value in features.get(feature, []):
                        try:
                            house.remove_remaining_feature_values(
                                feature, value)
                            for other_house in houses:
                                if other_house != house:
                                    other_house.remove_feature(feature, value)
                            changes_made = True
                            break
                        except ValueError:
                            return None

    return changes_made


def _propagate_constraints(features: dict, houses: list[House]):
    """Apply constraint propagation to narrow domains."""
    # Enforce alldiff constraint
    result = _enforce_alldiff_constraint(features, houses)
    if result is None:
        return None

    changes_from_alldiff = result

    # Enforce unique assignment constraint
    result = _enforce_unique_assignment(features, houses)
    if result is None:
        return None

    changes_from_unique = result

    return changes_from_alldiff or changes_from_unique


def _synchronize_equal_values(constraint: Constraint, left_feature: str, right_feature: str,
                              houses: list[House]) -> bool:
    """Ensure A and B are in the same house."""
    changes_made = False

    for house in houses:
        left_vals = house.get_feature_values(left_feature)
        right_vals = house.get_feature_values(right_feature)

        # If left value is UNIQUELY assigned to this house, right value must be too
        if len(left_vals) == 1 and constraint.left == left_vals[0] and constraint.right in right_vals:
            if len(right_vals) > 1:
                try:
                    house.remove_remaining_feature_values(
                        right_feature, constraint.right)
                    for other_house in houses:
                        if other_house != house:
                            other_house.remove_feature(
                                right_feature, constraint.right)
                    changes_made = True
                except ValueError:
                    pass

        # If right value is UNIQUELY assigned to this house, left value must be too
        if len(right_vals) == 1 and constraint.right == right_vals[0] and constraint.left in left_vals:
            if len(left_vals) > 1:
                try:
                    house.remove_remaining_feature_values(
                        left_feature, constraint.left)
                    for other_house in houses:
                        if other_house != house:
                            other_house.remove_feature(
                                left_feature, constraint.left)
                    changes_made = True
                except ValueError:
                    pass

    return changes_made


def _eliminate_impossible_pairs(constraint: Constraint, left_feature: str, right_feature: str,
                                houses: list[House]) -> bool:
    """Eliminate impossible pairings: if A is uniquely assigned but B can't be, remove A."""
    changes_made = False

    for house in houses:
        left_vals = house.get_feature_values(left_feature)
        right_vals = house.get_feature_values(right_feature)

        # If left is uniquely assigned but right can't be, remove left
        if len(left_vals) == 1 and constraint.left == left_vals[0] and constraint.right not in right_vals:
            try:
                house.remove_feature(left_feature, constraint.left)
                changes_made = True
            except ValueError:
                pass

        # If right is uniquely assigned but left can't be, remove right
        if len(right_vals) == 1 and constraint.right == right_vals[0] and constraint.left not in left_vals:
            try:
                house.remove_feature(right_feature, constraint.right)
                changes_made = True
            except ValueError:
                pass

    return changes_made


def _apply_equality_propagation(constraint: Constraint, features: dict, houses: list[House]) -> bool:
    """Synchronize domains when A = B (must be in same house)."""
    if constraint.operator != Operation.EQUAL:
        return False

    try:
        left_feature = _find_feature(constraint.left, features)
        right_feature = _find_feature(constraint.right, features)
    except ValueError:
        return False

    # Synchronize values that must be equal
    changes1 = _synchronize_equal_values(
        constraint, left_feature, right_feature, houses)

    # Eliminate impossible pairings
    changes2 = _eliminate_impossible_pairs(
        constraint, left_feature, right_feature, houses)

    return changes1 or changes2


def _check_constraint(constraint: Constraint, left_val: str, right_val: str, left_pos: int, right_pos: int) -> bool:
    """Check if constraint is satisfied by given values and positions."""
    operator = constraint.operator

    # Check value assignments match constraint
    if left_val != constraint.left or right_val != constraint.right:
        return operator == Operation.NOT_EQUAL

    # Check positional relationships based on operator
    if operator == Operation.EQUAL:
        return left_pos == right_pos
    elif operator == Operation.NOT_EQUAL:
        return left_pos != right_pos
    elif operator == Operation.DIRECTLY_LEFT_OF:
        return left_pos + 1 == right_pos
    elif operator == Operation.DIRECTLY_RIGHT_OF:
        return right_pos + 1 == left_pos
    elif operator == Operation.LEFT_OF:
        return left_pos < right_pos
    elif operator == Operation.RIGHT_OF:
        return right_pos < left_pos
    elif operator == Operation.NEXT_TO:
        return abs(left_pos - right_pos) == 1
    elif operator == Operation.BETWEEN:
        return min(left_pos, right_pos) + 1 < max(left_pos, right_pos)
    elif operator == Operation.N_HOUSES_BETWEEN:
        expected_distance = constraint.distance + 1
        return abs(left_pos - right_pos) == expected_distance

    return True


def _backtrack_solve(domains: dict, features: dict, constraints_parsed: list, house_count: int, depth: int = 0, constraint_cache: dict = None) -> dict:
    """Search for complete assignment using backtracking with MRV heuristic and optimizations."""
    # Limit recursion depth to prevent infinite loops
    # Increased limit for complex puzzles
    if depth > 2000:
        return {}
    
    # Initialize constraint cache on first call
    if constraint_cache is None:
        constraint_cache = {}
        # Pre-compute feature lookups for constraints to avoid repeated searches
        for constraint in constraints_parsed:
            try:
                left_feature = _find_feature(constraint.left, features)
                right_feature = _find_feature(constraint.right, features)
                constraint_cache[id(constraint)] = (left_feature, right_feature)
            except ValueError:
                constraint_cache[id(constraint)] = (None, None)

    # Ensure all domains are initialized
    for h_num in range(1, house_count + 1):
        if h_num not in domains:
            domains[h_num] = {f: list(values)
                              for f, values in features.items()}

    # Check if complete assignment (each feature has exactly one value per house)
    all_assigned = True
    assignment = {}

    for h_num in range(1, house_count + 1):
        assignment[h_num] = {}
        for feature in features:
            if feature not in domains[h_num] or len(domains[h_num][feature]) != 1:
                all_assigned = False
            else:
                assignment[h_num][feature] = domains[h_num][feature][0]

    if all_assigned:
        # Verify all constraints are satisfied
        # Also ensure alldiff: no two houses have the same singleton for a feature
        for feature in features:
            seen = set()
            for h_num in range(1, house_count + 1):
                val = assignment[h_num].get(feature)
                if val is None:
                    continue
                if val in seen:
                    # Duplicate assignment for a feature - invalid
                    return {}
                seen.add(val)
        for constraint in constraints_parsed:
            # Check for positional constraints (where right is a house number string)
            try:
                house_num = int(constraint.right)
                # This is a positional constraint
                if constraint.operator == Operation.EQUAL:
                    # Value (constraint.left) must be in house house_num (1-indexed)
                    found = False
                    for f, value in assignment.get(house_num, {}).items():
                        if value == constraint.left:
                            found = True
                            break
                    if not found:
                        return {}  # Positional constraint violated
                elif constraint.operator == Operation.NOT_EQUAL:
                    # Value must NOT be in house house_num
                    for f, value in assignment.get(house_num, {}).items():
                        if value == constraint.left:
                            return {}  # Positional constraint violated
            except (ValueError, TypeError):
                # Not a positional constraint, handle feature-to-feature constraints
                left_pos = None
                right_pos = None

                for h_num in range(1, house_count + 1):
                    for f, value in assignment[h_num].items():
                        if value == constraint.left:
                            left_pos = h_num - 1  # 0-indexed
                        if value == constraint.right:
                            right_pos = h_num - 1  # 0-indexed

                # If both values are assigned, check constraint satisfaction
                if left_pos is not None and right_pos is not None:
                    if not _check_constraint(constraint, constraint.left, constraint.right, left_pos, right_pos):
                        return {}
                # If only left is assigned, do early pruning for directional constraints
                elif left_pos is not None and constraint.operator in (Operation.LEFT_OF, Operation.DIRECTLY_LEFT_OF, Operation.RIGHT_OF):
                    # Check if constraint is violated based on left's position
                    if ((constraint.operator == Operation.LEFT_OF and left_pos == house_count - 1) or
                        (constraint.operator == Operation.DIRECTLY_LEFT_OF and left_pos == house_count - 1) or
                            (constraint.operator == Operation.RIGHT_OF and left_pos == 0)):
                        return {}
                # If only right is assigned, do early pruning
                elif right_pos is not None and constraint.operator in (Operation.LEFT_OF, Operation.DIRECTLY_LEFT_OF, Operation.RIGHT_OF):
                    # Check if constraint is violated based on right's position
                    if ((constraint.operator == Operation.LEFT_OF and right_pos == 0) or
                        (constraint.operator == Operation.DIRECTLY_LEFT_OF and right_pos == 0) or
                            (constraint.operator == Operation.RIGHT_OF and right_pos == house_count - 1)):
                        return {}

        return domains

    # Find unassigned variable with smallest domain (MRV heuristic)
    best_var = None
    best_domain_size = float('inf')

    for h_num in range(1, house_count + 1):
        for feature in features:
            domain_size = len(domains[h_num].get(feature, []))
            # If any domain is empty, this is unsolvable
            if domain_size == 0:
                return {}
            # Pick smallest non-unit domain for backtracking
            if domain_size > 1 and domain_size < best_domain_size:
                best_domain_size = domain_size
                best_var = (h_num, feature)

    if best_var is None:
        return {}

    h_num, feature = best_var

    # Try each value in domain
    for value in domains[h_num][feature]:
        # Quick early pruning: check if this value violates any constraint using cache
        can_assign = True

        # Check EQUAL constraints - if this value is part of an EQUAL constraint,
        # check if its partner can exist
        for constraint in constraints_parsed:
            if constraint.operator == Operation.EQUAL:
                # Use cached feature lookups
                left_feature, right_feature = constraint_cache.get(id(constraint), (None, None))
                if left_feature is None or right_feature is None:
                    continue

                # If we're assigning the left value, check if right is possible in this house
                if left_feature == feature and constraint.left == value:
                    if constraint.right not in domains[h_num].get(right_feature, []):
                        can_assign = False
                        break

                # If we're assigning the right value, check if left is possible in this house
                if right_feature == feature and constraint.right == value:
                    if constraint.left not in domains[h_num].get(left_feature, []):
                        can_assign = False
                        break

        if not can_assign:
            continue
        # Save old domains - optimize by only saving what we change
        saved_domains = {}
        saved_domains[h_num] = {feature: domains[h_num][feature].copy()}
        
        # Track other houses we modify for this feature
        modified_houses = []
        for other_h in range(1, house_count + 1):
            if other_h != h_num and value in domains[other_h][feature]:
                if other_h not in saved_domains:
                    saved_domains[other_h] = {}
                saved_domains[other_h][feature] = domains[other_h][feature].copy()
                modified_houses.append(other_h)

        # Assign value
        domains[h_num][feature] = [value]

        # Forward check: remove value from other houses for this feature
        valid = True
        for other_h in modified_houses:
            domains[other_h][feature].remove(value)
            if not domains[other_h][feature]:
                    valid = False
                    break

        # Early constraint violation check: Check if this assignment violates any constraint
        if valid:
            # Build partial assignment for constraint checking (only singleton domains)
            partial_assignment = {}
            for h in range(1, house_count + 1):
                partial_assignment[h] = {}
                for f in features:
                    if len(domains[h].get(f, [])) == 1:
                        partial_assignment[h][f] = domains[h][f][0]

            # Check constraints against partial assignment
            for constraint in constraints_parsed:
                # Find positions if values are determined
                left_pos = None
                right_pos = None
                left_feature, right_feature = constraint_cache.get(id(constraint), (None, None))

                for h in range(1, house_count + 1):
                    for f, val in partial_assignment[h].items():
                        if val == constraint.left:
                            left_pos = h
                            left_feature = f
                        if val == constraint.right:
                            right_pos = h
                            right_feature = f

                # If both positions are determined, check constraint
                if left_pos is not None and right_pos is not None:
                    if not _check_constraint(constraint, constraint.left, constraint.right, left_pos - 1, right_pos - 1):
                        valid = False
                        break

                # If one position is determined, check viability of the other
                # If one position is determined but the other's position is not,
                # ensure the undecided value can still be placed in some house.
                if valid and left_pos is not None and right_pos is None and right_feature is not None:
                    # Check if the right value is possible in any house's domain for right_feature
                    exists = any(constraint.right in domains[h].get(
                        right_feature, []) for h in range(1, house_count + 1))
                    if not exists:
                        valid = False
                        break

                if valid and right_pos is not None and left_pos is None and left_feature is not None:
                    # Check if the left value is possible in any house's domain for left_feature
                    exists = any(constraint.left in domains[h].get(
                        left_feature, []) for h in range(1, house_count + 1))
                    if not exists:
                        valid = False
                        break

        # Recursively try to complete assignment
        if valid:
            result = _backtrack_solve(
                domains, features, constraints_parsed, house_count, depth + 1, constraint_cache)
            if result:
                return result

        # Backtrack: restore only the domains we saved
        for h, saved_features in saved_domains.items():
            for f, saved_domain in saved_features.items():
                domains[h][f] = saved_domain

    return {}


def solve_puzzle(features: dict, constraints: list[str], house_count: int) -> list[House]:
    """Solve CSP using AC-3 propagation followed by backtracking."""
    canonical_features = {}
    feature_keys = list(features.keys())
    for k in feature_keys:
        canonical_key = k.lower().rstrip('s') if k.endswith('s') else k.lower()

        vals = [str(v).strip().lower() for v in features[k]]
        if canonical_key not in canonical_features:
            canonical_features[canonical_key] = []
        for val in vals:
            if val not in canonical_features[canonical_key]:
                canonical_features[canonical_key].append(val)

    features = canonical_features

    houses = [House(i, features) for i in range(1, house_count + 1)]

    parsed_constraints = []
    for constraint in constraints:
        try:
            parsed_constraints.append(_parse_constraints(features, constraint))
        except Exception:
            continue

    max_iterations = 100
    iteration = 0
    puzzle_unsolvable = False

    while iteration < max_iterations:
        iteration += 1
        changes_made = False

        result = _propagate_constraints(features, houses)
        if result is None:
            puzzle_unsolvable = True
            break
        elif result:
            changes_made = True

        for constraint in parsed_constraints:
            try:
                if _apply_positional_not_constraint(constraint, features, houses):
                    changes_made = True
            except ValueError:
                puzzle_unsolvable = True
                break

        if puzzle_unsolvable:
            break

        for constraint in parsed_constraints:
            try:
                if _apply_positional_equal_constraint(constraint, features, houses):
                    changes_made = True
            except ValueError:
                puzzle_unsolvable = True
                break

        if puzzle_unsolvable:
            break

        for constraint in parsed_constraints:
            try:
                if _apply_equality_propagation(constraint, features, houses):
                    changes_made = True
            except ValueError:
                puzzle_unsolvable = True
                break

        if puzzle_unsolvable:
            break

        for constraint in parsed_constraints:
            try:
                if _apply_directional_constraint(constraint, features, houses):
                    changes_made = True
            except ValueError:
                puzzle_unsolvable = True
                break

        if puzzle_unsolvable:
            break

        if not changes_made:
            break

    # Check if fully solved
    fully_solved = all(
        all(len(house.get_feature_values(feature)) == 1 for feature in features)
        for house in houses
    )

    # Check if any domain is empty (unsatisfiable state)
    has_empty_domain = any(
        len(house.get_feature_values(feature)) == 0 for house in houses for feature in features
    )

    if (not fully_solved and not puzzle_unsolvable) or has_empty_domain:
        if has_empty_domain:
            for house in houses:
                for feature in features:
                    if len(house.get_feature_values(feature)) == 0:
                        assigned_values = set()
                        for other_house in houses:
                            other_vals = other_house.get_feature_values(
                                feature)
                            if len(other_vals) == 1:
                                assigned_values.add(other_vals[0])
                        house.features[feature] = [
                            v for v in features[feature] if v not in assigned_values]

        domains = {}
        for house in houses:
            domains[house.number] = {
                f: house.get_feature_values(f).copy() for f in features}

        result = _backtrack_solve(
            domains, features, parsed_constraints, house_count)

        if result is not None and result:
            for feature in features:
                seen_values = set()
                result_valid = True
                for h_num in range(1, house_count + 1):
                    if h_num in result:
                        for val in result[h_num].get(feature, []):
                            if len(result[h_num][feature]) == 1 and val in seen_values:
                                result_valid = False
                                break
                            if len(result[h_num][feature]) == 1:
                                seen_values.add(val)
                if not result_valid:
                    result = {}
                    break

        if result is not None and result:
            for h_num in range(1, house_count + 1):
                if h_num in result:
                    for feature, values in result[h_num].items():
                        if len(values) == 1:
                            try:
                                houses[h_num -
                                       1].remove_remaining_feature_values(feature, values[0])
                            except ValueError:
                                pass

    for house in houses:
        for feature in features:
            feature_values = house.get_feature_values(feature)
            if len(feature_values) != 1:
                return []

    for feature in features:
        values_seen = set()
        for house in houses:
            house_vals = house.get_feature_values(feature)
            if len(house_vals) == 1:
                val = house_vals[0]
                if val in values_seen:
                    return []
                values_seen.add(val)

    for constraint in parsed_constraints:
        left_feature = None
        right_feature = None
        try:
            left_feature = _find_feature(constraint.left, features)
            right_feature = _find_feature(constraint.right, features)
        except ValueError:
            pass

        if left_feature is None or right_feature is None:
            try:
                position = int(constraint.right)
                if left_feature is None:
                    try:
                        left_feature = _find_feature(constraint.left, features)
                    except ValueError:
                        continue
                found = False
                for house in houses:
                    if constraint.left in house.get_feature_values(left_feature):
                        if ((constraint.operator == Operation.EQUAL and house.number == position) or
                                (constraint.operator == Operation.NOT_EQUAL and house.number != position)):
                            found = True
                            break
                if not found and constraint.operator == Operation.EQUAL:
                    return []
            except ValueError:
                pass
            continue

        left_pos = None
        right_pos = None

        for house in houses:
            left_vals = house.get_feature_values(left_feature)
            right_vals = house.get_feature_values(right_feature)

            if len(left_vals) == 1 and left_vals[0] == constraint.left:
                left_pos = house.number - 1
            if len(right_vals) == 1 and right_vals[0] == constraint.right:
                right_pos = house.number - 1

        if left_pos is not None and right_pos is not None:
            if not _check_constraint(constraint, constraint.left, constraint.right, left_pos, right_pos):
                return []

    return houses


def _apply_positional_not_constraint(constraint: Constraint, features: dict, houses: list[House]) -> bool:
    """Remove value from specified house (X is not in house N)."""
    if constraint.operator != Operation.NOT_EQUAL:
        return False

    # Check if this is a positional NOT constraint (right side is a house number)
    try:
        position = int(constraint.right)
        if position < 1:
            return False
    except ValueError:
        # Not a positional constraint - handle normally
        return False

    # Find the feature that contains the left value
    try:
        feature = _find_feature(constraint.left, features)
    except ValueError:
        return False

    changes_made = False

    # Remove the value from the specified house
    for house in houses:
        if house.number == position:
            if constraint.left in house.get_feature_values(feature):
                try:
                    house.remove_feature(feature, constraint.left)
                    changes_made = True
                except ValueError:
                    pass
            break

    return changes_made


def _apply_positional_equal_constraint(constraint: Constraint, features: dict, houses: list[House]) -> bool:
    """Assign value to specified house (X is in house N)."""
    if constraint.operator != Operation.EQUAL:
        return False

    # Check if this is a positional EQUAL constraint (right side is a house number)
    try:
        position = int(constraint.right)
        if position < 1:
            return False
    except ValueError:
        # Not a positional constraint - handle normally
        return False

    # Find the feature that contains the left value
    try:
        feature = _find_feature(constraint.left, features)
    except ValueError:
        return False

    changes_made = False

    # Assign value to target house
    target_house = next((h for h in houses if h.number == position), None)
    if target_house and constraint.left in target_house.get_feature_values(feature):
        if len(target_house.get_feature_values(feature)) > 1:
            try:
                target_house.remove_remaining_feature_values(
                    feature, constraint.left)
                # Remove from all other houses
                for other_house in houses:
                    if other_house != target_house:
                        other_house.remove_feature(feature, constraint.left)
                changes_made = True
            except ValueError:
                pass

    # Remove value from all other houses
    for house in houses:
        if house.number != position:
            if constraint.left in house.get_feature_values(feature):
                try:
                    house.remove_feature(feature, constraint.left)
                    changes_made = True
                except ValueError:
                    pass

    return changes_made


def _apply_directional_constraint(constraint: Constraint, features: dict, houses: list[House]) -> bool:
    """Apply directional constraints (LEFT_OF, RIGHT_OF, DIRECTLY_LEFT_OF, etc.) during propagation."""
    if constraint.operator not in (Operation.LEFT_OF, Operation.RIGHT_OF, 
                                   Operation.DIRECTLY_LEFT_OF, Operation.DIRECTLY_RIGHT_OF,
                                   Operation.NEXT_TO):
        return False
    
    # Find features for left and right values
    try:
        left_feature = _find_feature(constraint.left, features)
        right_feature = _find_feature(constraint.right, features)
    except ValueError:
        return False
    
    changes_made = False
    house_count = len(houses)
    
    # For each house, check if the constraint can be satisfied
    # If a value is in a position where the constraint cannot be satisfied, remove it
    for house in houses:
        left_vals = house.get_feature_values(left_feature)
        right_vals = house.get_feature_values(right_feature)
        
        # Check if left value can be in this house
        if constraint.left in left_vals:
            can_satisfy = False
            
            if constraint.operator == Operation.DIRECTLY_LEFT_OF:
                # Left must be directly before right, so right must be able to be in house.number + 1
                if house.number < house_count:
                    next_house = houses[house.number]  # 0-indexed, so house.number is next
                    if constraint.right in next_house.get_feature_values(right_feature):
                        can_satisfy = True
            elif constraint.operator == Operation.DIRECTLY_RIGHT_OF:
                # Left must be directly after right, so right must be able to be in house.number - 1
                if house.number > 1:
                    prev_house = houses[house.number - 2]  # 0-indexed
                    if constraint.right in prev_house.get_feature_values(right_feature):
                        can_satisfy = True
            elif constraint.operator == Operation.LEFT_OF:
                # Left must be before right, so right must be able to be in any house after this
                for h in range(house.number, house_count):
                    if constraint.right in houses[h].get_feature_values(right_feature):
                        can_satisfy = True
                        break
            elif constraint.operator == Operation.RIGHT_OF:
                # Left must be after right, so right must be able to be in any house before this
                for h in range(0, house.number - 1):
                    if constraint.right in houses[h].get_feature_values(right_feature):
                        can_satisfy = True
                        break
            elif constraint.operator == Operation.NEXT_TO:
                # Left must be next to right (either side)
                if house.number > 1:
                    prev_house = houses[house.number - 2]
                    if constraint.right in prev_house.get_feature_values(right_feature):
                        can_satisfy = True
                if house.number < house_count:
                    next_house = houses[house.number]
                    if constraint.right in next_house.get_feature_values(right_feature):
                        can_satisfy = True
            
            if not can_satisfy:
                try:
                    house.remove_feature(left_feature, constraint.left)
                    changes_made = True
                except ValueError:
                    pass
        
        # Check if right value can be in this house
        if constraint.right in right_vals:
            can_satisfy = False
            
            if constraint.operator == Operation.DIRECTLY_LEFT_OF:
                # Right must be directly after left, so left must be able to be in house.number - 1
                if house.number > 1:
                    prev_house = houses[house.number - 2]
                    if constraint.left in prev_house.get_feature_values(left_feature):
                        can_satisfy = True
            elif constraint.operator == Operation.DIRECTLY_RIGHT_OF:
                # Right must be directly before left, so left must be able to be in house.number + 1
                if house.number < house_count:
                    next_house = houses[house.number]
                    if constraint.left in next_house.get_feature_values(left_feature):
                        can_satisfy = True
            elif constraint.operator == Operation.LEFT_OF:
                # Right must be after left, so left must be able to be in any house before this
                for h in range(0, house.number - 1):
                    if constraint.left in houses[h].get_feature_values(left_feature):
                        can_satisfy = True
                        break
            elif constraint.operator == Operation.RIGHT_OF:
                # Right must be before left, so left must be able to be in any house after this
                for h in range(house.number, house_count):
                    if constraint.left in houses[h].get_feature_values(left_feature):
                        can_satisfy = True
                        break
            elif constraint.operator == Operation.NEXT_TO:
                # Right must be next to left (either side)
                if house.number > 1:
                    prev_house = houses[house.number - 2]
                    if constraint.left in prev_house.get_feature_values(left_feature):
                        can_satisfy = True
                if house.number < house_count:
                    next_house = houses[house.number]
                    if constraint.left in next_house.get_feature_values(left_feature):
                        can_satisfy = True
            
            if not can_satisfy:
                try:
                    house.remove_feature(right_feature, constraint.right)
                    changes_made = True
                except ValueError:
                    pass
    
    return changes_made


def _find_feature(value: str, features: dict) -> str:
    for feature, values in features.items():
        if value in values:
            return feature
    raise ValueError(f"Feature not found for value: {value}")


def _parse_positional_constraint(constraint: str, features: dict) -> Constraint | None:
    """Parse 'X is (not) in the Nth house'."""
    position_match = re.search(
        r'is (not )?in the (\w+) house', constraint, re.IGNORECASE)
    if not position_match:
        return None

    is_not = position_match.group(1)  # Will be "not " or None
    position_word = position_match.group(2).lower()
    word_to_num = {
        'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
        'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10
    }

    if position_word not in word_to_num:
        raise ValueError(f"Could not parse position in: {constraint}")

    before_position = constraint[:position_match.start()].strip()
    value_text = re.sub(r'^the (person )?(who |whose |that )?',
                        '', before_position, flags=re.IGNORECASE)
    value = _extract_value_from_text(value_text, features)

    if not value:
        raise ValueError(f"Could not parse position in: {constraint}")

    position_num = str(word_to_num[position_word])
    operation = Operation.NOT_EQUAL if is_not else Operation.EQUAL
    return Constraint(value, position_num, operation, 0)


def _parse_directional_constraint(constraint: str) -> tuple | None:
    """Parse directional constraints and return operation and splits."""
    if constraint.find(" next to ") != -1:
        return (Operation.NEXT_TO, constraint.split(AND_SEPARATOR))
    elif constraint.find(" directly left of ") != -1:
        return (Operation.DIRECTLY_LEFT_OF, constraint.split(" directly left of "))
    elif constraint.find(" directly right of ") != -1:
        return (Operation.DIRECTLY_RIGHT_OF, constraint.split(" directly right of "))
    elif constraint.find(" left of ") != -1:
        return (Operation.LEFT_OF, constraint.split(" left of "))
    elif constraint.find(" right of ") != -1:
        return (Operation.RIGHT_OF, constraint.split(" right of "))
    return None


def _parse_between_constraint(constraint: str) -> tuple | None:
    """Parse 'between' constraints and return operation, splits, and distance."""
    between_match = re.search(
        r'there (are|is) (\w+) houses? between', constraint, re.IGNORECASE)
    if between_match:
        number_str = between_match.group(2)
        word_to_num = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }
        distance = word_to_num.get(number_str.lower())
        if distance is None:
            try:
                distance = int(number_str)
            except ValueError:
                distance = 0
        return (Operation.N_HOUSES_BETWEEN, constraint.split(AND_SEPARATOR), distance)

    elif constraint.find(" between ") != -1:
        return (Operation.BETWEEN, constraint.split(AND_SEPARATOR), 0)

    return None


def _parse_equality_constraint(constraint: str) -> tuple | None:
    """Parse simple equality/inequality constraints."""
    if constraint.find(" is not ") != -1:
        return (Operation.NOT_EQUAL, constraint.split(" is not "))
    elif constraint.find(" is ") != -1:
        return (Operation.EQUAL, constraint.split(" is "))
    return None


def _parse_constraints(features: dict, constraint: str) -> Constraint:
    """Parse constraint string and return Constraint object."""
    # Try positional constraint first
    try:
        parsed = _parse_positional_constraint(constraint, features)
        if parsed:
            return parsed
    except ValueError as e:
        raise ValueError(f"Could not parse position in: {constraint}") from e

    # Try directional constraint
    result = _parse_directional_constraint(constraint)
    if result:
        operation, splits = result
        left, right = _find_keyword_in_constraint(splits, features)
        return Constraint(left, right, operation, 0)

    # Try between constraint
    result = _parse_between_constraint(constraint)
    if result:
        operation, splits, distance = result
        left, right = _find_keyword_in_constraint(splits, features)
        return Constraint(left, right, operation, distance)

    # Try equality constraint
    result = _parse_equality_constraint(constraint)
    if result:
        operation, splits = result
        left, right = _find_keyword_in_constraint(splits, features)
        return Constraint(left, right, operation, 0)

    raise ValueError(f"Unknown constraint format: {constraint}")


def _find_value_in_words(words: list[str], all_values: list[str]) -> str:
    """Match word tokens to feature values."""
    for word in words:
        # Try exact word match
        for value in all_values:
            if value.lower() == word:
                return value

    return ""


def _extract_value_from_text(text: str, features: dict) -> str:
    """Extract feature value from text with case-insensitive matching."""
    text = text.strip().rstrip('.,;:!?')

    # Build list of all feature values
    all_values = []
    for values in features.values():
        all_values.extend(values)

    # Try exact full text match FIRST (before number extraction)
    for value in all_values:
        if value.lower() == text.lower():
            return value

    # Try to match against full feature values (for multi-word values like "iphone 13")
    # Sort by length (longest first) to prefer longer matches
    for value in sorted(all_values, key=len, reverse=True):
        if value.lower() in text.lower():
            return value

    # Try word-based matching (before number extraction)
    words = text.lower().split()
    result = _find_value_in_words(words, all_values)
    if result:
        return result

    # Only then check if there's a number embedded in the text (e.g., "house 1")
    # This is a fallback for positional constraints
    number_match = re.search(r'\d+', text)
    if number_match:
        return number_match.group()

    # Last resort: try partial substring matches
    best_match = ""
    for value in all_values:
        if value.lower() in text.lower() and len(value) > len(best_match):
            best_match = value

    return best_match


def _find_keyword_in_constraint(constraint: list[str], features: dict) -> Tuple[str, str]:
    """Extract two feature values from constraint parts."""
    if not constraint:
        raise ValueError("Could not find a matching keyword in constraint.")

    # Build list of all feature values
    all_feature_values = []
    for values in features.values():
        all_feature_values.extend(values)
    
    # Sort by length (longest first) to prefer longer matches and avoid substring issues
    sorted_values = sorted(all_feature_values, key=len, reverse=True)
    
    # Extract all possible values from all parts
    all_values = []
    for part in constraint:
        # Normalize part for matching (replace hyphens with spaces for comparison)
        part_normalized = part.lower().replace('-', ' ')
        
        # Try to find ALL values in this part, not just one
        # Use a set to track which positions have been matched to avoid overlaps
        matched_positions = set()
        
        for value in sorted_values:
            value_normalized = value.lower().replace('-', ' ')
            if value not in all_values:
                # Find all occurrences of this value in the part
                start_pos = 0
                while True:
                    pos = part_normalized.find(value_normalized, start_pos)
                    if pos == -1:
                        break
                    
                    # Check if this position range overlaps with already matched text
                    end_pos = pos + len(value_normalized)
                    overlap = False
                    for matched_start, matched_end in matched_positions:
                        if not (end_pos <= matched_start or pos >= matched_end):
                            overlap = True
                            break
                    
                    if not overlap:
                        # Check word boundaries to avoid substring matches
                        # e.g., "short" shouldn't match inside "very short"
                        before_ok = (pos == 0 or not part_normalized[pos-1].isalnum())
                        after_ok = (end_pos >= len(part_normalized) or not part_normalized[end_pos].isalnum())
                        
                        if before_ok and after_ok:
                            all_values.append(value)
                            matched_positions.add((pos, end_pos))
                            break
                    
                    start_pos = pos + 1

    if len(all_values) < 2:
        raise ValueError(f"Could not extract two values from: {constraint}")

    # Return first two unique values found
    return all_values[0], all_values[1]


def _parse_solution_from_csv(solution_str: str) -> dict:
    """Parse solution dict from CSV string representation."""
    # Extract header names from the first array
    header_match = re.search(
        r"'header':\s*array\(\[(.*?)\],\s*dtype", solution_str, re.DOTALL)
    if not header_match:
        raise ValueError("Could not find header in solution")

    header_str = header_match.group(1)
    header = [s.strip().strip("'\"") for s in header_str.split(',')]

    # Extract rows - find all array([...], dtype=object) patterns
    rows = []
    array_pattern = r"array\(\[(.*?)\],\s*dtype=object\)"

    # Find all matching arrays in the solution, skipping the first one (which is header)
    match_count = 0
    for match in re.finditer(array_pattern, solution_str, re.DOTALL):
        match_count += 1
        if match_count == 1:  # Skip header array
            continue

        row_data_str = match.group(1)
        row_data = [s.strip().strip("'\"") for s in row_data_str.split(',')]
        rows.append(row_data)

    # Convert to dict format
    result = {}
    for row_idx, row_data in enumerate(rows):
        result[row_idx] = {}
        for col_idx, col_name in enumerate(header):
            if col_name.lower() != 'house':
                result[row_idx][col_name.lower()] = row_data[col_idx]

    return result


if __name__ == "__main__":
    """Main execution: Test solver on CSP benchmark dataset.

    Loads puzzles from output.csv, solves each one, and compares against
    expected solutions. Tracks accuracy and prints diagnostics for failures.
    """
    df = pd.read_csv("output.csv", index_col='id')

    correct_count = 0
    total_count = 0

    for idx, row in df.iterrows():
        # Parse features and constraints from CSV (stored as Python literal strings)
        try:
            features = ast.literal_eval(row['features'])
            constraints = ast.literal_eval(row['constraints'])
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing row {idx}: {e}")
            continue

        total_count += 1

        solved_solution = solve_puzzle(features, constraints, row['houses'])

        # Parse the expected solution from CSV
        try:
            expected_solution = _parse_solution_from_csv(row['solution'])
            # NOTE: We will NOT rely on feature keys when comparing solutions.
            # Instead we compare by value -> house index mapping so that
            # differences in header/key naming won't affect correctness checks.
            # Keep the parsed expected_solution as-is (house_idx -> {key: value}).
        except Exception as e:
            print(f"Could not parse solution for {idx}: {e}")
            continue

        # Build a normalized dict representation of our solved solution
        solved_solution_dict = {}
        for house_idx, house in enumerate(solved_solution):
            solved_solution_dict[house_idx] = {}
            for feature, values in house.features.items():
                # If the solver determined a single value, use it; otherwise empty string
                if values and len(values) == 1:
                    # Normalize solved values (case-insensitive comparison)
                    solved_solution_dict[house_idx][feature] = str(
                        values[0]).strip().lower()
                else:
                    solved_solution_dict[house_idx][feature] = ""
        # Compare expected_solution (from CSV) with our solved_solution_dict
        # New strategy: ignore feature keys completely. Build maps of value -> house_idx
        # for both expected and solved, then ensure every expected value is placed
        # in the same house in the solved mapping.

        def _values_to_house_map(solution: dict) -> dict:
            """Convert solution dict {house_idx: {key: value}} to value->house_idx map.

            Empty or blank values are ignored.
            """
            mapping = {}
            for h_idx, house in solution.items():
                for _k, v in house.items():
                    if v is None:
                        continue
                    # Normalize values for robust comparison (strip + lower)
                    vs = str(v).strip().lower()
                    if vs == "":
                        continue
                    # If a value appears multiple times in expected, keep the first
                    # occurrence (shouldn't happen in well-formed puzzles).
                    if vs not in mapping:
                        mapping[vs] = h_idx
            return mapping

        matched = False
        try:
            expected_value_map = _values_to_house_map(expected_solution)
            solved_value_map = _values_to_house_map(solved_solution_dict)

            # All expected values must appear in the solved mapping and be in the same house
            matched = True
            for val, expected_house_idx in expected_value_map.items():
                solved_house_idx = solved_value_map.get(val)
                if solved_house_idx is None or solved_house_idx != expected_house_idx:
                    matched = False
                    break
        except Exception:
            matched = False

        # If they match, mark correct. If not, mark incorrect and print a
        # short diagnostic so the user can inspect what went wrong.
        is_correct = bool(matched)

        if not is_correct:
            try:
                print(
                    f"DISCREPANCY {idx}: expected vs solved (showing up to 5 houses)")
                # show up to first 5 houses to avoid huge dumps
                for i in range(min(5, max(len(expected_solution), len(solved_solution_dict)))):
                    exp = expected_solution.get(i, {})
                    sol = solved_solution_dict.get(i, {})
                    print(f" House {i}: EXPECTED={exp}  SOLVED={sol}")
            except Exception:
                # Don't let diagnostics break the run
                pass

        # Print result
        if is_correct:
            print(f"PASS {idx}")
            correct_count += 1
        else:
            print(f"FAIL {idx}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Correct: {correct_count}/{total_count}")
    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
        print(f"Accuracy: {accuracy:.1f}%")
