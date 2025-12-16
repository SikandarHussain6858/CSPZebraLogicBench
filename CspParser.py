import pandas as pd
import json
import ast
from pathlib import Path
from typing import Dict, List, Any

# Parser Class
class CSVToCSPParser:

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)

    # Every Row inside the CSV File is a specific CSP, so we will get them Row by row and convert all of them
    # into a JSON Object
    def parse_all(self) -> List[Dict[str, Any]]:
        # Read CSV into DataFrame
        df = pd.read_csv(self.csv_path)

        # This List will hold all the CSP
        csps: List[Dict[str, Any]] = []

        # Now we need to iterate through each Row, so that we can access each Constraint
        for _, row in df.iterrows():
            # As first, display the id of the specific Problem
            problem_id = row["id"]
            # Get the Amount of Houses
            house_count = int(row["houses"])
            # Convert all the Features into a List and store them
            features: Dict[str, List[str]] = ast.literal_eval(row["features"])
            # Output all the Variable Names and the Corresponding Domains
            variables: Dict[str, List[int]] = {}
            # Output all the Domains -> we need this when we want build the Constraints
            nodes: List[str] = []
            # Output all the Variable names
            categories: List[str] = []
            # Here we will have a List of all the Constraints
            constraints: List[Dict[str, Any]] = []

            # Iterate through all the features and Categories
            for category, values in features.items():
                # Store Category name
                categories.append(category)

                # Normalizing Values -> lower case to match with clues
                normalized_values = [v.lower() for v in values]

                # Here we have to make sure that all the Values, which belongs to same feature, must be
                # assigned to different domains -> in our case through all the houses, otherwise a single house can
                # get multiple values from the same category.
                constraints.append({
                    "type": "ALL_DIFFERENT_CATEGORY",
                    "vars": normalized_values
                })

                # Assign domains to each variable
                for v in normalized_values:
                    if v not in variables:
                        variables[v] = list(range(1, house_count + 1))
                        nodes.append(v)

            # Convert all the Clues into a List of Strings.
            clues: List[str] = ast.literal_eval(row["constraints"])

            # Now we can iterate through all the Clues
            for clue in clues:
                clue = clue.lower()

                # Here we will store all the Domains, that comes in a specific Clue.
                # The way we do this, on top we have a Nodes List, which contains all the Domains of a specific CSP.
                # What we will do is, we will check if our clue contains any String, which is in our Nodes List.
                names = [v for v in nodes if v in clue]

                # Based on the different matching constraints, we will add them to the Constrains List
                if "directly left" in clue and len(names) == 2:
                    constraints.append({
                        "type": "DIRECTLY_LEFT",
                        "vars": names
                    })

                elif "somewhere to the left" in clue and len(names) == 2:
                    constraints.append({
                        "type": "SOMEWHERE_TO_THE_LEFT",
                        "vars": names
                    })

                elif "next to" in clue and len(names) == 2:
                    constraints.append({
                        "type": "NEXT_TO",
                        "vars": names
                    })

                elif "one house between" in clue and len(names) == 2:
                    constraints.append({
                        "type": "DISTANCE",
                        "vars": names,
                        "param": 2
                    })

                elif "two houses between" in clue and len(names) == 2:
                    constraints.append({
                        "type": "DISTANCE",
                        "vars": names,
                        "param": 3
                    })

                elif " is " in clue and len(names) == 2:
                    constraints.append({
                        "type": "SAME_SLOT",
                        "vars": names
                    })

            # This our JSON Format -> Representation of each CSP
            csps.append({
                "metadata": {
                    "problem_name": problem_id,
                    "house_count": house_count,
                    "attribute_count": len(categories)
                },
                "variables_and_domains": {
                    "categories": categories,
                    "nodes": nodes,
                    "domain_values": variables
                },
                "constraints": {
                    "puzzle_rules": constraints
                }
            })

        return csps

if __name__ == "__main__":
    parser = CSVToCSPParser("output.csv")
    all_csps = parser.parse_all()

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(all_csps, f, indent=4)
        print("JSON File has been created successfully")
