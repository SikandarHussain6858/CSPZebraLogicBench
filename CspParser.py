import pandas as pd
import json
import re
from pathlib import Path
from typing import Dict, List, Any

# Convert an Input into a lower case String without any spaces
def normalize(text: str) -> str:
    return str(text).strip().lower()

# Extract the Words inside backticks, because the Domains of each Variable is inside of a Backtick
def split_backtick_list(cell: str) -> List[str]:
    return [normalize(x) for x in re.findall(r"`([^`]+)`", cell)]

# Parser Class
class CSVToCSPParser:
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)

    # Every Row inside the CSV File is a specific CSP, so we will get them Row by row and convert all of them
    # into a JSON Object
    def parse_all(self) -> List[Dict[str, Any]]:
        df = pd.read_csv(self.csv_path)

        # This List will hold all the CSP
        csps: List[Dict[str, Any]] = []

        # Now we need to iterate through each Row, so that we can access each Constraint
        for _, row in df.iterrows():
            # As first, display the id of the specific Problem
            problem_name = row["id"]
            # Get the Amount of Houses
            house_count = int(row["houses"])
            # Get the amount of fetures
            feature_count = int(row["features"])
            # Output all the Variable Names and the Corresponding Domains
            variables: Dict[str, List[int]] = {}
            # Output all the Domains -> we need this when we want build the Constraints
            nodes: List[str] = []
            # Output all the Variable names
            categories: List[str] = []
            # Here we will have a List of all the Constraints
            constraints: List[Dict[str, Any]] = []

            # Get the Feature Columns from all the Columns
            feature_cols = [
                c for c in df.columns
                if c not in {"id", "houses", "features", "constraints"}
            ][:feature_count]

            for col in feature_cols:
                cell = row[col]

                if pd.isna(cell):
                    continue

                sub_cell = cell.split(":", 1)

                # Append the name of each Variable
                categories.append(sub_cell[0])
                # Add all the Domains of a specific Variable
                values = split_backtick_list(sub_cell[1])

                # Here we have to make sure that all the Values, which belongs to same feature, must be
                # assigned to different domains -> in our case through all the houses, otherwise a single house can
                # get multiple values from the same category.
                if values:
                    constraints.append({
                        "type": "ALL_DIFFERENT_CATEGORY",
                        "vars": values
                    })

                # Assign Domains to each Varaiable
                for v in values:
                    if v not in variables:
                        variables[v] = list(range(1, house_count + 1))
                        nodes.append(v)

            # Now we have to get the Constraint Texts
            clue_text = str(row["constraints"]).lower()
            # A Clue starts always in a new line and has a Number, so we have to extract them
            clues = re.split(r"\n|\r|\d+\. ", clue_text)

            # Now we can iterate through all the Clues
            for clue in clues:
                clue = clue.strip()
                # ignore empty clues
                if not clue:
                    continue

                # Here we will store all the Domains, that comes in a specific Clue.
                # The way we do this, on top we have a Nodes List, which contains all the Domains of a specific CSP.
                # What we will do is, we will check if our clue contains any String, which is in our Nodes List.
                names = [v for v in nodes if v in clue]

                # Based on the different matching constraints, we will add them to the Constrains List
                if "directly left" in clue and len(names) == 2:
                    constraints.append({"type": "DIRECTLY_LEFT", "vars": names})

                elif "somewhere to the left" in clue and len(names) == 2:
                    constraints.append({"type": "SOMEWHERE_TO_THE_LEFT", "vars": names})

                elif "next to" in clue and len(names) == 2:
                    constraints.append({"type": "NEXT_TO", "vars": names})

                elif "one house between" in clue and len(names) == 2:
                    constraints.append({"type": "DISTANCE", "vars": names, "param": 2})

                elif "two houses between" in clue and len(names) == 2:
                    constraints.append({"type": "DISTANCE", "vars": names, "param": 3})

                elif " is " in clue and len(names) == 2:
                    constraints.append({"type": "SAME_SLOT", "vars": names})

            # This our JSON Format -> Representation of each CSP
            csps.append({
                "metadata": {
                    "problem_name": problem_name,
                    "attribute_count": feature_count,
                    "house_count": house_count
                },
                "variables_and_domains": {
                    "categories": categories,
                    "domain_values": variables,
                    "nodes": nodes
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
        print("Json file has been created successfully")
