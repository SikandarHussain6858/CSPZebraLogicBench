import pandas as pd
import re


def _parse_features(puzzle: str) -> dict:
    """Extract features from puzzle text."""
    dirty_features = [line for line in puzzle.split('\n')
                      if line.strip().startswith('- ')]

    return {
        item.split(":")[0].split(" ")[-1]: item.split(":")[1].strip().replace("`", "").split(", ")
        for item in dirty_features
    }


def _parse_constraints(puzzle: str) -> list[str]:
    """Extract constraints from puzzle text."""
    dirty_constraints = [line for line in puzzle.split('\n')
                         if re.match(r'^\d+\.\s', line.strip())]

    return [item.split(". ")[-1].strip() for item in dirty_constraints]


def parquet_to_csv(parquet_path: str, csv_path: str) -> None:
    df = pd.read_parquet(parquet_path)
    df.set_index('id', inplace=True)

    # Extract the number of houses from the 'size' column (split on '*')
    # and parse features and constraints from 'puzzle' column
    output = pd.DataFrame({
        'houses': df['size'].astype(str).str.split('*').str[0].astype(int),
        'features': df['puzzle'].apply(_parse_features),
        'constraints': df['puzzle'].apply(_parse_constraints),
    }, index=df.index)

    # Add solution column else set it to None
    output['solution'] = df['solution'] if 'solution' in df.columns else None

    # Write CSV
    output.to_csv(csv_path, index=True, index_label='id')


if __name__ == "__main__":
    parquet_file = "test-00000-of-00001.parquet"
    csv_file = "output.csv"

    parquet_to_csv(parquet_file, csv_file)
    print("Conversion complete.")
