import pandas as pd

def clean_feature(cell):
    if pd.isna(cell):
        return cell
    cell_str = str(cell)
    colon_index = cell_str.find(':')
    if colon_index == -1:
        return cell_str  # no colon, return as-is
    words = cell_str[:colon_index].split()
    if not words:
        return cell_str
    last_word = words[-1]
    return last_word + cell_str[colon_index:]

def parquet_to_csv(parquet_path, csv_path):
    df = pd.read_parquet(parquet_path)

    # Drop 'created_at' column
    if 'created_at' in df.columns:
        df = df.drop(columns=['created_at'])

    # Parse size "3*10" → houses/features
    if 'size' in df.columns:
        size_split = df['size'].astype(str).str.split('*', expand=True)
        df['houses'] = size_split[0].astype(int)
        df['features'] = size_split[1].astype(int)
        df = df.drop(columns=['size'])

    # Clean puzzle into features + constraints
    if 'puzzle' in df.columns:
        # Keep everything after first dash including dash
        df['puzzle'] = df['puzzle'].astype(str).str.replace(r'^[^-]*', '', regex=True)
        puzzle_clean = df['puzzle'].str.lstrip('-')

        # Extract constraints (after final #)
        df['constraints'] = puzzle_clean.str.split('#').str[-1]

        # Everything before # are features
        before_constraints = puzzle_clean.str.split('#').str[0]
        feature_parts = before_constraints.str.split('-')

        # Limit to max 6 features
        max_features = 6
        for i in range(max_features):
            df[f'feature_{i+1}'] = feature_parts.apply(lambda x: x[i] if i < len(x) else None)

        # Clean feature_* columns
        feature_cols = [f'feature_{i+1}' for i in range(max_features)]
        for col in feature_cols:
            df[col] = df[col].apply(clean_feature)

    # Drop the original puzzle column
    if 'puzzle' in df.columns:
        df = df.drop(columns=['puzzle'])

    # Reorder columns: solution at the end
    cols = [c for c in df.columns if c != 'solution']
    if 'solution' in df.columns:
        cols.append('solution')
    df = df[cols]

    # Write CSV
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    parquet_file = "test-00000-of-00001.parquet"
    csv_file = "output.csv"

    parquet_to_csv(parquet_file, csv_file)
    print("Conversion complete.")
