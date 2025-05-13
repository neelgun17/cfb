import csv
import os
import pandas as pd

# Helper function to process a single CSV file
def _process_single_csv_file(input_file_path, column_name, column_value):
    """
    Filters a single CSV file based on a specific column and value.
    The output file is dynamically named and placed in a 'qb_only' subdirectory
    relative to the input file's directory, prefixed with 'qb_'.
    Also prints the header of the input CSV.
    """
    print(f"Processing file: {input_file_path}")
    filtered_data = []
    try:
        # Dynamically construct the output path
        input_dir = os.path.dirname(input_file_path)
        input_filename = os.path.basename(input_file_path)

        output_subdir = "qb_only" 
        output_dir_path = os.path.join(input_dir, output_subdir)

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir_path, exist_ok=True)

        output_filename = f"qb_{input_filename}"
        output_path = os.path.join(output_dir_path, output_filename)

        # Open the input CSV file
        with open(input_file_path, mode='r', newline='') as infile:
            csv_reader = csv.reader(infile)

            # Read the header row
            header = next(csv_reader)
            # print(f"  CSV Header for {input_filename}: {header}")
            filtered_data.append(header)

            try:
                filter_column_index = header.index(column_name)
            except ValueError:
                print(f"  Error: Column '{column_name}' not found in {input_filename}. Skipping this file.")
                return 
            # Iterate over the rest of the rows and filter
            for row in csv_reader:
                # Add a check for row length before accessing by index
                if len(row) > filter_column_index and row[filter_column_index] == column_value:
                    filtered_data.append(row)
        # Write the filtered data to a new CSV file
        if len(filtered_data) > 1: 
            with open(output_path, mode='w', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerows(filtered_data)
            print(f"  Filtered data for {input_filename} saved to {output_path}")
        elif len(filtered_data) == 1:
            print(f"  Only header found or no rows matched filter for {input_filename}. Output file '{output_path}' not created with data rows.")
        else: 
            print(f"  No data (not even header) processed for {input_filename}. Output file not created.")

    except FileNotFoundError:
        print(f"  Error: The file {input_file_path} was not found.")
    except StopIteration:
        print(f"  Error: The CSV file {input_file_path} is empty or has no header row.")
    except Exception as e:
        print(f"  An error occurred while processing {input_file_path}: {e}")

# Main function that handles file or directory
def filter_csv_by_position(path_or_dir, column_name, column_value):
    """
    Filters CSV file(s) based on a specific column and value.
    If path_or_dir is a file, it processes that file.
    If path_or_dir is a directory, it processes all .csv files in that directory.
    """
    if os.path.isfile(path_or_dir):
        if path_or_dir.lower().endswith(".csv"):
            _process_single_csv_file(path_or_dir, column_name, column_value)
        else:
            print(f"Skipping non-CSV file: {path_or_dir}")
    elif os.path.isdir(path_or_dir):
        print(f"Processing directory: {path_or_dir}")
        for item_name in os.listdir(path_or_dir):
            item_path = os.path.join(path_or_dir, item_name)
            if os.path.isfile(item_path) and item_name.lower().endswith(".csv"):
                _process_single_csv_file(item_path, column_name, column_value)
    else:
        print(f"Error: Path '{path_or_dir}' is not a valid file or directory.")

def extract_player_passing():
    """
    Parses qb_passing_summary.csv, drops irrelevant columns, adds new calculated columns
    td_int_ratio, and saves it to processed_data/qb_player_summary.csv.
    """
    input_qb_passing_summary_path = "/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/raw_data/qb_only/qb_passing_summary.csv"

    processed_data_dir = "/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/processed_data"
    output_summary_filename = "qb_player_summary.csv"
    output_summary_path = os.path.join(processed_data_dir, output_summary_filename)

    os.makedirs(processed_data_dir, exist_ok=True)

    columns_to_drop = ['bats', 'thrown_aways', 'declined_penalties', "hit_as_threw", 
                       "spikes", "penalties", "pressure_to_sack_rate",
                       "position", "franchise_id","aimed_passes","drop_rate",
                       "completions","avg_time_to_throw", "grades_hands_fumble"] # Example columns

    # --- TD/INT Ratio Column Configuration ---
    td_int_ratio = 'td_int_ratio'
    td_name = 'touchdowns' 
    int_name = 'interceptions'
    # --- End TD/INT Ratio Column Configuration ---

    print(f"\nCreating player summary from: {input_qb_passing_summary_path}")
    print(f"Outputting to: {output_summary_path}")
    print(f"Dropping columns: {columns_to_drop}")
    print(f"Adding new column: '{td_int_ratio}' based on '{td_name}' and '{int_name}'")

    processed_rows = []

    try:
        with open(input_qb_passing_summary_path, mode='r', newline='') as infile:
            reader = csv.reader(infile)
            original_header = next(reader)
            
            # --- Determine indices for calculation columns from original_header ---
            td_idx, int_idx = -1, -1

            try:
                td_idx = original_header.index(td_name)
            except ValueError:
                print(f"  Warning: Column '{td_name}' not found. Cannot calculate '{td_int_ratio}'.")
            try:
                int_idx = original_header.index(int_name)
            except ValueError:
                print(f"  Warning: Column '{int_name}' not found. Cannot calculate '{td_int_ratio}'.")
            # --- End Determine indices ---

            # Determine which columns to keep and build the new header
            indices_to_keep = []
            final_header_parts = []
            for i, col_name in enumerate(original_header):
                if col_name not in columns_to_drop:
                    indices_to_keep.append(i)
                    final_header_parts.append(col_name)
            
            if not final_header_parts:
                print("  Error: No columns left after dropping. Check 'columns_to_drop'.")
                return None

            # Add new column names to the header
            final_header = final_header_parts + [td_int_ratio]
            processed_rows.append(final_header)
            print()
            print(f"  Final output header Passing: {final_header}")
            print(len(final_header))

            # Process data rows
            for original_row in reader:
                kept_values_row = []
                for idx in indices_to_keep:
                    if idx < len(original_row):
                        kept_values_row.append(original_row[idx])
                    else:
                        kept_values_row.append('')

                # --- Calculate TD/INT Ratio ---
                ratio_value = "0.0" 
                if td_idx != -1 and int_idx != -1:
                    try:
                        tds = int(original_row[td_idx])
                        ints = int(original_row[int_idx])
                        if ints == 0:
                            ratio_value = str(tds) if tds > 0 else "0"
                        else:
                            ratio_value = str(round(tds / ints, 2))
                    except (ValueError, IndexError):
                        ratio_value = "NAN"
                # --- End Calculate TD/INT Ratio ---

                # Combine kept values with new calculated values
                final_row_for_output = kept_values_row + [ratio_value]
                
                if len(final_row_for_output) == len(final_header):
                    processed_rows.append(final_row_for_output)
                else:
                    print(f"  Warning: Skipping malformed row. Expected {len(final_header)} columns, got {len(final_row_for_output)}. Original row: {original_row}")


        if len(processed_rows) > 1: 
            with open(output_summary_path, mode='w', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerows(processed_rows)
            print(f"  Player summary with new columns saved to {output_summary_path}")
            return output_summary_path
        else:
            print(f"  No data (or only header) to write for player summary. File not created or is empty.")
            return None
    except FileNotFoundError:
        print(f"  Error: Input file for summary not found: {input_qb_passing_summary_path}")
        print(f"  Please ensure 'filter_csv_by_position' has run successfully and created this file.")
        return None
    except StopIteration:
        print(f"  Error: The input summary file {input_qb_passing_summary_path} is empty or has no header.")
        return None
    except Exception as e:
        print(f"  An error occurred during player summary creation: {e}")
        return None
    
def extract_player_rushing():
    """
    Parses qb_passing_rushing.csv, drops irrelevant columns
    and saves it to processed_data/qb_player_summary.csv.
    """
    input_qb_rushing_summary_path = "/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/raw_data/qb_only/qb_rushing_summary.csv"
    output_summary_path =  "/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/processed_data/qb_player_rushing_summary.csv"
    processed_rows = []
    columns_to_drop = ["position","team_name","player_game_count",
                       "declined_penalties", "drops","penalties","rec_yards","receptions","routes",
                       "zone_attempts", "grades_pass_block", "gap_attempts", "elu_recv_mtf",
                       "targets", "grades_offense_penalty", "grades_pass_route", "def_gen_pressures",
                       "grades_offense","grades_run_block","yco_attempt","elu_rush_mtf","grades_hands_fumble",
                       "franchise_id","scrambles","grades_pass", "grades_run"]
    try:
        with open(input_qb_rushing_summary_path, mode='r', newline='') as infile:
            reader = csv.reader(infile)
            original_header = next(reader)
            indices_to_keep = []
            final_header = []
            for i, col_name in enumerate(original_header):
                if col_name not in columns_to_drop:
                    indices_to_keep.append(i)
                    final_header.append(col_name)
            
            if not final_header:
                print("  Error: No columns left after dropping. Check 'columns_to_drop'.")
                return None
            processed_rows.append(final_header)
            # print(final_header)
            for original_row in reader:
                kept_values_row = []
                for idx in indices_to_keep:
                    if idx < len(original_row):
                        kept_values_row.append(original_row[idx])
                    else:
                        kept_values_row.append('')
                processed_rows.append(kept_values_row)
            # Rename column in final_header and update the first row of processed_rows
            def rename(name, end):
                if name in final_header:
                    col_index = final_header.index(name)
                    final_header[col_index] = name +"_" + end
                    processed_rows[0][col_index] = name+ "_" + end
            rename("attempts", "designed_rushing")
            rename("touchdowns","rushing")
            rename("yards", "rushing")
            rename("ypa","rushing")
            rename("first_downs","rushing")
            print()
            print(f"  Final output header rushing: {final_header}")
            print(len(final_header))
            with open(output_summary_path, mode='w', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerows(processed_rows)
            return processed_rows

    except Exception as e:
        print(f"  An error occurred during player summary creation: {e}")
        return None   
         
def extract_concept_fields():
    summary_path = "PFF_Data/raw_data/qb_only/qb_passing_concept.csv"
    output_summary_path =  "/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/processed_data/qb_player_concept_summary.csv"
    recommended_fields = [
    # General Info
    "player", "player_id", "position", "team_name", "player_game_count",

    # Overall efficiency
    "comp_pct_diff", "ypa_diff",

    # No Screen Splits (core QB metrics)
    "no_screen_accuracy_percent",
    "no_screen_attempts",
    "no_screen_avg_depth_of_target",
    "no_screen_avg_time_to_throw",
    "no_screen_big_time_throws",
    "no_screen_btt_rate",
    "no_screen_completion_percent",
    "no_screen_def_gen_pressures",
    "no_screen_drop_rate",
    "no_screen_interceptions",
    "no_screen_pressure_to_sack_rate",
    "no_screen_qb_rating",
    "no_screen_sack_percent",
    "no_screen_touchdowns",
    "no_screen_turnover_worthy_plays",
    "no_screen_twp_rate",
    "no_screen_ypa",

    # Non-Play Action Splits
    "npa_accuracy_percent",
    "npa_attempts",
    "npa_avg_depth_of_target",
    "npa_avg_time_to_throw",
    "npa_big_time_throws",
    "npa_btt_rate",
    "npa_completion_percent",
    "npa_def_gen_pressures",
    "npa_drop_rate",
    "npa_interceptions",
    "npa_pressure_to_sack_rate",
    "npa_qb_rating",
    "npa_sack_percent",
    "npa_touchdowns",
    "npa_turnover_worthy_plays",
    "npa_twp_rate",
    "npa_ypa",

    # Play Action Splits
    "pa_accuracy_percent",
    "pa_attempts",
    "pa_avg_depth_of_target",
    "pa_avg_time_to_throw",
    "pa_big_time_throws",
    "pa_btt_rate",
    "pa_completion_percent",
    "pa_def_gen_pressures",
    "pa_drop_rate",
    "pa_interceptions",
    "pa_pressure_to_sack_rate",
    "pa_qb_rating",
    "pa_sack_percent",
    "pa_touchdowns",
    "pa_turnover_worthy_plays",
    "pa_twp_rate",
    "pa_ypa",

    # Screen Splits (for scheme fit)
    "screen_accuracy_percent",
    "screen_attempts",
    "screen_avg_depth_of_target",
    "screen_avg_time_to_throw",
    "screen_big_time_throws",
    "screen_btt_rate",
    "screen_completion_percent",
    "screen_def_gen_pressures",
    "screen_drop_rate",
    "screen_interceptions",
    "screen_pressure_to_sack_rate",
    "screen_qb_rating",
    "screen_sack_percent",
    "screen_touchdowns",
    "screen_turnover_worthy_plays",
    "screen_twp_rate",
    "screen_ypa",
]

    print(len(recommended_fields))
    # Always use the actual processed summary path for reading
    try:
        with open(summary_path, mode='r', newline='') as infile:
            reader = csv.reader(infile)
            original_header = next(reader)
            # Set final_header to recommended_fields, regardless of presence in original_header
            final_header = recommended_fields
            orig_idx = {col: i for i, col in enumerate(original_header)}
            filtered_rows = [final_header]
            for row in reader:
                filtered_row = []
                for f in recommended_fields:
                    if f in orig_idx:
                        idx = orig_idx[f]
                        filtered_row.append(row[idx] if idx < len(row) else '')
                    else:
                        filtered_row.append('')
                filtered_rows.append(filtered_row)
        with open(output_summary_path, mode='w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(filtered_rows)
            print()
            print(f"  Final output header concept: {final_header}")
            print(len(final_header))
        return output_summary_path
    except Exception as e:
        print(f"  An error occurred during player concept summary creation: {e}")
        return None
 
def extract_depth_fields():
    summary_path = "PFF_Data/raw_data/qb_only/qb_passing_depth.csv"
    output_summary_path =  "/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/processed_data/qb_player_depth_summary.csv"
    splits = [
    "deep", "medium", "short", "behind_los",
    # "left_deep", "left_medium", "left_short", "left_behind_los",
    # "center_deep", "center_medium", "center_short", "center_behind_los",
    # "right_deep", "right_medium", "right_short", "right_behind_los"
    ]

    core_fields = [
        "player", "player_id", "position", "team_name", "player_game_count", "base_attempts", "base_dropbacks"
    ]

    split_fields = [
        "{split}_attempts",
        "{split}_completion_percent",
        "{split}_accuracy_percent",
        "{split}_ypa",
        "{split}_big_time_throws",
        "{split}_btt_rate",
        "{split}_turnover_worthy_plays",
        "{split}_twp_rate",
        "{split}_interceptions",
        "{split}_touchdowns",
        "{split}_avg_depth_of_target",
        "{split}_avg_time_to_throw",
        "{split}_def_gen_pressures",
        "{split}_pressure_to_sack_rate",
        "{split}_sack_percent",
        "{split}_qb_rating"
    ]

    recommended_fields = core_fields + [
        field.format(split=split) for split in splits for field in split_fields
    ]

    print(len(recommended_fields))
    # Always use the actual processed summary path for reading
    try:
        with open(summary_path, mode='r', newline='') as infile:
            reader = csv.reader(infile)
            original_header = next(reader)
            # Set final_header to recommended_fields, regardless of presence in original_header
            final_header = recommended_fields
            orig_idx = {col: i for i, col in enumerate(original_header)}
            filtered_rows = [final_header]
            for row in reader:
                filtered_row = []
                for f in recommended_fields:
                    if f in orig_idx:
                        idx = orig_idx[f]
                        filtered_row.append(row[idx] if idx < len(row) else '')
                    else:
                        filtered_row.append('')
                filtered_rows.append(filtered_row)
        with open(output_summary_path, mode='w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(filtered_rows)
            print()
            print(f"  Final output header depth: {final_header}")
            print(len(final_header))
        return output_summary_path
    except Exception as e:
        print(f"  An error occurred during player concept summary creation: {e}")
        return None

def extract_pressure_fields():
    summary_path = "PFF_Data/raw_data/qb_only/qb_passing_pressure.csv"
    output_summary_path =  "/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/processed_data/qb_player_pressure_summary.csv"
    splits = ["blitz_", "no_blitz_", "pressure_", "no_pressure_"]

    core_fields = [
        "player", "player_id", "position", "team_name", "player_game_count", "base_dropbacks"
    ]

    split_fields = [
        "{split}attempts",
        "{split}completion_percent",
        "{split}accuracy_percent",
        "{split}ypa",
        "{split}big_time_throws",
        "{split}btt_rate",
        "{split}turnover_worthy_plays",
        "{split}twp_rate",
        "{split}interceptions",
        "{split}touchdowns",
        "{split}avg_depth_of_target",
        "{split}avg_time_to_throw",
        "{split}def_gen_pressures",
        "{split}pressure_to_sack_rate",
        "{split}sack_percent",
        "{split}qb_rating"
    ]

    recommended_fields = core_fields + [
        field.format(split=split) for split in splits for field in split_fields
    ]


    print(len(recommended_fields))
    # Always use the actual processed summary path for reading
    try:
        with open(summary_path, mode='r', newline='') as infile:
            reader = csv.reader(infile)
            original_header = next(reader)
            # Set final_header to recommended_fields, regardless of presence in original_header
            final_header = recommended_fields
            orig_idx = {col: i for i, col in enumerate(original_header)}
            filtered_rows = [final_header]
            for row in reader:
                filtered_row = []
                for f in recommended_fields:
                    if f in orig_idx:
                        idx = orig_idx[f]
                        filtered_row.append(row[idx] if idx < len(row) else '')
                    else:
                        filtered_row.append('')
                filtered_rows.append(filtered_row)
        with open(output_summary_path, mode='w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(filtered_rows)
            print()
            print(f"  Final output header pressure: {final_header}")
            print(len(final_header))
        return output_summary_path
    except Exception as e:
        print(f"  An error occurred during player concept summary creation: {e}")
        return None

def extract_pocket_time_fields():
    summary_path = "PFF_Data/raw_data/qb_only/qb_time_in_pocket.csv"
    output_summary_path =  "/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/processed_data/qb_player_pocket_time_summary.csv"
    splits = ["less_", "more_"]

    core_fields = [
        "player", "player_id", "position", "team_name", "player_game_count", "avg_time_to_throw"
    ]

    split_fields = [
        "{split}attempts",
        "{split}completion_percent",
        "{split}accuracy_percent",
        "{split}ypa",
        "{split}big_time_throws",
        "{split}btt_rate",
        "{split}turnover_worthy_plays",
        "{split}twp_rate",
        "{split}interceptions",
        "{split}touchdowns",
        "{split}avg_depth_of_target",
        "{split}avg_time_to_throw",
        "{split}def_gen_pressures",
        "{split}pressure_to_sack_rate",
        "{split}sack_percent",
        "{split}qb_rating"
    ]

    recommended_fields = core_fields + [
        field.format(split=split) for split in splits for field in split_fields
    ]



    print(len(recommended_fields))
    # Always use the actual processed summary path for reading
    try:
        with open(summary_path, mode='r', newline='') as infile:
            reader = csv.reader(infile)
            original_header = next(reader)
            # Set final_header to recommended_fields, regardless of presence in original_header
            final_header = recommended_fields
            orig_idx = {col: i for i, col in enumerate(original_header)}
            filtered_rows = [final_header]
            for row in reader:
                filtered_row = []
                for f in recommended_fields:
                    if f in orig_idx:
                        idx = orig_idx[f]
                        filtered_row.append(row[idx] if idx < len(row) else '')
                    else:
                        filtered_row.append('')
                filtered_rows.append(filtered_row)
        with open(output_summary_path, mode='w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(filtered_rows)
            print()
            print(f"  Final output header time: {final_header}")
            print(len(final_header))
        return output_summary_path
    except Exception as e:
        print(f"  An error occurred during player concept summary creation: {e}")
        return None

# Merge concept fields into original summary
def merge_concept_to_summary():
    """
    Reads all processed summary CSVs, merges them on ["player_id", "player"], and writes the merged result.
    """
    base_dir = "/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/processed_data/"
    csv_files = [
        "qb_player_summary.csv",
        "qb_player_rushing_summary.csv",
        "qb_player_concept_summary.csv",
        "qb_player_depth_summary.csv",
        "qb_player_pressure_summary.csv",
        "qb_player_pocket_time_summary.csv"
    ]
    # Read all CSVs into dataframes
    dfs = []
    for fname in csv_files:
        fpath = os.path.join(base_dir, fname)
        try:
            df = pd.read_csv(fpath)
            dfs.append(df)
        except Exception as e:
            print(f"  Error reading {fpath}: {e}")
            return
    # Sequentially merge on ["player_id", "player"] with outer join
    merged_df = dfs[0]
    for df in dfs[1:]:
        df = df.drop(columns=["team_name", "position", "player_game_count"], errors="ignore")
        merged_df = pd.merge(merged_df, df, on=["player_id", "player"], how="left")

   
    output_path = os.path.join(base_dir, "qb_player_merged_summary.csv")

    merged_df = merged_df[merged_df["player_game_count"].astype(int) >= 5]
    merged_df.to_csv(output_path, index=False) # Save the filtered DataFrame
    print(f"Successfully merged all summaries. Output written to: {output_path}")
    return merged_df

def reduce_columns(df):
    base_dir = "/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/processed_data/"
    output_path = os.path.join(base_dir, "qb_player_merged_summary.csv")
     # --- Insert adjusted_attempt_rate calculation here ---
    try:
        df["tot_attempts"] = round((df["attempts"]) + (df["attempts_designed_rushing"]), 2)
    except Exception as e:
        print(f"  Error computing 'tot_attempts': {e}")
    try:
        df["tot_rushing_attempts"]  = round((df["attempts_designed_rushing"] + df["scrambles"]),2)
    except Exception as e:
        print(f"  Error computing 'adjusted_attempt_rate': {e}")
    try:
        df["designed_run_rate"] = round((df["attempts_designed_rushing"]) / (df["tot_attempts"]), 2)
    except Exception as e:
        print(f"  Error computing 'adjusted_attempt_rate': {e}")
    try:
        df["scramble_rate"] = round((df["scrambles"]) / (df["dropbacks"]), 2)
    except Exception as e:
        print(f"  Error computing 'adjusted_attempt_rate': {e}")
    try:
        df["pa_rate"] = round((df["pa_attempts"]) / (df["attempts"]), 2)
    except Exception as e:
        print(f"  Error computing 'pa_rate': {e}")
    try:
        df["screen_rate"] = round((df["screen_attempts"]) / (df["attempts"]), 2)
    except Exception as e:
        print(f"  Error computing 'screen_rate': {e}")     
    try:
        df["deep_attempt_rate"] = round((df["deep_attempts"]) / (df["attempts"]), 2)
    except Exception as e:
        print(f"  Error computing 'deep_attempt_rate': {e}")
    try:
        df["medium_attempt_rate"] = round((df["medium_attempts"]) / (df["attempts"]), 2)
    except Exception as e:
        print(f"  Error computing 'medium_attempt_rate': {e}")    
    try:
        df["short_attempt_rate"] = round((df["short_attempts"]) / (df["attempts"]), 2)
    except Exception as e:
        print(f"  Error computing 'short_attempt_rate': {e}")
    try:
        df["behind_los_attempt_rate"] = round((df["behind_los_attempts"]) / (df["attempts"]), 2)
    except Exception as e:
        print(f"  Error computing 'behind_los_attempt_rate': {e}")    
    try:
        df["pressure_rate"] = round((df["def_gen_pressures"]) / (df["dropbacks"]), 2)
    except Exception as e:
        print(f"  Error computing 'pressure_rate': {e}")
    try:
        df["blitz_rate"] = round((df["blitz_attempts"]) / (df["dropbacks"]), 2)
    except Exception as e:
        print(f"  Error computing 'blitz_rate': {e}")            
    try:
        df["quick_throw_rate"] = round((df["less_attempts"]) / (df["attempts"]), 2)
    except Exception as e:
        print(f"  Error computing 'quick_throw_rate': {e}")
    try:
        df["long_throw_rate"] = round((df["more_attempts"]) / (df["attempts"]), 2)
    except Exception as e:
        print(f"  Error computing 'long_throw_rate': {e}")        
    try:
        df["designed_YPA"] = round((df["designed_yards"]) / (df["attempts_designed_rushing"]), 2)
    except Exception as e:
        print(f"  Error computing 'designed_YPA': {e}")          
    try:
        df["pct_total_yards_rushing"] = round((df["yards_rushing"]) / (df["yards_rushing"] + df["yards"]), 2)
    except Exception as e:
        print(f"  Error computing 'pct_total_yards_rushing': {e}")   
    try:
        df["qb_designed_run_rate_of_all_plays"] = round((df["attempts_designed_rushing"]) / (df["tot_attempts"]), 2)
    except Exception as e:
        print(f"  Error computing 'qb_designed_run_rate_of_all_plays': {e}")   
    try:
        df["rush_attempts_per_game"] = round((df["tot_rushing_attempts"]) / (df["player_game_count"]), 2)
    except Exception as e:
        print(f"  Error computing 'rush_attempts_per_game': {e}")     
    try:
        df["YAC_per_rush_attempt"] = round((df["yards_after_contact"]) / (df["tot_rushing_attempts"]), 2)
    except Exception as e:
        print(f"  Error computing 'YAC_per_rush_attempt': {e}")    
    try:
        df["qb_rush_attempt_rate"] = round((df["tot_rushing_attempts"]) / (df["dropbacks"] + (df["tot_rushing_attempts"])), 2)
    except Exception as e:
        print(f"  Error computing 'qb_rush_attempt_rate': {e}")                     
    # Drop the original columns after calculation
    df.drop(columns=["scrambles"], inplace=True, errors="ignore")
    # --- End insertion ---
    df.to_csv(output_path, index=False) # Save the filtered DataFrame
    print(f"Successfully merged all summaries. Output written to: {output_path}")
    print(df.shape)
    df = df[df["dropbacks"].astype(int) >= 150]
    print(df.shape)

def main():
    # Define filter criteria
    filter_column_name = "position"
    filter_column_value = "QB"

    input_file_for_filtering = "/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/raw_data/passing_summary.csv"
    filter_csv_by_position(input_file_for_filtering, filter_column_name, filter_column_value)

    # Now, create the player summary from the filtered file
    summary_path = extract_player_passing()
    rushing_path = extract_player_rushing()
    concept_path = extract_concept_fields()
    depth_path = extract_depth_fields()
    pressure_path = extract_pressure_fields()
    pocket_time_path = extract_pocket_time_fields()
    merged_df = merge_concept_to_summary()
    reduce_columns(merged_df)

if __name__ == "__main__":
    main()
