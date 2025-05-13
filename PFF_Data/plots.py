import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration ---
# <<< IMPORTANT: Replace with the actual path to your CSV file >>>
FILE_PATH = '/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/processed_data/qb_player_merged_summary.csv'

# <<< Confirm this is the correct column name in your CSV >>>
DROPBACKS_COLUMN = 'tot_attempts'

# --- Load the Data ---
try:
    df = pd.read_csv(FILE_PATH)
    print(f"Successfully loaded data from: {FILE_PATH}")
    print(f"DataFrame shape: {df.shape}")
    # Optional: Print columns to verify name
    # print(f"Columns: {df.columns.tolist()}")
except FileNotFoundError:
    print(f"Error: File not found at '{FILE_PATH}'. Please check the path.")
    exit()
except Exception as e:
    print(f"An error occurred loading the CSV: {e}")
    exit()

df = df[df["tot_attempts"].astype(int) >= 175]
print(df.shape)

# --- Validate Column Exists ---
if DROPBACKS_COLUMN not in df.columns:
    print(f"Error: Column '{DROPBACKS_COLUMN}' not found in the CSV.")
    print(f"Available columns are: {df.columns.tolist()}")
    exit()

# --- Data Cleaning: Ensure Numeric and Handle Missing ---
# Convert to numeric, coercing errors will turn non-numeric values into NaN
df[DROPBACKS_COLUMN] = pd.to_numeric(df[DROPBACKS_COLUMN], errors='coerce')

# Drop rows where dropbacks are missing (NaN) as they can't be plotted
initial_rows = len(df)
df_filtered = df.dropna(subset=[DROPBACKS_COLUMN])
removed_rows = initial_rows - len(df_filtered)
if removed_rows > 0:
    print(f"Removed {removed_rows} rows with missing/invalid '{DROPBACKS_COLUMN}' values.")

if df_filtered.empty:
    print(f"No valid data found in the '{DROPBACKS_COLUMN}' column to plot.")
    exit()

# --- Create the Histogram ---
plt.figure(figsize=(12, 7)) # Adjust figure size as needed

# Using seaborn's histplot is often nicer and integrates well with pandas
# kde=True adds a density curve for visualizing the shape
# bins='auto' tries to find a reasonable number of bins, or you can set an integer like bins=50
sns.histplot(data=df_filtered, x=DROPBACKS_COLUMN, bins='auto', kde=True)

# --- Customize and Show Plot ---
plt.title(f'Distribution of Quarterback {DROPBACKS_COLUMN.capitalize()} (Season Data)')
plt.xlabel(f'Number of {DROPBACKS_COLUMN.capitalize()}')
plt.ylabel('Number of Quarterbacks (Frequency)')
plt.grid(axis='y', linestyle='--', alpha=0.7) # Add horizontal grid lines

# Optional: Add vertical lines for potential thresholds if you want to visualize them
# plt.axvline(100, color='red', linestyle='--', linewidth=1, label='Threshold = 100')
# plt.axvline(150, color='orange', linestyle='--', linewidth=1, label='Threshold = 150')
# plt.legend()

print("\n--- Descriptive Statistics for Dropbacks ---")
print(df_filtered[DROPBACKS_COLUMN].describe())
print("---------------------------------------------")

print("\nShowing histogram plot...")
plt.show()

print(f"\nLook at the histogram shape and the statistics above.")
print(f"Consider where the frequency drops significantly to choose your minimum {DROPBACKS_COLUMN} threshold.")