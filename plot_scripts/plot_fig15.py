import matplotlib.pyplot as plt
import sys, os
from matplotlib.ticker import FuncFormatter
import numpy as np

# Function to read the file and return y-values and corresponding x-values
def average_from_file(filename) -> float:
    try:
        with open(filename, 'r') as file:
            numbers = [int(line.strip()) for line in file]
        return sum(numbers) / len(numbers) if numbers else 0
    except FileNotFoundError:
        return 0

def to_repeated_hex_lower(idx):
    hex_str = hex(idx)[2:].lower()
    return ''.join([ch*2 for ch in hex_str])

def to_repeated_hex_upper(idx):
    hex_str = hex(idx)[2:].upper()
    return ''.join([ch*2 for ch in hex_str])

def percentage_formatter(x, pos):
    return f'{x:.0f}%'

def plot_heatmap(data):
    # Sort columns by sum
    column_sums = np.sum(data, axis=0)
    sorted_column_indices = np.argsort(column_sums, stable=True)[::-1]
    data_sorted_columns = data[:, sorted_column_indices]
    
    # Identify and remove columns with all zero entries
    nonzero_column_mask = np.any(data_sorted_columns != 0, axis=0)
    data_sorted_columns = data_sorted_columns[:, nonzero_column_mask]
    sorted_column_indices = sorted_column_indices[nonzero_column_mask]
    
    # Sort rows by sum of the sorted columns
    row_sums = np.sum(data_sorted_columns, axis=1)
    sorted_row_indices = np.argsort(row_sums)[::-1]
    data_sorted_rows = data_sorted_columns[sorted_row_indices, :]
    
    # Identify and remove rows with all zero entries
    nonzero_row_mask = np.any(data_sorted_rows != 0, axis=1)
    sorted_data = data_sorted_rows[nonzero_row_mask, :]
    sorted_row_indices = sorted_row_indices[nonzero_row_mask]

    plt.imshow(sorted_data, cmap='hot_r', interpolation='nearest')

    hex_column_labels = [to_repeated_hex_upper(idx) for idx in sorted_column_indices]
    hex_row_labels = [to_repeated_hex_upper(idx) for idx in sorted_row_indices]

    # Set x and y ticks and labels
    plt.xticks(ticks=np.arange(len(sorted_column_indices)), labels=hex_column_labels)
    plt.yticks(ticks=np.arange(len(sorted_row_indices)), labels=hex_row_labels)

    plt.ylabel('Victim Data Pattern', fontsize=14)
    plt.xlabel('Aggressor Data Pattern', fontsize=14)
    
    cbar = plt.colorbar()
    cbar.set_label("% of hammers\ntriggering bit-flips", rotation=270, labelpad=25)
    # Set the formatter for the colorbar
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

    plt.tight_layout()

data_pat = ['00', '11', '22', '33', '44', '55', '66', '77', '88', '99', 'aa', 'bb', 'cc', 'dd', 'ee', 'ff']

# Main execution
if __name__ == "__main__":

    num_agg = 24

    # TODO: Update these lists into a dictionary
    bank_lst = [256, 2048, 2048, 2048, 5120, 6400, 6400, 6400]
    flip_lst = [30329, 3543, 13057, 23029, 4371, 13635, 21801, 28498]

    i = int(sys.argv[1])
    bank = bank_lst[i]
    flip = flip_lst[i]

    # Generate input and output directories
    HAMMER_ROOT = os.environ['HAMMER_ROOT']
    INPUT_DIR = os.path.join(HAMMER_ROOT, "results", "fig15", "raw_files")
    OUTPUT_DIR = os.path.join(HAMMER_ROOT, "results", f"fig15_flip{i}")

    data = np.zeros((16, 16))

    for j in range(16):
        for k in range(16):
            vic_pat, agg_pat = to_repeated_hex_lower(j), to_repeated_hex_lower(k)
            filename = os.path.join(INPUT_DIR, 
                                    f"{num_agg}agg_b{bank}_count_flip{flip}_{vic_pat}{agg_pat}.txt")
            data[j][k] = average_from_file(filename) / 50 * 100
    
    plt.figure(figsize=(7, 3))
    plot_heatmap(data)

    output_image = os.path.join(OUTPUT_DIR, f"fig15_flip{i}.pdf")
    plt.savefig(output_image)
    plt.close()