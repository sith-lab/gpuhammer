import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sys, os

# Report the percentage of bit-flips triggered as recorded in the file.
def average_from_file(filename) -> float:
    with open(filename, 'r') as file:
        numbers = [int(line.strip()) for line in file]
    return sum(numbers) / len(numbers) if numbers else 0


def plot_histogram(x, y):
    plt.bar(x, y, edgecolor='black')
    
    plt.xlabel('n-Sided Aggressor Pattern', fontsize=14)
    plt.ylabel('% of hammers\ntriggering bit-flips', fontsize=14)
    
    plt.xticks(ticks=x, labels=[str(i) for i in x])
    plt.grid(axis='y', linestyle='-', alpha=0.7)
    plt.tight_layout()


if __name__ == "__main__":

    # Generate input and output directories
    HAMMER_ROOT = os.environ['HAMMER_ROOT']
    INPUT_DIR = os.path.join(HAMMER_ROOT, "results", "fig11", "raw_files")
    OUTPUT_DIR = os.path.join(HAMMER_ROOT, "results", "fig11")
    
    bank_no = 256
    reps = 50

    # Read the data from the files
    x = list(range(8, 25))      # Test 8 to 24 sided patterns
    y = [0] * 17
    for num_agg in range(8, 25):
        filename = os.path.join(INPUT_DIR, f"{num_agg}agg_b{bank_no}_count.txt")
        y[num_agg - 8] = average_from_file(filename) / reps

    # Plot
    plt.figure(figsize=(6, 2.5))
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    plot_histogram(x, y)
    plt.savefig(os.path.join(OUTPUT_DIR, "fig11.pdf"))