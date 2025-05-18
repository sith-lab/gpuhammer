import matplotlib.pyplot as plt
import numpy as np
import sys
import math

import os
HAMMER_ROOT = os.environ['HAMMER_ROOT']

def read_third_elements_from_folder(folder_path):
    all_lists = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            third_elements = []
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        try:
                            third_elements.append(float(parts[2]))
                        except ValueError:
                            # Skip non-numeric values
                            continue
            if third_elements:  # Only add non-empty lists
                all_lists.append(third_elements)

    return all_lists

def compute_elementwise_average(lists):
    if not lists:
        return []

    # Ensure all lists are the same length
    min_length = min(len(lst) for lst in lists)
    trimmed_lists = [lst[:min_length] for lst in lists]

    # Compute element-wise average
    average = [
        sum(values) / len(values)
        for values in zip(*trimmed_lists)
    ]
    return average

def running_max(lst):
    if not lst:
        return []
    result = [lst[0]]
    for x in lst[1:]:
        result.append(max(result[-1], x))
    return result

# Example usage
folder = f'{HAMMER_ROOT}/results/fig12_t4/D1/'
lists = read_third_elements_from_folder(folder)
for i in range(len(lists)):
    lists[i] = running_max(lists[i])
average = compute_elementwise_average(lists)

print("Element-wise average:", average)
# # # Sample data
# categories = list(["None", ".ca", ".cg", ".cs", ".cv", ".volatile"])
# cmap = plt.get_cmap("tab10")

fig,ax = plt.subplots()
fig.set_size_inches(7, 2)
ax.tick_params(axis='both', which='major', labelsize=16, left=False)
ax.tick_params(axis='both', which='minor', labelsize=16, left=False)
# Creating the bar plot
for x in ax.spines.values():
    x.set_alpha(0.5)
    x.set_edgecolor('grey') 
plt.grid(axis='y', color='grey', linestyle='-', alpha=0.5,zorder=0, linewidth=1)
plt.plot(average ,  label='Different Bank', linewidth=3,zorder=3)

# # # Adding labels
plt.xlabel('Exploit Attempts', fontsize=20)
plt.ylabel('Average\nRAD', fontsize=20)
# plt.xlabel('Load Modifiers', fontsize=22)
plt.yticks([x/100 for x in range(20, 101, 20)], [f"{x}%" for x in range(20, 101, 20)])
# # plt.yscale('log')

# # Set custom y-ticks
# # plt.yticks([1e3, 1e5, 1e7], ['10^3', '10^5', '10^7'])

# # Adding legend
# # handles, labels = ax.get_legend_handles_labels()
# # plt.legend(
# #     loc="upper center",
# #     fontsize=16,
# #     ncols=5,
# #     bbox_to_anchor=(0.5, 1.24))

fig.savefig(f"{HAMMER_ROOT}/results/fig12_t4/fig12.pdf", transparent=True, format="pdf", bbox_inches="tight")