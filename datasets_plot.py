import os
import matplotlib.pyplot as plt

# Path to the main directory
main_dir = 'datasets/IRMAS-TrainingData-images'

# Dictionary to store the count of files in each subdirectory
subdir_file_count = {}

# Iterate over each subdirectory in the main directory
for subdir in os.listdir(main_dir):
    subdir_path = os.path.join(main_dir, subdir)
    if os.path.isdir(subdir_path):
        # Count the number of files in the subdirectory
        file_count = len([f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))])
        subdir_file_count[subdir] = file_count

# Plotting the bar chart
# plt.bar(subdir_file_count.keys(), subdir_file_count.values())
# plt.xlabel('Classes')
# plt.ylabel('Samples Count')
# plt.title('Number of samples in each class')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()
# Assign different colors to each bar
colors = plt.cm.tab20(range(len(subdir_file_count)))

# Plotting the bar chart with colors
plt.bar(subdir_file_count.keys(), subdir_file_count.values(), color=colors)
plt.xlabel('Classes')
plt.ylabel('Samples Count')
plt.title('Classes distribution')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()