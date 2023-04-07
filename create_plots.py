import os
import numpy as np
import matplotlib.pyplot as plt

# define the parameters
hidden_size_list = [50, 100, 150]
batch_size_list = [50, 100, 150]
lr_list = [0.1, 0.5, 1.0]

# define the folder where the text files are stored
folder = "performance"

# define the x values
x1 = range(1, 11)
x2 = range(11, 21)

# create the figures folder if it doesn't exist
if not os.path.exists("figures"):
    os.makedirs("figures")

# define the color palette
color_palette = plt.cm.get_cmap('tab20', len(hidden_size_list) * len(batch_size_list) * len(lr_list))

# create empty lists to store the left values and colors for each parameter combination
left_values_list = [[] for i in range(len(lr_list) * len(hidden_size_list) * len(batch_size_list))]
color_list = []

# loop over all combinations of parameters and read the data
for i, hidden_size in enumerate(hidden_size_list):
    for j, batch_size in enumerate(batch_size_list):
        for k, lr in enumerate(lr_list):
            # Construct the file name
            file_name = f"model_h{hidden_size}_b{batch_size}_lr{lr}.txt"
            file_path = os.path.join(folder, file_name)
            
            # Read the data from the file and extract the left values
            with open(file_path, "r") as f:
                data = f.read().splitlines()
            left_values = []
            for line in data:
                elements = line.split()
                if len(elements) > 0:
                    left_values.append(float(elements[0]))
            
            # Append the left values to the appropriate list
            left_values_list[i*9+j*3+k] = left_values
            
            # Assign a color to the parameter combination
            color_list.append(color_palette((i*len(batch_size_list)*len(lr_list) + j*len(lr_list) + k)/(len(hidden_size_list)*len(batch_size_list)*len(lr_list))))

# create the plots for listener0 and speaker0
plt.figure(figsize=(12, 6))
plt.suptitle("Loss per Epoch - listener0", fontsize=16, fontweight='bold')
plt.xlabel("Epochs", fontweight='bold')
plt.ylabel("Loss", fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim([1, 10])
plt.xticks(range(1, 11))
for i in range(len(lr_list) * len(hidden_size_list) * len(batch_size_list)):
    plt.plot(x1, left_values_list[i][:10], label=f"h{hidden_size_list[i//9]}_b{batch_size_list[(i%9)//3]}_lr{lr_list[i%3]}", color=color_list[i])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, fontsize=12)
plt.savefig(f"figures/listener0.png", dpi=300, bbox_inches="tight")
plt.clf()

plt.figure(figsize=(12, 6))
plt.suptitle("Loss per Epoch - speaker0", fontsize=16, fontweight='bold')
plt.xlabel("Epochs", fontweight='bold')
plt.ylabel("Loss", fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim([11, 20])
plt.xticks(range(11, 21), [str(i) for i in range(1, 11)])
for i in range(len(lr_list) * len(hidden_size_list) * len(batch_size_list)):
    plt.plot(x2, left_values_list[i][10:], label=f"h{hidden_size_list[i//9]}_b{batch_size_list[(i%9)//3]}_lr{lr_list[i%3]}", color=color_list[i])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, fontsize=12)
plt.savefig(f"figures/speaker0.png", dpi=300, bbox_inches="tight")
plt.clf()

# create empty lists to store the minimum loss values and corresponding model names for each epoch
listener_min_losses = [[] for i in range(10)]
speaker_min_losses = [[] for i in range(10)]

# loop over all combinations of parameters and read the data
for i, hidden_size in enumerate(hidden_size_list):
    for j, batch_size in enumerate(batch_size_list):
        for k, lr in enumerate(lr_list):
            # Construct the file name
            file_name = f"model_h{hidden_size}_b{batch_size}_lr{lr}.txt"
            file_path = os.path.join(folder, file_name)
            
            # Read the data from the file and extract the left values
            with open(file_path, "r") as f:
                data = f.read().splitlines()
            left_values = []
            for line in data:
                elements = line.split()
                if len(elements) > 0:
                    left_values.append(float(elements[0]))
            
            # Find the three smallest losses for each epoch and store the corresponding model names
            for epoch in range(10):
                epoch_losses = left_values[epoch*10:(epoch+1)*10]
                min_losses = sorted(zip(epoch_losses, [f"h{hidden_size}_b{batch_size}_lr{lr}"]*10))[:3]
                listener_min_losses[epoch].extend(min_losses)
                speaker_min_losses[epoch].extend(min_losses)
                
# print the results for the listener and speaker losses
for epoch in range(1):
    listener_min_losses_epoch = sorted(listener_min_losses[epoch])
    speaker_min_losses_epoch = sorted(speaker_min_losses[epoch])
    print(f"Listener Loss: {listener_min_losses_epoch[:3]}")
    print(f"Speaker Loss: {speaker_min_losses_epoch[:3]}")
