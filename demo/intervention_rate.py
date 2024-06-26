import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def scale_array_to_100_elements(arr):
    original_length = len(arr)
    if original_length == 100:
        return arr
    
    # Original indices
    original_indices = np.arange(original_length)
    
    # New indices for interpolation
    new_indices = np.linspace(0, original_length - 1, 100)
    
    # Interpolation function
    interp_function = interp1d(original_indices, arr, kind='linear')
    
    # New scaled array
    scaled_array = interp_function(new_indices)
    
    return scaled_array



intervention = np.load('/home/necl/code/IDA/demo/experiments/lunar_lander/BM/A/block_0/intervention.npy', allow_pickle=True)
states = np.load('/home/necl/code/IDA/demo/experiments/lunar_lander/BM/A/block_0/states.npy', allow_pickle=True)

n_episodes = states.shape[0]
intervention_p = np.zeros((n_episodes, 100))
for episode_ix in range(n_episodes):
    intervention_p[episode_ix] = scale_array_to_100_elements(intervention[episode_ix])

plt.plot(np.mean(intervention_p, axis=0))
plt.xlabel("Normalized Time in Episode")
plt.ylabel("Intervention Probability")
plt.ylim([-0.1, 1.2])
plt.show()
pos_hist = np.vstack(states)[:,:2]
intervention_flag = np.hstack(intervention)

intervened_locs = pos_hist[np.where(intervention_flag==True)[0]]
x = intervened_locs[:, 0]
y = intervened_locs[:, 1]

# Create a hexbin plot
plt.figure(figsize=(10, 8))
hb = plt.hexbin(x, y, gridsize=100, cmap='inferno', mincnt=1, vmax=5)

# Add a color bar to show the counts
cb = plt.colorbar(hb, label='Count')

# Add titles and labels
plt.title('Heatmap of (x, y) Positions')
plt.xlabel('X Position')
plt.ylabel('Y Position')

# Show the plot
plt.show()

######
# pos_hist = np.load("/home/necl/code/IDA/eval/experiments/surrogate_pilots/lunarlander/lunarlander/noise/IDA/2024-06-04 13:44:15.239024/9_position_history.npy")
# intervention_flag = np.load("/home/necl/code/IDA/eval/experiments/surrogate_pilots/lunarlander/lunarlander/noise/IDA/2024-06-04 13:44:15.239024/9_goals_intervened_flag.npy")
# durations = np.load("/home/necl/code/IDA/eval/experiments/surrogate_pilots/lunarlander/lunarlander/noise/IDA/2024-06-04 13:44:15.239024/9_episode_lengths.npy")
# OLD  CODE
# # we have 10 episodes
# last_ix = 0
# ix=0
# intervention_prob = np.zeros((100, 100))
# for duration in durations:
#     print(duration[0])
#     intervention_prob[ix]=scale_array_to_100_elements(intervention_flag[last_ix:last_ix+duration[0]])
#     last_ix = last_ix+duration[0]
#     ix += 1

# plt.plot(np.mean(intervention_prob, axis=0))
# plt.xlabel("Normalized Time in Episode")
# plt.ylabel("Intervention Probability")
# plt.ylim([0, 0.5])
# plt.show()

# # Sample data: replace this with your actual (x, y) positions
# intervened_locs = pos_hist[np.where(intervention_flag==True)[0]]
# x = intervened_locs[:, 0]
# y = intervened_locs[:, 1]

# # Create a hexbin plot
# plt.figure(figsize=(10, 8))
# hb = plt.hexbin(x, y, gridsize=100, cmap='inferno', mincnt=1, vmax=5)

# # Add a color bar to show the counts
# cb = plt.colorbar(hb, label='Count')

# # Add titles and labels
# plt.title('Heatmap of (x, y) Positions')
# plt.xlabel('X Position')
# plt.ylabel('Y Position')

# # Show the plot
# plt.show()
# breakpoint()