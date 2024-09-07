import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def perform_pca(matrix, n_components=4):
    """Perform PCA on a given matrix and return the transformed components."""
    pca = PCA(n_components=n_components)
    projections = pca.fit_transform(matrix)
    return (projections, pca)

def plot_3d_transition(projections, color_map, labels=('PC 1', 'PC 2', 'PC 3', 'PC 4'), explained_variance=None):
    """Plot a 3D transition between the first two principal components (PC1/PC2) and the next two (PC3/PC4)."""
    if explained_variance is None:
        explained_variance = np.zeros(len(labels))
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for PC1 vs PC2, adding an artificial 'depth' dimension
    ax.scatter(projections[:, 0], projections[:, 1], s = 32, zs=0, zdir='z', c=color_map, marker='o', label='PC1 vs PC2')

    # Scatter plot for PC3 vs PC4 at a different depth
    ax.scatter(projections[:, 2], projections[:, 3], s=32, zs=1, zdir='z', c=color_map, marker='o', label='PC3 vs PC4')
    # Add lines connecting corresponding points across the 'depth' dimension
    for i in range(len(projections)):
        ax.plot([projections[i, 0], projections[i, 2]],
                [projections[i, 1], projections[i, 3]],
                [0, 1], color='gray', linestyle='--')

    x_min, x_max = min(projections[:, 0].min(), projections[:, 2].min()), max(projections[:, 0].max(), projections[:, 2].max())
    y_min, y_max = min(projections[:, 1].min(), projections[:, 3].min()), max(projections[:, 1].max(), projections[:, 3].max())
    x_min = y_min = min(x_min, y_min, -x_max, -y_max)
    x_max = y_max = max(x_max, y_max, -x_min, -y_min)

    x_min *= 1.1
    x_max *= 1.1
    y_min *= 1.1
    y_max *= 1.1

    # Add shading for the PC1/PC2 plane at z=0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 96), np.linspace(y_min, y_max, 96))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, color='blue', alpha=0.14)
    # Add shading for the PC3/PC4 plane at z=1
    zz = np.ones_like(xx)
    ax.plot_surface(xx, yy, zz, color='green', alpha=0.14)
    print(x_min, x_max)
    print(y_min, y_max)

 
    # Add the xy box at z=0, 1
    for z in (0, 1):
        ax.plot([x_min, x_max], [y_min, y_min], [z, z], color='black', linestyle='-', linewidth=1)
        ax.plot([x_min, x_max], [y_max, y_max], [z, z], color='black', linestyle='-', linewidth=1)
        ax.plot([x_min, x_min], [y_min, y_max], [z, z], color='black', linestyle='-', linewidth=1)
        ax.plot([x_max, x_max], [y_min, y_max], [z, z], color='black', linestyle='-', linewidth=1)
        ax.quiver(0, 0, z, 0, -y_max, 0, color='black', linewidth=1.6, arrow_length_ratio=0.08)
        ax.quiver(0, 0, z, x_max, 0, 0, color='black', linewidth=1.6, arrow_length_ratio=0.08)
        ax.quiver(0, 0, z, -x_max, 0, 0, color='black', linewidth=1.6, arrow_length_ratio=0.08)
        ax.quiver(0, 0, z, 0, y_max, 0, color='black', linewidth=1.6, arrow_length_ratio=0.08)
        #ax.plot([0, 0], [y_min, y_max], [z, z], color='black', linestyle='-', linewidth=1)
        #ax.plot([x_min, x_max], [0, 0], [z, z], color='black', linestyle='-', linewidth=1)

    ax.text(x_max+.35, 0, 0, f'PC1 pct var {explained_variance[0]:.2%}', color='black', ha='center', va='center', zdir=(0, y_min, 0))
    ax.text(0, y_min-.35, 0, f'PC2 pct var {explained_variance[1]:.2%}', color='black', ha='center', va='center', zdir=(x_min,0,0))
    ax.text(x_min-.35, 0, 1, f'PC3 pct var {explained_variance[2]:.2%}', color='black', ha='center', va='center', zdir =(0, y_min, 0))
    ax.text(0, y_max+.35, 1, f'PC4 pct var {explained_variance[3]:.2%}', color='black', ha='center', va='center', zdir = (x_min, 0, 0))


    ax.set_xlim(xmin=x_min, xmax=x_max)
    ax.set_ylim(ymin=y_min, ymax=y_max)
    # Add a vertical z-axis line connecting the two origins
    # ax.quiver(0, 0, 0, 0, 0, 1, color='black', linewidth=2, arrow_length_ratio=0.1)

    # Labels for the 3D plot
    #ax.set_xlabel(f'{labels[0]} / {labels[2]}')
    #ax.set_ylabel(f'{labels[1]} / {labels[3]}')
    # ax.set_zlabel('Transition')
    ax.set_zticks([])
    ax.set_xticks([])
    ax.set_yticks([])

    # ax.zaxis.line.set_lw(0)

    ax.set_facecolor('white')
    ax.xaxis.line.set_lw(0)
    ax.yaxis.line.set_lw(0)
    ax.zaxis.line.set_lw(0)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        # axis.set_ticklabels([])
        axis._axinfo['axisline']['linewidth'] = 0
        axis._axinfo['axisline']['color'] = "white"
        #axis._axinfo['grid']['linewidth'] = 0.5
        #axis._axinfo['grid']['linestyle'] = "--"
        #axis._axinfo['grid']['color'] = "#d1d1d1"
        #axis._axinfo['tick']['inward_factor'] = 0.0
        #axis._axinfo['tick']['outward_factor'] = 0.0
        axis.set_pane_color((1, 1, 1))
    ax.set_xlim(xmin=x_min, xmax=x_max)
    ax.set_ylim(ymin=y_min, ymax=y_max)
    plt.title('Transition between PC1/PC2 and PC3/PC4 with Full Axis Lines')

    ax.grid(False)

    # Show the 3D plot
    plt.show()

    return fig

def create_color_map(n_samples):
    """Create a simple color map dividing samples into 3 groups."""
    colors = ['red', 'green', 'blue']
    return [colors[i // (n_samples // 3)] for i in range(n_samples)]

# Example usage with a random matrix
if __name__ == "__main__":
    matrix = np.random.rand(9, 100)  # Random matrix where each row is a sample and each column a measurement
    projections, pcaobj = perform_pca(matrix)
    color_map = create_color_map(len(projections))
    explained_variance = pcaobj.explained_variance_ / pcaobj.explained_variance_.sum()
    fig = plot_3d_transition(projections, color_map, explained_variance=explained_variance)
    fig.savefig('demo.png')
