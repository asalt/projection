import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def perform_pca(matrix, n_components=4):
    """Perform PCA on a given matrix and return the transformed components."""
    pca = PCA(n_components=n_components)
    projections = pca.fit_transform(matrix)
    return projections

def plot_3d_transition(projections, color_map, labels=('PC 1', 'PC 2', 'PC 3', 'PC 4')):
    """Plot a 3D transition between the first two principal components (PC1/PC2) and the next two (PC3/PC4)."""
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for PC1 vs PC2, adding an artificial 'depth' dimension
    ax.scatter(projections[:, 0], projections[:, 1], zs=0, zdir='z', c=color_map, marker='o', label='PC1 vs PC2')

    # Scatter plot for PC3 vs PC4 at a different depth
    ax.scatter(projections[:, 2], projections[:, 3], zs=1, zdir='z', c=color_map, marker='o', label='PC3 vs PC4')

    # Add lines connecting corresponding points across the 'depth' dimension
    for i in range(len(projections)):
        ax.plot([projections[i, 0], projections[i, 2]],
                [projections[i, 1], projections[i, 3]],
                [0, 1], color='gray', linestyle='--')

    # Add shading for the PC1/PC2 plane at z=0
    x_min, x_max = projections[:, 0].min(), projections[:, 0].max()
    y_min, y_max = projections[:, 1].min(), projections[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, color='blue', alpha=0.2)

    # Add shading for the PC3/PC4 plane at z=1
    x_min, x_max = projections[:, 2].min(), projections[:, 2].max()
    y_min, y_max = projections[:, 3].min(), projections[:, 3].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
    zz = np.ones_like(xx)
    ax.plot_surface(xx, yy, zz, color='green', alpha=0.2)

    # Add axis lines with arrows for PC1/PC2 at z=0 (positive and negative directions)
    ax.quiver(0, 0, 0, x_max, 0, 0, color='black', linewidth=2, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, -x_max, 0, 0, color='black', linewidth=2, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, y_max, 0, color='black', linewidth=2, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, -y_max, 0, color='black', linewidth=2, arrow_length_ratio=0.1)

    # Add axis lines with arrows for PC3/PC4 at z=1 (positive and negative directions)
    ax.quiver(0, 0, 1, x_max, 0, 0, color='black', linewidth=2, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 1, -x_max, 0, 0, color='black', linewidth=2, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 1, 0, y_max, 0, color='black', linewidth=2, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 1, 0, -y_max, 0, color='black', linewidth=2, arrow_length_ratio=0.1)

    # Add a vertical z-axis line connecting the two origins
    ax.quiver(0, 0, 0, 0, 0, 1, color='black', linewidth=2, arrow_length_ratio=0.1)

    # Labels for the 3D plot
    ax.set_xlabel(f'{labels[0]} / {labels[2]}')
    ax.set_ylabel(f'{labels[1]} / {labels[3]}')
    ax.set_zlabel('Transition')
    plt.title('Transition between PC1/PC2 and PC3/PC4 with Full Axis Lines')

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
    projections = perform_pca(matrix)
    color_map = create_color_map(len(projections))
    fig = plot_3d_transition(projections, color_map)
    fig.savefig('demo.png')
