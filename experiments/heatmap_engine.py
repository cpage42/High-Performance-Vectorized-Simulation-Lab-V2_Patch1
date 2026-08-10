import numpy as np
from flask import jsonify

def generate_heatmap_payload(x_array, y_array, grid_resolution=400):
    # Determine the bounding box dynamically
    x_min, x_max = np.min(x_array), np.max(x_array)
    y_min, y_max = np.min(y_array), np.max(y_array)

    # Calculate 2D histogram (density grid)
    heatmap, xedges, yedges = np.histogram2d(
        x_array, y_array, 
        bins=grid_resolution, 
        range=[[x_min, x_max], [y_min, y_max]]
    )

    # Flatten and convert to standard types for JSON serialization
    return jsonify({
        'density_grid': heatmap.T.flatten().tolist(),
        'x_bounds': [float(x_min), float(x_max)],
        'y_bounds': [float(y_min), float(y_max)],
        'resolution': grid_resolution
    })
