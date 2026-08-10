class HeatmapRenderer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.width = this.canvas.width;
        this.height = this.canvas.height;
    }

    render(payload, colorTheme) {
        const grid = payload.density_grid;
        const res = payload.resolution;
        const maxDensity = Math.max(...grid) || 1; 

        // Size of each drawn pixel block
        const cellW = this.width / res;
        const cellH = this.height / res;

        // Clear previous render
        this.ctx.fillStyle = '#000000';
        this.ctx.fillRect(0, 0, this.width, this.height);

        // Draw the density grid
        for (let row = 0; row < res; row++) {
            for (let col = 0; col < res; col++) {
                const val = grid[row * res + col];
                if (val > 0) {
                    const i = val / maxDensity;
                    let r, g, b;
                    
                    if (colorTheme === 'fire') {
                        r = Math.floor(255 * i); g = Math.floor(100 * i); b = 50;
                    } else if (colorTheme === 'ice') {
                        r = 50; g = Math.floor(150 * i); b = Math.floor(255 * i);
                    } else if (colorTheme === 'matrix') {
                        r = 0; g = Math.floor(255 * i); b = 0;
                    } else { // Default grayscale
                        r = Math.floor(255 * i); g = Math.floor(255 * i); b = Math.floor(255 * i);
                    }
                    
                    this.ctx.fillStyle = `rgb(${r},${g},${b})`;
                    this.ctx.fillRect(col * cellW, row * cellH, cellW, cellH);
                }
            }
        }
    }
}
