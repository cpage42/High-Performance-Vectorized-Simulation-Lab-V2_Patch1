document.addEventListener("DOMContentLoaded", () => {
    const executeBtn = document.getElementById("executeBtn");
    const renderer = new HeatmapRenderer("heatmapCanvas");

    executeBtn.addEventListener("click", async () => {
        // Grab heatmap config values
        const resInput = document.getElementById("heatmapRes")?.value || 400;
        const colorTheme = document.getElementById("heatmapColor")?.value || "fire";

        // Grab standard V2 inputs (assuming IDs match your V2 layout)
        const payload = {
            walkers: parseInt(document.getElementById("walkers")?.value || 1000000),
            steps: parseInt(document.getElementById("steps")?.value || 100),
            variance: parseFloat(document.getElementById("variance")?.value || 2.25),
            resolution: parseInt(resInput)
        };

        try {
            const response = await fetch('/execute_sim', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (response.ok) {
                const data = await response.json();
                renderer.render(data, colorTheme);
            } else {
                console.error("Server Error:", response.statusText);
            }
        } catch (e) {
            console.error("Fetch Error:", e);
        }
    });
});
