# Brazil HexGrid Generator  

Generates a hexagonal grid of coordinates covering Brazilian cities for general purpose (e.g., solar energy analysis).  

![Example Grid Plot](https://github.com/Mekepi/brazil-hexgrid-generator/blob/main/outputs/plots/SP/%5B3550308%5D_S%C3%A3o%20Paulo.png) *Example output for São Paulo*

## Features  
- **OOP-based** processing of cities/states (`city`, `state`, `country` classes).  
- **Parallelized** grid generation using `multiprocessing`.  
- **Adjustable hexagon radius**.  
- **Visualization** with Matplotlib (optional).  

## Usage  
### 1. Generate Grids  
```python
from src.hexgrid import hexgrid_generator
hexgrid_generator(radius=1.35)  # Saves to /outputs/Brasil
```

### 2. Plot Grids
```python
from src.hexgrid import hexgrid_plot
hexgrid_plot()  # Saves PNGs to /outputs/plots/[STATE]/[CITY].png
```

## Data Preparation
The geographic data used in this project comes from IBGE's BC250 series.
