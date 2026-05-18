# Qatar Greenhouse Atlas

This is a Streamlit dashboard for national greenhouse suitability and investment screening in Qatar.

It includes:

- A clickable Qatar map.
- Suitability scoring for microclimate, grid access, roads, land use, and exclusion buffers.
- Hard exclusion of residential, industrial, protected, flood-prone, and urban expansion areas.
- Hard exclusion of water/offshore clicks outside the Qatar land mask.
- Crop-technology calculations for yield, water, cooling water, energy, costs, profit, payback, and ROI.
- Advanced greenhouse microclimate estimates for indoor temperature, RH, ventilation, and air changes per hour.
- Crop-tech comparison and optimisation tabs.
- CSV and PDF report downloads.
- Synthetic GIS layers that run immediately.
- Optional support for official GeoJSON layers.

## Run

Open Terminal and run:

```bash
cd "/Users/sshannak/Documents/Codex/2026-05-17/i-need-you-to-developa-greenbhouse"
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/streamlit run app.py --server.port 8504 --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Then open:

```text
http://127.0.0.1:8504
```

Keep the Terminal window open while using the dashboard.

## Optional Data Layers

Create a `data/` folder and add any of these files:

```text
data/qatar_kahramaa_lines.geojson
data/qatar_ashghal_roads.geojson
data/qatar_landuse.geojson
```

If a file is missing, the app uses synthetic planning-grade proxy layers.

The land-use file should include:

```text
landuse
greenhouse_ok
```

`greenhouse_ok` should be `true` for land classes where greenhouse development can be considered, and `false` for residential, industrial, reserves, flood-prone basins, and other excluded classes. If `greenhouse_ok` is missing, the app infers it only for obvious agricultural/open land labels.
