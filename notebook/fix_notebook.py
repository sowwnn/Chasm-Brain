import nbformat

path = "/home/sowwn/Workspace/ws/2026/I2fMRI/notebook/explore_what_where_insights.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == "code" and "corr_diff_volume = np.zeros_like(streams_data" in cell.source:
        source = cell.source
        if "stream_data = {r['stream']: r['diff'] for r in results}" not in source:
            source = source.replace(
                "corr_diff_volume = np.zeros_like(streams_data, dtype=np.float32)",
                "stream_data = {r['stream']: r['diff'] for r in results}\ncorr_diff_volume = np.zeros_like(streams_data, dtype=np.float32)"
            )
            cell.source = source
            print("Modified the problematic cell.")

with open(path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

