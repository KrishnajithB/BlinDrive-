## 📂 Dataset

The dataset is not included due to size (>2GB).

### Format

images/
 ├── frame_000001.png
 ├── frame_000002.png

labels.csv:
image, steering
frame_000001.png, 0.25
frame_000002.png, -0.30

### How to generate data

Use:
python scripts/collect_data.py