## 🧠 Model Architecture

The model is based on NVIDIA’s end-to-end self-driving CNN approach, where the network directly predicts the steering angle from an input image.

### Input

* Image size: **466 × 405 × 3 (RGB)**
* Normalized to **[0, 1]**

### Preprocessing (inside model)

* Crop top region (removes sky / irrelevant area)
* Resize to **66 × 200** (standard for driving models)

---

### CNN Layers

The model uses 5 convolution layers:

* Conv1 → 24 filters (5×5, stride 2)
* Conv2 → 36 filters (5×5, stride 2)
* Conv3 → 48 filters (5×5, stride 2)
* Conv4 → 64 filters (3×3)
* Conv5 → 64 filters (3×3)


---

### Fully Connected Layers

After feature extraction:

* Dense → 100
* Dense → 50
* Dense → 10
* Output → 1 (steering angle)

---

### Output

* Single value: **steering angle**
* Range: approximately **[-1, 1]**

  * Left → positive
  * Right → negative

---

### Summary

```text
Image → CNN → Features → Dense Layers → Steering Angle
```

This architecture is inspired by NVIDIA’s self-driving model, with a simplified and efficient design.

---

### ⚠️ Current Limitation

The model performs well on:

* straight roads
* slight curves

But struggles with:

* sharp turns
* complex road layouts

This is mainly due to dataset limitations, not the architecture itself.
