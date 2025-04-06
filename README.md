# M-SCAHN (Science Fair 2024) – Deep Learning + Image Processing Demo  
**Live demo:** [https://m-scahn-software.vercel.app/](https://m-scahn-software.vercel.app/)

This repository contains a proof-of-concept demo I built for my 2024 science fair project: a deep learning and image processing system for skin lesion analysis. It's intended to showcase how the models and algorithms work under the hood; not to serve as downloadable software.

**Note:** This codebase exists to demonstrate the architecture and image processing pipeline. The actual software can only be used through the live demo linked above.

---

## What This Is

This is a stripped-down version of the full application stack used in my science fair project, featuring:

- **Models** – Classification and segmentation.
- **Image Processing** – Preprocessing, segmentation overlays, ABC feature scoring (asymmetry, border, color).
- **Frontend UI** – Built with PyQt5 (for this repo) and deployed via Next.js (in the demo).

The repo demonstrates how these components interact to form a functioning DL pipeline. If you're curious about the real user experience, check out the live web demo:

👉 **[Try the Live Web App](https://m-scahn-software.vercel.app/)**

---

## Project Structure - Interesting Files

```
models/              # Model implementations
process_image.py     # All image processing and inference pipeline
```