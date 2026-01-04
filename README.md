#  FaultLens — AI-Powered Product Inspection System

**FaultLens** (Fault Lens Analysis System) is an intelligent AI pipeline for **inspecting products** in e-commerce and manufacturing workflows. It uses **Computer Vision, Deep Learning, Multi-Agent reasoning, and NLP** to detect defects, compare images, and provide transparent, human-readable explanations for each inspection decision.

---

##  Features

-  **Image Comparison**  
  Automatically compares **customer-submitted images** with **reference product images** per order.

-  **Multi-Agent Inspection Pipeline**  
   **Vision Agent**: Detects product parts, crops, and extracts features.  
   **Comparison Agent**: Highlights differences or defects between reference and submitted images.  
   **Policy Agent**: Validates findings against company policies.  
   **Decision Agent**: Generates final inspection verdict with explanations.

-  **RAG-Powered Reasoning**  
  Uses **Retrieval-Augmented Generation (RAG)** to explain why a product passed or failed inspection in plain English.

- **Insightful Dashboard (WIP)**  
  Displays inspection results, defect counts, and policy-based reasoning visually.

-  **Tech Stack**  
   Language: Python  
   Computer Vision: YOLOv8, OpenCV  
   Machine Learning: Scikit-learn / lightweight models  
   Deep Learning: PyTorch (optional CNN/MLP)  
   NLP: LLMs via Ollama + RAG  
   UI: Gradio (interactive image upload)  
   Data Storage: Local file system for reference images  

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/USERNAME/FaultLens.git
cd FaultLens

# 2. (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```
---
## 🧠 System Pipeline Overview

```bash

Product Category
      ↓
Dimensions Agent
      ↓
Upload photos
      ↓
Vision Agent (YOLOv8 + Analysis)
      ↓
Comparison Agent (Diff & Feature Analysis)
      ↓
Policy Agent (RAG + Policy Docs)
      ↓
Decision Agent (Final Verdict + Explanation)
      ↓
Result Gradio UI
```

- Enter product Category
- Upload customer-submitted image, Order ID and issue description 
- FaultLens detects defects, compares images, checks policies, and produces a verbal explanation with results.
  
---

## 📄 License
**This project is licensed under the Apache 2.0 License.** 

---

## 👥 Contributors
- Reem Hatem Zekry
- Shimaa Reda
- Ahmed Khafagy
- Ahmed Hegazy
  
---

## 💡 Future Plans

**Multi-language support for product policies** 

**Real-time inspection (video streams)** 

**Advanced reasoning with larger LLMs** 

**Interactive analytics dashboard**

---

## 📬 Contact
For questions or feedback, open an issue on GitHub or contact the contributors directly.
