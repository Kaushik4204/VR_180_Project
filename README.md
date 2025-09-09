# 🎥 VR180 Inception MVP  

An experimental **VR180 media application** that leverages deep learning and modern web frameworks to deliver immersive video experiences. Built as a **minimum viable product (MVP)** to explore the intersection of **computer vision, generative AI, and VR**.  

---

## 🚀 Features  
- **VR180 Support** → optimized for stereoscopic 180° media.  
- **Deep Learning Integration** → AI-powered processing & enhancements.  
- **Web-based Interface** → lightweight UI powered by Gradio.  
- **Hackathon-ready Deployment** → easily launch on Hugging Face Spaces or locally.  

---

## 🛠️ Tech Stack  
- **Python 3.9+**  
- **Gradio** (for UI)  
- **PyTorch / TensorFlow** (for deep learning models)  
- **OpenCV** (for video/image processing)  

---

## 📦 Installation  

Clone the repo:  
```bash
git clone https://github.com/Kaushik4204/vr180_inception_mvp.git
cd vr180_inception_mvp
```

## Create and activate virtual environment:
```bash
python -m venv .venv
# Activate venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate
```

## Install dependencies:
```bash
pip install -r requirements.txt
```

## ▶️ Running the App

Run locally:
```bash
python app.py
```
---

## 🌐 Deploy on Hugging Face Spaces

Push this repo to GitHub.

Go to Hugging Face Spaces
 → New Space.

Select:

Repository type: Gradio

Link GitHub Repo: paste your repo link.

Done 🎉 – your app runs live in the browser.

---

## Folder Structure

vr180_inception_mvp/

├── app.py

├── app/

│   ├── __init__.py

│   └── gradio_app.py   👈 put the code you pasted here

├── pipeline/

│   ├── videoio.py

│   ├── depth.py

│   ├── dibr.py

│   └── vr180_pack.py

└── outputs/

└── app,py

└── requirements.txt
--
## 🤝 Contributing

Contributions are welcome!

Fork the repo

Create a new branch

Submit a pull request 🚀

---

## 📜 License

This project is licensed under the MIT License 


## 👨‍💻 Author

Kaushik Puli

GitHub: Kaushik4204


