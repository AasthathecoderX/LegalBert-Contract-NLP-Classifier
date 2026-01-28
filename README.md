# 📄 NLP-Based Contract Clause Classification & Entity Extraction

An **end-to-end legal document analysis system** that automatically **classifies contract clauses** and **extracts legal entities** using **LegalBERT + NLP**, wrapped in a clean Flask web application 🚀

---

## ✨ Features

- ✅ Upload **PDF / TXT legal documents**
- ✅ Automatic **clause segmentation & classification**
- ✅ **Named Entity Recognition (NER)** for legal entities

- ✅ Interactive UI 

- ✅ Flask-based backend API

---

## 🧠 Powered By

- 🤖 **LegalBERT** (fine-tuned)
- 🧪 NLP pipelines for clause classification
- 🔍 Entity extraction (NER)
- 🐍 Python + Flask backend
- 🎨 HTML / CSS / Vanilla JavaScript frontend

---

## 🗂️ Project Structure

```text
LegalBert/
│
├── backend/
│   ├── app.py                  # Flask backend
│   ├── models/                 # Trained LegalBERT model
│   └── utils/
│       ├── inference.py        # Model inference logic
│       └── pdf_extractor.py    # PDF text extraction
│
├── frontend/
│   ├── templates/
│   │   ├── index.html          # Home page
│   │   ├── analyze.html        # Analyze page
│   │   └── about.html
│   │
│   └── static/
│       ├── styles.css          # Global styles
│       ├── analyze.css         # Analyze page styles
│       └── analyze.js          # Frontend logic
│
├── uploads/                    # Temporary uploaded files
└── README.md
```

---

## 🖥️ Frontend Highlights

### 📤 File Upload
- Drag & drop or **Choose File**
- Supports `.pdf` and `.txt`

### 📊 Results Summary
- Total clauses
- Clauses with labels
- Total entities
- High-importance clauses

### 📑 Clause Viewer
- Shows **Top 10 clauses** by default
- 🔘 **View More** → see all classified clauses
- 🔁 Toggle back to **Top 10**

### 🏷️ Entity Display
- Entity label + text

---

## 🚀 How to Run

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
    ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt

   ```

3. **Start the server**
   ```bash
   cd backend
   python app.py

   ```

---
## 🎯 Use Cases

- 📜 **Contract review**  
- ⚖️ **Legal compliance analysis**  
- 📚 **Academic legal NLP research**  
- 🏢 **Enterprise contract intelligence**




