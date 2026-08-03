# 🛰️ DataPilot AI

> **An AI-powered autonomous data analyst built with Streamlit that lets you explore datasets, connect SQLite databases, and chat with your data using natural language.**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green.svg)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-purple.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ✨ Features

- 📊 **Interactive Data Analysis**
  - Upload CSV and Excel datasets.
  - Browse, filter, and explore data.
  - View descriptive statistics instantly.

- 🤖 **AI Data Analyst**
  - Ask questions in natural language.
  - Generate insights automatically.
  - AI-generated Python and SQL code.

- 📈 **Visualization**
  - Interactive Plotly charts.
  - Automatic histograms and bar charts.
  - Data distribution analysis.

- 🧹 **Data Quality Analysis**
  - Missing value detection.
  - Duplicate row detection.
  - Constant column identification.
  - Data quality scoring.

- 🗄️ **SQLite Integration**
  - Connect to SQLite databases.
  - Browse database tables.
  - Load SQL tables directly into Pandas.

- 💬 **Conversation History**
  - Persistent chat session.
  - Search previous conversations.
  - Export chats as Markdown.

- 📥 **Export Support**
  - Download filtered datasets.
  - Export AI conversation history.

- 🎨 **Modern UI**
  - Glassmorphism design.
  - Neon futuristic theme.
  - Responsive layout.

---

# 📸 Preview

> Add screenshots here

```
assets/
├── home.png
├── overview.png
├── explore.png
└── chat.png
```

---

# 🏗️ Project Structure

```
DataPilot-AI/
│
├── app.py
├── agent.py
├── graph.py
├── config.py
│
├── database/
│   └── sqlite_manager.py
│
├── tools/
│   ├── dataset_loader.py
│   └── pandas_tool.py
│
├── utils/
│   ├── helpers.py
│   └── logger.py
│
├── assets/
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## Clone the repository

```bash
git clone https://github.com/yourusername/DataPilot-AI.git

cd DataPilot-AI
```

## Create Virtual Environment

```bash
python -m venv venv
```

Activate it

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

The application will start at

```
http://localhost:8501
```

---

# 📂 Supported Data Sources

- CSV
- Excel (.xlsx)
- SQLite Database
- Sample Dataset

---

# 🛠️ Tech Stack

| Technology | Purpose |
|------------|----------|
| Python | Backend |
| Streamlit | Web Application |
| Pandas | Data Processing |
| NumPy | Numerical Computing |
| Plotly | Interactive Visualization |
| SQLite | Database |
| AI Agent Graph | Natural Language Analytics |

---

# 💡 Example Questions

You can ask the AI:

- Give me a summary of this dataset.
- What are the top trends?
- Show missing values.
- Which category has the highest revenue?
- Create a histogram of sales.
- Show correlations between numeric columns.
- Detect duplicate records.
- Which region performs the best?
- Generate SQL for this analysis.
- Visualize monthly revenue.

---

# 📊 Data Quality Features

The application automatically checks for:

- ✅ Missing values
- ✅ Duplicate rows
- ✅ Constant columns
- ✅ High-cardinality columns
- ✅ Dataset health score

---

# 🚀 Future Improvements

- [ ] PDF report generation
- [ ] Multi-database support
- [ ] PostgreSQL integration
- [ ] MySQL integration
- [ ] Dark/Light themes
- [ ] User authentication
- [ ] Dashboard builder
- [ ] Scheduled reports
- [ ] Machine Learning models
- [ ] LLM provider selection

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create your feature branch

```bash
git checkout -b feature/NewFeature
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push

```bash
git push origin feature/NewFeature
```

5. Open a Pull Request

---

# 📄 License

This project is licensed under the **MIT License**.

---

# 👨‍💻 Author

**Your Name**

- GitHub: https://github.com/yourusername
- LinkedIn: https://linkedin.com/in/yourprofile

---

## ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub!
