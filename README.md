# 🎯 YouTube Sentiment Analyzer Chrome Extension

A powerful Machine Learning-powered Chrome Extension that enables **YouTube creators** to quickly analyze and visualize the **sentiment of comments** on their videos. Forget scrolling through thousands of comments — get clear sentiment insights, word maps, and trend analysis with a click.

> ⭐ **Best Model:** LightGBM outperforms others (Random Forest, Decision Tree, Boosting)

---

## 📌 Features

- 🎯 **Sentiment Analysis (Positive, Neutral, Negative)**  
  Automatically classifies each comment based on sentiment using trained ML models.

- 🔍 **Word Map Generation**  
  Visualizes frequently used words in the comments to understand audience focus and topics.

- 📈 **Trends Graph**  
  Graphical representation of sentiment trends to monitor viewer reaction over time.

- 🧩 **Chrome Extension Integration**  
  Easily analyze YouTube comments directly within the YouTube platform with a seamless browser extension.

---

## 🚀 How It Works

1. **ML Model Design**
   - Developed using multiple ML algorithms: LightGBM, Random Forest, Decision Tree, and Boosting.
   - After evaluation, **LightGBM** was selected for final deployment due to its superior accuracy and performance.

2. **YouTube Comment Scraping**
   - The extension scrapes comments from the current video being viewed.

3. **Preprocessing**
   - Comments are cleaned and prepared (tokenization, stop-word removal, etc.) for sentiment analysis.

4. **Prediction**
   - Comments are passed through the trained LightGBM model to get sentiment predictions.

5. **Visualization**
   - Outputs include:
     - Sentiment Summary
     - WordMap Visualization
     - Sentiment Trend Graph

6. **User Interface**
   - All outputs are embedded within the YouTube video page via the Chrome Extension interface.

---

## 🧠 ML Model Information

| Algorithm        | Performance | Notes                      |
|------------------|-------------|----------------------------|
| LightGBM         | ✅ Best     | Fast and highly accurate   |
| Random Forest    | Good        | Slower, slightly less accurate |
| Decision Tree    | Average     | Prone to overfitting       |
| Boosting         | Good        | Effective but slower       |

---

## 🧩 Chrome Extension

### Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/youtube-sentiment-extension.git
   cd youtube-sentiment-extension
````

2. Load the extension:

   * Open `chrome://extensions/`
   * Enable "Developer mode"
   * Click "Load unpacked"
   * Select the `extension/` folder

### Usage

1. Open a YouTube video.
2. Click the extension icon.
3. Click **"Analyze Comments"**
4. View sentiment insights, word cloud, and trends within seconds.

---

## 🔮 Future Plans

* 💬 **LLM Integration (Coming Soon!)**

  * Allow users to **interact with the extension** using natural language queries
  * Get **AI-generated suggestions** on how to improve video engagement, based on comments and feedback
  * Personalized content strategy tips for creators

---

## 🛠 Tech Stack

* **Languages**: Python, JavaScript, HTML, CSS
* **Libraries**: scikit-learn, LightGBM, pandas, matplotlib, wordcloud
* **Tools**: Chrome Extension API, BeautifulSoup / Selenium (for scraping), Flask (if used as backend API)

---

## 📂 Repository Structure

```
youtube-sentiment-extension/
│
├── model/
│   ├── sentiment_model.pkl
│   └── preprocess.py
│
├── extension/
│   ├── manifest.json
│   ├── popup.html
│   ├── popup.js
│   └── style.css
│
├── server/ (if applicable)
│   └── app.py
│
├── utils/
│   └── wordmap.py
│   └── trends.py
│
├── README.md
└── requirements.txt
```

---

## 📦 Installation (Dev)

1. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. (Optional) Run local Flask server (if ML model is hosted separately):

   ```bash
   python server/app.py
   ```

---

## 🤝 Contributing

Contributions are welcome! Please fork the repo and submit a pull request.

---

## 📄 License

This project is licensed under the **UN License** – meaning it is free to use, modify, and distribute with proper credit.

---

## 👤 Creator

**Shashank Rai**
M.Tech | Indian Institute of Information Technology (IIIT) Lucknow
📧 [shashankjirai4@gmail.com](mailto:shashankjirai4@gmail.com) *(Replace with your actual email)*

---

## ⭐ Show your support

If you find this project useful, consider giving it a ⭐ on GitHub!

---



