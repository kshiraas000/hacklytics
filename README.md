# **Cryptocurrency Pump-and-Dump Scheme Detection System**

## **Overview**
This project aims to address one of the most pressing challenges in the cryptocurrency market: **pump-and-dump schemes**. These fraudulent activities manipulate cryptocurrency prices, causing significant financial losses for investors. Our solution leverages **machine learning** and **real-time data analysis** to detect and predict these schemes before they reach their devastating conclusion.

---

## **Team Members**
- **Javad**
- **Khalil**
- **Aashna**
- **Phuc**

We are final-year Computer Science students specializing in **Machine Learning** and **Financial Technology**.

---

## **Problem Statement**
Cryptocurrency markets are highly susceptible to manipulation, with **25% of all new crypto tokens in 2022** created specifically for pump-and-dump schemes. These schemes exploit psychological factors like **FOMO (Fear of Missing Out)**, **fake social proof**, and **greed**, leading to massive financial losses for investors. Recent examples, such as the **Argentina cryptocurrency scandal**, highlight the urgent need for a solution.

---

## **Solution**
We developed a **neural network-based system** that detects pump-and-dump schemes by analyzing **social media activity** and **market capitalization data**. Our system identifies suspicious patterns and provides early warnings to regulators or investors.

### **Key Features**
1. **Real-Time Social Media Analysis**:
   - Utilizes the **LunarCrush API** to monitor platforms like **Reddit**, **X (Twitter)**, **YouTube**, and **TikTok**.
   - Tracks engagement metrics, post frequency, and sentiment analysis.
   
2. **Market Data Integration**:
   - Correlates social media data with **CoinMarketCap** market capitalization data.
   - Identifies unusual price movements and trading volumes.

3. **Neural Network Model**:
   - Trained on historical pump-and-dump cases to recognize patterns.
   - Detects subtle indicators of manipulation, such as coordinated social media campaigns and sudden price spikes.

4. **Real-World Application**:
   - Successfully identified the **Valor token** as a potential pump-and-dump scheme days before significant price movements.
   - Provides actionable insights for **regulatory authorities** or **investors**.

---

## **Technical Implementation**

### **Data Sources**
1. **LunarCrush API**:
   - Provides social media engagement metrics for cryptocurrencies.
   - Tracks mentions, sentiment, and influencer activity.

2. **CoinMarketCap API**:
   - Supplies real-time market data, including price, volume, and market capitalization.

### **Neural Network Architecture**
- **Input Layer**:
  - Social media engagement metrics (e.g., post frequency, sentiment score).
  - Market data (e.g., price, volume, market cap).
  
- **Hidden Layers**:
  - Multiple dense layers with **ReLU activation** for feature extraction.
  - Dropout layers to prevent overfitting.

- **Output Layer**:
  - Binary classification (pump-and-dump scheme or legitimate activity).
  - Sigmoid activation function for probability output.

- **Training**:
  - Dataset: Historical data of known pump-and-dump schemes.
  - Loss Function: Binary cross-entropy.
  - Optimizer: Adam.

### **System Workflow**
1. **Data Collection**:
   - Fetch social media and market data using APIs.
   
2. **Preprocessing**:
   - Normalize and clean data.
   - Extract relevant features (e.g., sentiment score, price change rate).

3. **Prediction**:
   - Feed processed data into the neural network.
   - Generate predictions and confidence scores.

4. **Alert System**:
   - Trigger alerts for suspicious activity.
   - Provide detailed reports for regulators or investors.

---

## **Potential Applications**
1. **Regulatory Use**:
   - Alert authorities about potential fraud, insider trading, or money laundering.
   - Help maintain market integrity and protect investors.

2. **Investment Strategy**:
   - Enable short-selling opportunities before market crashes.
   - Provide insights for risk-averse investors.

---

## **Future Work**
- Expand data sources to include additional platforms (e.g., Telegram, Discord).
- Improve model accuracy by incorporating more historical data.
- Develop a user-friendly dashboard for real-time monitoring.

---

## **How to Use**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/crypto-pump-dump-detection.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the system:
   ```bash
   python main.py
   ```

---

## **Contributing**
We welcome contributions! Please fork the repository and submit a pull request with your changes.

---

## **Contact**
For questions or collaborations, please contact:  
**Email**: kzina6@gatech.edu  
**GitHub**: https://github.com/fantabnina
