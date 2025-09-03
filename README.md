# Email-spam-classification
# 📧 Email Spam Classifier  

An interactive **Email Spam Detection System** built with **Machine Learning** and deployed on **Hugging Face Spaces** using **Streamlit**.  
This project classifies emails as **Spam** or **Ham (Not Spam)** using text preprocessing and a trained ML model.  

---

## 🚀 Live Demo  
👉 [Try it on Hugging Face Spaces](https://huggingface.co/spaces/qShihab/SMS-Email-Spam-Classification)  

---
(**Must**) Please download the model.pkl file from below link : 
- You can download those files from the Google Drive link below.
  Download the files from [here](https://drive.google.com/file/d/1wSsVq__jCA2QYKBkg-TqCSYtgXK5tj0j/view?usp=sharing)  
## 📌 Features  
- ✅ Detects whether an email is **Spam** or **Ham**  
- ✅ Preprocessing pipeline:  
  - Remove punctuation & numbers  
  - Tokenization  
  - Stopword removal (NLTK)  
  - Stemming (PorterStemmer)  
- ✅ **TF-IDF Vectorization** for feature extraction  
- ✅ Multiple ML models tested → Final selected model (e.g., **Multinomial Naive Bayes / SVM / XGBoost**)  
- ✅ User-friendly **Streamlit interface**  
- ✅ One-click deployment on **Hugging Face Spaces**  

---

## 🛠️ Technologies Used  
- **Programming Language:** Python 3  
- **Libraries & Tools:**  
  - Scikit-learn → ML model training & evaluation  
  - NLTK → Natural Language Processing (stopwords, stemming)  
  - Pandas & NumPy → Data handling  
  - Streamlit → Web app framework  
  - Pickle → Model & vectorizer serialization  
  - Hugging Face Spaces → Cloud deployment  

## 📊 Dataset  
- Dataset used: **SMS Spam Collection Dataset** (UCI Machine Learning Repository / Kaggle)  
- Contains **5,574 messages** labeled as *Spam* or *Ham*  
- Class distribution:  
  - **Ham (Not Spam): ~87%**  
  - **Spam: ~13%**  

---

---

## 📷 Screenshots  

### 🔹 Home Page  
 <img width="1918" height="853" alt="Screenshot 2025-09-03 220556" src="https://github.com/user-attachments/assets/d21c07a7-7b46-449f-812c-04f6f2c4c7b9" />


### 🔹 Prediction Example  
<img width="1910" height="829" alt="Screenshot 2025-09-03 230657" src="https://github.com/user-attachments/assets/2f036770-5bad-446a-929a-99540121b570" />

## 🔄 Workflow  

1. **Data Collection** → Load dataset from CSV  
2. **Preprocessing** →  
   - Lowercasing  
   - Removing punctuation/numbers  
   - Tokenization  
   - Stopword removal  
   - Stemming  
3. **Feature Extraction** → TF-IDF vectorization  
4. **Model Training** → Naive Bayes, SVM, and XGBoost tested  
5. **Model Evaluation** → Accuracy, Precision, Recall, F1-score  
6. **Deployment** → Model + vectorizer saved with `pickle` and integrated into **Streamlit app**  
7. **Hosting** → Deployed to **Hugging Face Spaces**
