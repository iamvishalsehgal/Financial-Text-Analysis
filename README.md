# NLP Assignments: Stock Price Prediction and Topic Modeling

This repository contains the code and documentation for two assignments from the Natural Language Processing. Assignment 1 focuses on predicting stock price movements using text classification, while Assignment 2 explores topic modeling on financial news articles using BERTopic, LDA, and FLSA-W.

---

## Assignment 1: Stock Price Movement Prediction

### Objective
The goal of this assignment is to build a text classification model that predicts whether Nvidia's stock price will go up (1) or down (0) based on the content of newspaper articles mentioning Nvidia.

### Approach
1. **Dataset Creation**: 
   - Filter articles containing the keyword "nvidia".
   - Use historical stock price data from Yahoo Finance to label each article based on whether the stock price increased or decreased the next day.

2. **Preprocessing**:
   - Convert text to lowercase.
   - Remove punctuation except for financially relevant symbols like `$` and `%`.
   - Remove stop words and lemmatize words to reduce noise.
   - Filter out very short words and words that are not meaningful in the context.

3. **Feature Extraction**:
   - Use TF-IDF (Term Frequency-Inverse Document Frequency) to represent the text data.
   - Experiment with Word2Vec to capture semantic relationships between words.

4. **Modeling**:
   - Train a Logistic Regression classifier on both TF-IDF and Word2Vec features.
   - Evaluate the models using accuracy, precision, recall, and F1-score.

5. **Analysis**:
   - Identify the 50 most common words in the corpus.
   - Determine the 20 most indicative words for each class (stock up or down).
   - Examine the distribution of labels to check for class imbalance.

### How to Run
1. **Dataset**: Is in `Assignment-1/Dataset/assignment-2-data.csv`
2. **Dependencies**: Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Script**:
   ```
   Assignment 1 : `Assignment-1/Dataset/code.ipynb`
   ```

### Key Findings
- The TF-IDF + Logistic Regression model achieved an accuracy of X% (replace with actual value).
- Word2Vec provided slightly better/worse performance due to its ability to capture semantic meaning.
- The most indicative words for stock price increases included terms like "growth," "innovation," and "earnings beat."

---

## Assignment 2: Topic Modeling on Financial News Articles

### Objective
This assignment aims to apply and compare three topic modeling techniques—BERTopic, LDA (Latent Dirichlet Allocation), and FLSA-W (Fuzzy Latent Semantic Analysis with Word Embeddings)—to extract meaningful topics from financial news articles related to Nvidia. The topics are analyzed for their coherence, diversity, and relevance to stock price movements.

### Methodologies
- **BERTopic**: A transformer-based model that uses BERT embeddings, UMAP for dimensionality reduction, and HDBSCAN for clustering. It excels at generating contextually rich topics.
- **LDA**: A probabilistic model that assumes documents are mixtures of topics and topics are distributions over words. It is widely used for its interpretability.
- **FLSA-W**: A fuzzy clustering approach that incorporates word embeddings to handle polysemy and synonymy, making it suitable for financial texts with complex terminology.

### Experimental Setup
1. **Preprocessing**:
   - Remove stop words, punctuation, and irrelevant terms (e.g., days of the week).
   - Apply custom stop words to filter out common financial jargon.

2. **Model Application**:
   - Each model is run with varying numbers of topics (e.g., 10, 20, 30) to find the optimal configuration.

3. **Evaluation**:
   - **Coherence Score (C_V)**: Measures the semantic similarity of words within topics.
   - **Diversity Score**: Ensures topics are distinct by measuring the uniqueness of words across topics.

4. **Visualization**:
   - Word clouds for the most frequent words.
   - Topic distribution plots.
   - Intertopic distance maps and similarity matrices for BERTopic.

### Key Findings
- **BERTopic** consistently achieved higher coherence scores, indicating more interpretable topics.
- **LDA** performed well with moderate topic numbers but struggled with complex financial terms.
- **FLSA-W** provided better diversity, capturing a broader range of themes, though with slightly lower coherence.

These results suggest that BERTopic is particularly effective for financial news, while FLSA-W offers a complementary approach for diverse topic coverage.

### How to Run
1. **Dataset**: Is in `Assignment-1/Dataset/assignment-2-data.csv`.
2. **Dependencies**: Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Scripts**:
   - For BERTopic:
     ```
     Assignment-2/BertTopic.ipynb
     ```
   - For LDA:
     ```
     Assignment-2/LDA.ipynb
     ```
   - For FLSA-W:
     ```
     Assignment-2/FLSA-W.ipynb
     ```
