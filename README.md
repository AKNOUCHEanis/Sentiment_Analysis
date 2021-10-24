# SentimentAnalysis
Analyse de sentiments de Tweets de la US Airline :

**Avec CountVectorizer:**

MultinomialNB : 78%  Test accuracy, F1 score : Negative:86%     Neutral: 52%  Positive:69%

RegLog            : 80% Test  accuracy, F1 score : Negative:88%    Neutral: 59%   Positive:71%

**Avec Word2Vec  :**

RegLog            : 63% Test accuracy, F1 score : Negative:77%     Neutral:1%   Positive:13%

**Avec un modèle pré-entrainé :** gensim ‘glove-twitter-25’ (Word2Vec)

RegLog            : 65% Test accuracy, F1 score : Negative: 79%    Neutral:15%   Positive: 15%

**Environnement :** Python, ScikitLearn, Gensim, NLTK, Numpy, Pandas
