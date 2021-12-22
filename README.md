# sentiment-analyzer

This is a web application which is used to analyze the sentiment of customer reviews. First the text is tokenized into sentence level. Then a trained distilbert model is applied to each sentence. The model classifies the sentences into three classes; Positive, Negative and Neutral. A percentage score is shown for each class. Positive, negative and neutral sentences are highlighted respectively with green, red and yellow colors. The distilbert model is in onnx format. Flask library is used to build the web server. In addition, this is hosted on heroku.

Demo: <a href="https://sentiment-analyzer-trehx.herokuapp.com" target="_blank">https://sentiment-analyzer-trehx.herokuapp.com</a>

<p align="center">
  <img src="https://github.com/tharakarehan/sentiment-analyzer/blob/main/ss.png">
</p>
