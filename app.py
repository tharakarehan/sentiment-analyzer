from flask import Flask, render_template, request
from modelclasses.Distilbert_sst5.distilbertsst5QONNX import DistilBertSST5QONNXbase
global message
global Model
global T
Model = DistilBertSST5QONNXbase('models/distilbert_sst5/ONNX/Quantized/distilbert-sst5-q.onnx',"CPUExecutionProvider")
message = ''
T = 0

app = Flask(__name__,template_folder='Templates')
app.static_folder = 'Static'

@app.route('/')
def index():
    global message
    global T
    return render_template('index.html',negativevalue=0,time=T,
                                        positivevalue=0,neutralvalue=0, outputmessage=message)

@app.route('/', methods=['POST'])
def index_post():
    global message
    global T
    global Model
    if request.method == "POST":
       
        if request.form['submit'] == 'Submit':
            message = request.form['message']
            if message.strip() == '':
                return render_template('index.html', 
                                             negativevalue=0,time=T,
                                        positivevalue=0,neutralvalue=0, outputmessage='Please type your text here')
            else:
                if (message.strip()[-1] not in ['.', '!', '?']):
                    message=message.strip()+'.'
                # try:
                neg, pos, neu, T, senlist = Model.batchpredict(message)
                return render_template('index.html', 
                                            negativevalue=round(neg*100,1),time=int(round(T*1000,0)),
                                    positivevalue=round(pos* 100, 1),
                                    neutralvalue=round(neu* 100, 1), outputmessage=message, senlist=senlist)
                # except:
                #     neg, pos, neu, T = Model.predict(message)
                #     return render_template('index.html', 
                #                                 negativevalue=round(neg*100,1),time=int(round(T*1000,0)),
                #                             positivevalue=round(pos* 100, 1),
                #                             neutralvalue=round(neu* 100, 1),outputmessage=message)
                                            
        else:
            return 'Bad Parameter'
    else:
        return 'OK'

if __name__ == '__main__':

    app.run()