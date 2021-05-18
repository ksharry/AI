
from flask import Flask
from flask import request
from flask import jsonify
import datetime
import hashlib
import numpy as np
import pandas as pd

app = Flask(__name__)
####### PUT YOUR INFORMATION HERE #######
CAPTAIN_EMAIL = 'ksharry1025@gmail.com'          #
SALT = 'my_salt_ESAI'                        #
#########################################

def start_model(article1):
    sentence = article1

    ws_results = ws([sentence])

    for word in ws_results:
        if word in stops:
            sent2 = "".join('')
        else:
            sent2 = " ".join(word)
          
    regex = re.compile('([^\s\w]|_)+')
    sent2 = regex.sub('', sent2)
    sent2 = ' '.join(sent2.split())

    # 將詞彙序列轉為索引數字的序列
    x_test = tokenizer.texts_to_sequences([sent2])

    # 為數字序列加入 zero padding
    x_test = keras.preprocessing.sequence.pad_sequences(x_test,maxlen=MAX_SEQUENCE_LENGTH)

    # 利用已訓練的模型做預測
    global graph
    with graph.as_default():
        predictions99 = model.predict([x_test])
        keynum = np.argmax(predictions99[0])

        keyman = set({})
        if keynum == 1:
            pos_results = pos(ws_results)
            ner_results = ner(ws_results, pos_results)
            for name in ner_results[0]:
                if name[2]=='PERSON':
                    if bool(re.match(r'[\u4E00-\u9FFF]{2,4}', name[3])):
                        if name[3][0] in fn[0].values:
                            keyman.add(name[3])
    return keyman

def generate_server_uuid(input_string):
    """ Create your own server_uuid
    @param input_string (str): information to be encoded as server_uuid
    @returns server_uuid (str): your unique server_uuid
    """
    s = hashlib.sha256()
    data = (input_string+SALT).encode("utf-8")
    s.update(data)
    server_uuid = s.hexdigest()
    return server_uuid

def predict(article):
    """ Predict your model result
    @param article (str): a news article
    @returns prediction (list): a list of name
    """
    key = start_model(article)  
    key1 = list(key)

    #!/usr/bin/python
     
    # 開啟檔案
    fp = open("200806.txt", "a")

    # 將 lines 所有內容寫入到檔案
    cnt = 9999
    aaa = str(cnt) + "," + article + "," + str(key1) + '\n'
    fp.writelines(aaa)
      
    # 關閉檔案
    fp.close()
    ####### PUT YOUR MODEL INFERENCING CODE HERE #######
    prediction = key1
    
    
    ####################################################
    prediction = _check_datatype_to_list(prediction)
    return prediction

def _check_datatype_to_list(prediction):
    """ Check if your prediction is in list type or not. 
        And then convert your prediction to list type or raise error.
        
    @param prediction (list / numpy array / pandas DataFrame): your prediction
    @returns prediction (list): your prediction in list type
    """
    if isinstance(prediction, np.ndarray):
        _check_datatype_to_list(prediction.tolist())
    elif isinstance(prediction, pd.core.frame.DataFrame):
        _check_datatype_to_list(prediction.values)
    elif isinstance(prediction, list):
        return prediction
    raise ValueError('Prediction is not in list type.')

@app.route('/healthcheck', methods=['POST'])
def healthcheck():
    """ API for health check """
    data = request.get_json(force=True)  
    t = datetime.datetime.now()  
    ts = str(int(t.utcnow().timestamp()))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL+ts)
    server_timestamp = t.strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({'esun_uuid': data['esun_uuid'], 'server_uuid': server_uuid, 'captain_email': CAPTAIN_EMAIL, 'server_timestamp': server_timestamp})

@app.route('/inference', methods=['POST'])
def inference():
    """ API that return your model predictions when E.SUN calls this API """
    data = request.get_json(force=True)  
    esun_timestamp = data['esun_timestamp'] #自行取用
    
    t = datetime.datetime.now()  
    ts = str(int(t.utcnow().timestamp()))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL+ts)
    
    try:
        answer = predict(data['news'])
    except:
        raise ValueError('Model error.')        
    server_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({'esun_timestamp': data['esun_timestamp'], 'server_uuid': server_uuid, 'answer': answer, 'server_timestamp': server_timestamp, 'esun_uuid': data['esun_uuid']})

if __name__ == "__main__": 
    import tensorflow as tf
    import re
    from sklearn.model_selection import train_test_split
    from tensorflow.python import keras
    from tensorflow.python.keras import Input
    from tensorflow.python.keras.layers import Embedding,LSTM, concatenate, Dense
    from tensorflow.python.keras.models import Model
    from ckiptagger import construct_dictionary, WS, POS, NER
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    datapath = '/home/user072/anaconda3/envs/py36/esai001/'

    #載入停用字
    with open(datapath+'stopword.txt','r',encoding='utf-8-sig') as file:
        stops = file.read().split('\n')
    #載入姓氏
    fn = pd.read_csv(datapath+'CFN_TN.txt', sep=" ",header=None)  

    MAX_NUM_WORDS = 10000
    VALIDATION_RATIO = 0.1 
    RANDOM_STATE = 9527   # 小彩蛋
    NUM_CLASSES = 2      # 基本參數設置，有幾個分類
    MAX_NUM_WORDS = 10000   # 在語料庫裡有多少詞彙
    MAX_SEQUENCE_LENGTH = 250 # 一個標題最長有幾個詞彙
    NUM_EMBEDDING_DIM = 256  # 一個詞向量的維度
    NUM_LSTM_UNITS = 128   # LSTM 輸出的向量維度
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=MAX_NUM_WORDS)

    df = pd.read_excel(datapath+'tnse_data.xlsx', encoding='utf-8' )
    corpus = df.content
    corpus = pd.concat([corpus])
    corpus.shape
    pd.DataFrame(corpus.iloc[:5],
                columns=['title'])
    tokenizer.fit_on_texts(corpus)

    x_train = tokenizer.texts_to_sequences(corpus)

    model = keras.models.load_model(datapath + '/modellstm_10.h5')
    graph = tf.get_default_graph()

    ws = WS(datapath+"data")    #斷詞
    pos = POS(datapath+"data")   #詞性標記
    ner = NER(datapath+"data")   #命名實體識別

    app.run(host='0.0.0.0', port=80, debug=False, threaded=False)
