# -*- coding: utf-8 -*-
"""
Скрипт предназначен для формирования прогноза из командной строки.
В качестве аргументов необходимо указывать:
    
    --csv
            Для указания файла, по которому будет построен прогноз
    --threshold
            Для порога бинаризации, по-умолчанию - 0.42, что даёт хорошую 
            полноту (recall), но слабую точность (precision).
            Самую лучшую f1-меру показывал порог 0.8, но давал слабую полноту.
    --save
            Имя файла, в который будет сохранён прогноз.
            
Примеры использования:
    
$ python NBKI_predict.py --csv NBKI_test.csv    

$ python NBKI_predict.py --csv NBKI_test.csv --threshold 0.8 --save y_pred.csv


"""

import argparse
import pandas as pd
import numpy as np
import joblib


def createParser ():
    
    parser = argparse.ArgumentParser()
    parser.add_argument ('--csv', required=True)
    parser.add_argument ('--threshold', default=0.42) 
    parser.add_argument ('--save', default='y_predictions.csv') 
    
    return parser
 
 
if __name__ == '__main__':
    
    parser = createParser()
    namespace = parser.parse_args()
 
    X = pd.read_csv('{}'.format (namespace.csv), index_col=[0])

    model = joblib.load('model_for_script.pkl')

    y_probas = model.predict_proba(X)
    y_pred = np.where(y_probas[:, 1] >= float(namespace.threshold), 1, 0)
    y_pred = pd.Series(y_pred, index=X.index)
    
    y_pred.to_csv(namespace.save)
    
    
    print ("Порог бинаризации: {}".format (namespace.threshold) )
    print ("Предсказания сохранены в файл {}".format (namespace.save) )