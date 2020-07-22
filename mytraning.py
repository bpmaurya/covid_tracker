import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression



def data_split(data,ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data)* ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]


if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    train,test= data_split(df,0.2)
    x_train = train[['fever','bodyPain','age','runnyNose','diffBreath']].to_numpy()
    x_test = test[['fever','bodyPain','age','runnyNose','diffBreath']].to_numpy()

    y_train = train[['infectionProb']].to_numpy().reshape(120,)
    y_test = test[['infectionProb']].to_numpy().reshape(30,)
    
    clf = LogisticRegression()
    clf.fit(x_train,y_train)
    
    inputFeatures =[100,1,22,1,1]
    infoProb=clf.predict_proba([inputFeatures])[0][1]
    
    dbfile = open('model.pkl', 'wb') 
    # source, destination 
    pickle.dump(clf, dbfile)                      
    dbfile.close() 
  