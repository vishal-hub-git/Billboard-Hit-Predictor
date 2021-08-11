from flask import Flask, request, render_template
import pickle
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("final_data1.csv")
df_X=data.drop(['billboard_hit','Track','Artist','SpotifyID'],axis=1)
df_Y=data['billboard_hit']
model = pickle.load(open('forest_model.pkl', 'rb'))
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(df_X, df_Y, test_size=0.2, random_state=101)
forestVC=RandomForestClassifier(random_state=1,n_estimators=950,max_depth=20,min_samples_split=5,min_samples_leaf = 1,max_features="sqrt")
forestVC.fit(X_train, Y_train)
y_predVC2=forestVC.predict(X_test)

app = Flask(__name__)


@app.route('/',methods=['GET'])
def Home():
    return render_template('hit_predictor.html')


@app.route("/predict_hit", methods=['POST'])
def predict_hit():
    if request.method == 'POST':
        song_name = request.form['song_name']
        danceability = float(request.form['danceability'])
        energy = float(request.form['energy'])
        loudness = float(request.form['loudness'])
        speechiness = float(request.form['speechiness'])
        Key = float(request.form['Key'])
        duration = int(request.form['duration'])
        acousticness = float(request.form['acousticness'])
        instrumentalness = float(request.form['instrumentalness'])
        liveness = float(request.form['liveness'])
        valence = float(request.form['valence'])
        tempo = float(request.form['tempo'])
        mode = float(request.form['mode'])
        duration_ms = duration*1000
        duration_ms=np.log(duration_ms)
        if mode==0:
            Key_mode_ratio=0
        else:
            Key_mode_ratio=float(Key/mode)
        pred=forestVC.predict([[danceability,energy,loudness,speechiness,acousticness,instrumentalness,liveness,valence,tempo,Key_mode_ratio,duration_ms]])[0]
        print(pred)
        if(pred==1):
            return render_template('Result.html', hit='The song will be a Billboard hit.', not_hit='', song_name=song_name)
        else:
            return render_template('Result.html', hit='', not_hit='The song will not be a Billboard hit.', song_name=song_name)

    else:
        return render_template('Result.html')


if __name__ == '__main__':
    app.run(debug=True)