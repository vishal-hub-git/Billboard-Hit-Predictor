from flask import Flask, request, render_template
import pickle
import numpy as np

model = pickle.load(open('forest_model1.pkl', 'rb'))

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
        pred=model.predict([[danceability,energy,loudness,speechiness,acousticness,instrumentalness,liveness,valence,tempo,Key_mode_ratio,duration_ms]])[0]
        print(pred)
        if(pred==1):
            return render_template('Result.html', hit='The song will be a Billboard hit.', not_hit='', song_name=song_name)
        else:
            return render_template('Result.html', hit='', not_hit='The song will not be a Billboard hit.', song_name=song_name)

    else:
        return render_template('Result.html')


if __name__ == '__main__':
    app.run(debug=True)
