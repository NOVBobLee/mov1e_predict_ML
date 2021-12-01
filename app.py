from flask import Flask, render_template, request, url_for
import numpy as np
import pandas as pd
import pickle
from scipy.stats import boxcox
from scipy.special import inv_boxcox


app = Flask(
    __name__,
    static_url_path="/",
    template_folder='template'
    )


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        # transform filled form into preferable type
        iscollection = int(request.values['movie_collection'])
        budget_np = np.fromstring(request.values['movie_budget'] + ' 2', dtype=int, sep=' ')
        budget, _ = boxcox(budget_np)[0]

        runtime1 = np.log1p(int(request.values['movie_runtime']))
        gg_winner = int(request.values['movie_goldenglobe_winner'])
        gg_nominee = int(request.values['movie_goldenglobe_nominee'])
        os_winner = int(request.values['movie_oscar_winner'])
        os_nominee = int(request.values['movie_oscar_nominee'])

        dt_dict = {
            'collection': [iscollection],
            'month': [9],
            'weekday': [3],
            'budget': [budget],
            'runtime': [runtime1],
            'gg_actor_w': [gg_winner],
            'gg_actor_n': [gg_nominee],
            'osc_actor_w': [os_winner],
            'osc_actor_n': [os_nominee],
            'Action': [0],
            'Adventure': [0],
            'Animation': [0],
            'Comedy': [0],
            'Crime': [0],
            'Documentary': [0],
            'Drama': [0],
            'Family': [0],
            'Fantasy': [0],
            'History': [0],
            'Horror': [0],
            'Music': [0],
            'Mystery': [0],
            'Romance': [0],
            'Science Fiction': [0],
            'Thriller': [0],
            'TV Movie': [0],
            'War': [0],
            'Western': [0]
        }

        genreBooleanSwitcher(dt_dict, request.values['movie_genre1'])
        genreBooleanSwitcher(dt_dict, request.values['movie_genre2'])
        genreBooleanSwitcher(dt_dict, request.values['movie_genre3'])

        # predict
        dt = pd.DataFrame.from_dict(dt_dict)
        model = pickle.load(open('xgb29.mdl', 'rb'))
        y_boxcox = model.predict(dt)
        y_pred = inv_boxcox(y_boxcox, .157)
        y_pred_round = str(round(y_pred[0], 0))

        return render_template('page2.html', y_pred=y_pred_round)


def genreBooleanSwitcher(dt_dict, genre):
    if genre == "" or genre == "null":
        return
    else:
        dt_dict[genre] = [1]


if __name__ == "__main__":
    app.run()

