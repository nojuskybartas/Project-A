from flask import Flask, render_template, request
from model.settings import gru_bigboi3
from model.model import Model
from typing import ChainMap

app = Flask(__name__)


def pre_load(settings):
    my_model = Model(settings)
    my_model.run_inference(1, False, False)
    return my_model


model = pre_load(gru_bigboi3)

# EG: http://127.0.0.1:5000/api?timeframe=1&predict_forward=True&json_orient=split


@app.route('/api')
def index():
    default_args = {'timeframe': 1, 'predict_forward': False, 'json_orient': 'split'}
    args = ChainMap[request.args, default_args]

    df = model.run_inference(int(args['timeframe']), bool(args['predict_forward']), plot=False)
    return df.to_json(orient=args['json_orient'])
    
# NOTE: for unknown reason, if you add predict_forward=False in the args when calling the api,
# it will still return True. For now, the way to have it as False is to not include it in the args at all


if __name__ == "__main__":
    app.run(debug=True)
