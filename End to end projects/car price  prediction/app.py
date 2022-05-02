import pickle
from flask import Flask, render_template, request, jsonify
import sklearn

app = Flask(__name__)

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def results():
    fueltype = float(request.form['fueltype'])
    aspiration = float(request.form['aspiration'])
    doornumber = float(request.form['doornumber'])
    carbody = float(request.form['carbody'])
    drivewheel = float(request.form['drivewheel'])
    enginelocation = float(request.form['enginelocation'])
    wheelbase = float(request.form['wheelbase'])
    carlength = float(request.form['carlength'])
    carwidth = float(request.form['carwidth'])
    carheight = float(request.form['carheight'])
    curbweight = float(request.form['curbweight'])
    enginetype = float(request.form['enginetype'])
    cylindernumber = float(request.form['cylindernumber'])
    enginesize=float(request.form['enginesize'])
    fuelsystem= float(request.form['fuelsystem'])
    boreratio=float(request.form['boreratio'])
    stroke=float(request.form['stroke'])
    compressionratio=float(request.form['compressionratio'])
    horsepower=float(request.form['horsepower'])
    peakrpm=float(request.form['peakrpm'])
    citympg=float(request.form['citympg'])
    highwaympg=float(request.form['highwaympg'])

    X = np.array([[fueltype,aspiration,doornumber,carbody,drivewheel,enginelocation,wheelbase,carlength,carwidth,carheight,
    curbweight,enginetype,cylindernumber,enginesize,fuelsystem,boreratio,stroke,compressionratio,horsepower,peakrpm,citympg
    ,highwaympg]])

    model = pickle.load(open('car_model.pkl','rb'))
    Y_predict = model.predict(X)
    return jsonify({'Prediction': float(Y_predict)})


if __name__ == '__main__':
    app.run(debug = True, port = 1010)
