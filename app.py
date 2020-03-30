from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib


scaler=joblib.load("./Models/scaler.model")
model=joblib.load("./Models/clf.model")

def newFeatures(data):
    cols=['G<105','P<4','BP<48','ST<12','IN<50','BMI<21','G<125_IN<125','G<105_AGE<30','G<105_BMI<30','G<100_BP<48','ST<20_BMI<30']
    data_new=pd.DataFrame(np.zeros((data.shape[0],len(cols))), columns=cols)
    for i in range(data_new.shape[0]):
        if data['Glucose'][i]<105:
            data_new['G<105'][i]=1
        
        if data['Pregnancies'][i]<4:
            data_new['P<4'][i]=1
        if data['BloodPressure'][i]<48:
            data_new['BP<48'][i]=1
        if data['SkinThickness'][i]<12:
            data_new['ST<12'][i]=1
        if data['Insulin'][i]<50:
            data_new['IN<50'][i]=1
        if data['BMI'][i]<21:
            data_new['BMI<21'][i]=1
        if data['Glucose'][i]<125 and data['Insulin'][i]<125:
            data_new['G<125_IN<125'][i]=1
        if data['Glucose'][i]<105 and data['Age'][i]<30:
            data_new['G<105_AGE<30'][i]=1
        if data['Glucose'][i]<105 and data['BMI'][i]<30:
            data_new['G<105_BMI<30'][i]=1
        if data['Glucose'][i]<100 and data['BloodPressure'][i]<48:
            data_new['G<100_BP<48'][i]=1
        if data['SkinThickness'][i]<20 and data['BMI'][i]<30:
            data_new['ST<20_BMI<30'][i]=1
            
    data_new['BMI*ST']=data['BMI']*data['SkinThickness']       
    data_new['G/IN']=data['Glucose']/data['Insulin']
    data_new['P/A']=data['Pregnancies']/data['Age']
    
    return (pd.concat([data_new,data],axis=1)).values

columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
def transform(data):
	data=np.array(data)
	data=newFeatures(pd.DataFrame(data, columns=columns))
	data=scaler.transform(data)
	return data







app = Flask(__name__)

@app.route('/')
@app.route('/home')
def hello_world():
    return render_template('Templates/home.html')





@app.route('/predict', methods=['POST'])
def predict():
	try:
		pn=str()
		pn=str(request.form.get('pname'))
		data=[]

		for name in columns:
			try:
				data.append(float(request.form.get(name, 0.0)))
			except:
				data.append(0.0)

		print(data)
		data=transform([data])
		op=model.predict(data)
		pred=str()
		if op[0]==0:
			pred="Non-Diabetic"
		else:
			pred="Diabetic"

		return render_template('Templates/home.html', pred_text=pred, n=pn+" seems to be ")
	except:
		return render_template('Templates/home.html')




if __name__ == '__main__':
    app.run(debug=True)