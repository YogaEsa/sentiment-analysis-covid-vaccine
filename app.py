# In[ ]:

from flask import Flask, render_template, url_for, request

# ML PACKAGES

# In[ ]:


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/klasifikasi')
def klasifikasi():
    return render_template('klasifikasi.html')


@app.route('/data')
def data():
    return render_template('data.html')


if __name__ == '__main__':
    app.run(debug=True)
