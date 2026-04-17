@app.route('/search', methods=['POST'])
# def search():
#     query = request.form['movie']   # get input from user

#     # search movie (case insensitive)
#     results = data[data['Movie Name'].str.contains(query, case=False)]

#     return render_template('index.html', tables=[results.to_html()])