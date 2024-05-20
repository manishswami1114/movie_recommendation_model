from flask import Flask, render_template, request
from imdb import IMDb
from wikipedia import summary, exceptions

app = Flask(__name__)
ia = IMDb()
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/suggest', methods=['POST'])
def suggest():
    movie_name = request.form['movie_name']
    try:
        movie = ia.search_movie(movie_name)[0]
        movie_id = movie.movieID
        movie_info = ia.get_movie(movie_id)
        rating = movie_info['rating']
        summary_text = summary(movie_name, sentences=2)
        if rating > 7.0:
            watchable = "Yes, this movie is watchable!"
        else:
            watchable = "No, this movie is not watchable."
        return render_template('result.html', movie_name=movie_name, rating=rating, summary_text=summary_text, watchable=watchable)
    except exceptions.DisambiguationError as e:
        return "Disambiguation error: " + str(e)
    except exceptions.PageError as e:
        return "Page error: " + str(e)

@app.route('/return')
def return_home():
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)