from app import app, socketio
from flask import *

from rankings import get_top, restaurant_to_index

app = Flask(__name__, template_folder='app/templates')

# get user input
@app.route("/")
def query():
  data = []
  output_message = ''

  restaurant_query = request.args.get('fav_name')
  # if there is an input
  if restaurant_query:
    # if restaurant_query is in the data
    if restaurant_query in restaurant_to_index.keys():
      top_restaurants = get_top(restaurant_query)
      top_names = []
      for restaurant in top_restaurants[:3]:
        name = restaurant[0]
        top_names.append(name)
      output_message = "Your search: " + restaurant_query
      data = top_names
    # restaurant_query is not in the data
    else:
      output_message = "Your search " + restaurant_query + " is not in the dataset. Please try another restaurant"
  return render_template('search.html', output_message=output_message, data=data)

if __name__ == "__main__":
  print("Flask app running at http://0.0.0.0:5000")
  socketio.run(app, host="0.0.0.0", port=5000)
