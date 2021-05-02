from app import app, socketio
# import socketio
from flask import *
import string
from rankings import get_top, restaurant_to_index, get_reviews, web_scraping, restaurant_list
from userReview import filterRestaurants, computeCosine

app = Flask(__name__, template_folder='app/templates')

# get user input
@app.route("/", methods=["GET"])
def query():
  data = []
  output_message = ''
  filter_message = {"Price Point": "", "Cuisine": "", "Ambiance": "", "Ambiance Weight": "", "Review Weight": ""}

  restaurant_query = request.args.get('fav_name')
  price_query = request.args.get('max_price')
  cuisine_query = request.args.get('cuisine')
  user_review = request.args.get('user_review')

  # get ambiances
  ambiances_query = []
  ambiance_inputs = ['ambiance1', 'ambiance2', 'ambiance3', 'ambiance4', 'ambiance5', 'ambiance6', 'ambiance7', 'ambiance8']
  for ambiance_input in ambiance_inputs:
    if request.args.get(ambiance_input) != None:
      ambiances_query.append(request.args.get(ambiance_input))

  if request.args.get('weightRange') == None:
    ambiance_weight = 0.5
    review_weight = 0.5
  else:
    ambiance_weight = 1 - (float(request.args.get('weightRange')) / 100)
    review_weight = 1 - ambiance_weight

  if cuisine_query == None:
    cuisine_query = ""

  # create filter feedback message
  if len(ambiances_query) > 0:
    ambiances_query_capitalized = map(str.capitalize, ambiances_query)
    filter_message["Ambiance"] = ', '.join(ambiances_query_capitalized)
  else:
    filter_message["Ambiance"] = "None"

  if cuisine_query == "":
    filter_message["Cuisine"] = "None"
  else:
    filter_message["Cuisine"] = cuisine_query

  if price_query == None:
    filter_message["Price Point"] = "None"
  else:
    filter_message["Price Point"] = price_query.capitalize()

  filter_message["Ambiance Weight"] = round(ambiance_weight, 2)
  filter_message["Review Weight"] = round(review_weight, 2)


  print(filter_message)
  # if there is an input
  if restaurant_query:
    restaurant_query = restaurant_query

    # if restaurant_query is in the data
    if restaurant_query in restaurant_to_index.keys():
      top_tuple = get_top(restaurant_query, price_query, cuisine_query, ambiances_query, 5, review_weight, ambiance_weight, False, None)
      top_restaurants = [x[0] for x in top_tuple]
      top_sim_scores = [x[1] for x in top_tuple]
      app.logger.critical("got restaurants")
      output_message = "Your Search: " + restaurant_query
      data = web_scraping(top_restaurants, top_sim_scores, restaurant_to_index[restaurant_query])
      # create filtering message

    # restaurant_query is not in the data
    else:
      output_message = "Your search: User Review"
      review_query = request.args.get('user_review')
      rel_restaurants = filterRestaurants(price_query, cuisine_query)
      cosine_sim_restaurants = computeCosine(review_query, rel_restaurants)
      top_tuple = get_top("", price_query, cuisine_query, ambiances_query, 5, review_weight, ambiance_weight, True, cosine_sim_restaurants, review_query)
      # print(top_tuple)
      top_restaurants = [x[0] for x in top_tuple]
      top_sim_scores = [x[1] for x in top_tuple]
      app.logger.critical("got restaurants")
      output_message = "Your search: " + restaurant_query
      data = web_scraping(top_restaurants, top_sim_scores, 0)
  legend_bool = True
  if len(data) == 0:
    legend_bool = False
  return render_template('search.html', output_message=output_message, data=data, restaurant_list=restaurant_list, legend_bool=legend_bool, filter_message=filter_message)

if __name__ == "__main__":
  print("Flask app running at http://0.0.0.0:5000")
  socketio.run(app, host="0.0.0.0", port=5000)
