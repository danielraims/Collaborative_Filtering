### First you need to save the sparse dataset 

get_sparse('ratings.csv', 'results) for example will create the file sparse matrices, and all the map functions and save our data in file sparse 'movie_sparse.joblib'

### We can load our data using 

outputs = load_all_data('results') ### replace this with your path, this will return a dictionary 

### To train our model, we need all data saved previously,

map_user_to_index,map_index_to_user,map_movie_to_index,map_index_to_movie,\
data_by_user_train,data_by_movie_train,data_by_user_test,data_by_movie_test = outputs.values()

### Training

model = CollaborativeFilteringModel() ### this will load data with default parameters
model.train_model(data_train_by_user, data_train_by_movie, data_test_by_user, data_test_by_movie)

### You can plot netrics

model.plot_loss_rmse()

### After all of this, you can plot recommendation using:
movie_ids_and_colors = [1, 1997, 858, 89745, 142115]
plot_recommendations(path_movie, movie_ids_and_colors, model,'', map_index_to_movie, map_movie_to_index)
