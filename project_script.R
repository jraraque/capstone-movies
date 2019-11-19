################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


#####################################
# Create train and test sets from edx 
#####################################

set.seed(1999)
test_index <- createDataPartition(y=edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temp_test_set <- edx[test_index,]
# Make sure userId and movieId in test set are also in train set
test_set <- temp_test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")
# Add rows removed from test set back into train set, adn remove temporary files
removed <- anti_join(temp_test_set, test_set)
train_set <- rbind(train_set, removed)
rm(test_index, temp_test_set, removed)


#####################################
# Regularization and cross-validation
#####################################

# model calculations, based on regularization with movie and user bias, and penalty factor lambda See textbook 33.9.3

# set sequence of lambdas for cross-validation and search for optimal penalty
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)

  #use initial approximation formula to calculate movie bias, excluding user bias terms
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  # calculation of optimal user and movie bias, using 3 iterations to approximate asnwer
  for (iteration in 1:3){  
  # calculate user bias
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  #re-calculate movie bias, using correct formula derived from calculus
  b_i <- train_set %>% 
    left_join(b_u, by="userId")  %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - b_u - mu )/(n()+l))
  } #end of iterations to calculate optimizing user and movie bias for a given lambda
  
  # calculate predictions on test set , not on validation set
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  #calculate RMSE
  return(RMSE(predicted_ratings, test_set$rating))
})

qplot(lambdas, rmses)
best_lambda <- lambdas[which.min(rmses)]
paste0("min RMSES = ",round(min(rmses),5)," with lambda = ", best_lambda)

#########################################
# Calculation best prediction model found
#########################################

# Calculation of user and movie bias using the best lambda found during the cross-validations
mu <- mean(train_set$rating)             # calculated before, repeated here for code clarity
best_lambda <- lambdas[which.min(rmses)] # calculated before, repeated here for code clarity

#use initial approximation formula to calculate movie bias, excluding user bias terms
b_i <- train_set %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+best_lambda))

# calculation of optimal user and movie bias, using 10 iterations to improve convergence of answer to optimal answer
for (iteration in 1:10){
  # calculate user bias
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+best_lambda))
  
  #re-calculate movie bias, using correct formula derived from calculus
  b_i <- train_set %>% 
    left_join(b_u, by="userId")  %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - b_u - mu )/(n()+best_lambda))
} #end of iterations to calculate optimizing user and movie bias


##################
# Final Validation
##################

# calculate predictions on validation set
validation_predicted_ratings <- 
  validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

#calculate RMSE on validation set
paste0("RMSE in validation set = ",RMSE(validation_predicted_ratings, validation$rating))
