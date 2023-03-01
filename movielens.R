
# name: Movielens Project
# author: Kuntal Bhar
# creating a movie recommendation system that build the best model that can predict predict movie ratings for users in a large moveilens collected by GroupLens Research dataset with accuracy

  
#### DATASET ####
# **** Create edx set, final_holdout_test set, and submission file **** 
  
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)




### DATA ANALYSIS ###

# Most rated films
edx %>% group_by(title) %>%
  summarize(topn_ratings = n()) %>%
  arrange(desc(topn_ratings))

# Number of movies rated once
edx %>% group_by(title) %>%
  summarize(topn_ratings = n()) %>%
  filter(topn_ratings==1) %>%
  count() %>% pull()

# structure of edx training set and number of rows
str(edx)

# structure of edx training set and number of rows
str(final_holdout_test)


# summary movies and user 
summarize(n_users = n_distinct(userId), 
            n_movies = n_distinct(movieId))

# summary of training set
summary(edx)



# rating distribution graph
edx %>%
  ggplot(aes(rating)) +
  geom_histogram( binwidth = 0.25, color = "blue", fill="blue") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  scale_y_continuous(breaks = c(seq(0, 3000000, 500000))) +
  ggtitle("Rating By Count")


### METHODS  ###

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


## First Method -> Average movie rating ## 

# Calculate mean/average 
mu <- mean(edx$rating)
mu


#Calulate RMSE on test data final_holdout_tes
avg_rmse<-RMSE(final_holdout_test$rating, mu)
# Print RMSE
cat('RMSE for Average Rating is', avg_rmse)


## Second Method -> Rating by including Movie Bias  ##

# add movie bias b_i
b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

#draw histogram of training data (edx) with movie bias effect 
b_i %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"),
              main = "Effect of Movie Bias (b_i)")

# predict rating considering movie bias on test data final_holdout_test
prediction <- final_holdout_test %>% 
  left_join(b_i, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

# calculate RMSE 
movie_rmse <-RMSE(final_holdout_test$rating, prediction)

##Print RMSE
cat('RMSE for Rating that include Movie Bias is', movie_rmse)


## Third Method -> Rating now including User Bias ##

# add movie bias b_i
b_u <- edx %>% 
  left_join(b_i, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

#draw histogram of training data (edx) with user bias effect
b_u %>% qplot(b_u, geom ="histogram", bins = 10, data = ., color = I("black"),
              ylab = "Number of users", main = "Effect of User Bias (b_u)")

# predict rating considering user bias on test data final_holdout_test
prediction <- final_holdout_test %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# calculate RMSE 
user_rmse <-RMSE(final_holdout_test$rating, prediction)

##Print RMSE
cat('RMSE for Rating that include Movie and User Bias is', user_rmse)



## Third Method -> Apply Regularization to our previous method ##

# get lambda from a sequence
lambdas <- seq(0, 10, 0.25)

#Creating function that return RMSE using regularization that repeat earlier steps for each lambda
rmse_series <- sapply(lambdas, function(lmda){
  
  # calculate mean/average rating across all training data (edx)
  mu <- mean(edx$rating)
  
  # apply regularization on movie bias (b_i)
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lmda))
  
  # similarly apply regularization on user bias (b_u)
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lmda))
  
  # predict using above regularization on test data final_holdout_test 
  prediction <- final_holdout_test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  # output RMSE of these predictions
  return(RMSE(prediction, final_holdout_test$rating))
})


#plot shows RMSE vs. Lambda
qplot(lambdas, rmse_series, colour = I("blue"),xlab = "Lambda", ylab="RMSE")


# best Lambda
best_lambda<-lambdas[which.min(rmse_series)]
best_lambda

## Applying the best Lambda to our final method

# Use the best lambda on movie bias (b_i) on training set (edx)
b_i <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+best_lambda))

# Use the best lambda on user bias (b_u) on training set (edx)
b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+best_lambda))

# predict using best lambda applied above for regularization on test data final_holdout_test 
prediction <- final_holdout_test %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# output RMSE of our final model
# calculate RMSE 
regularization_rmse <-RMSE(final_holdout_test$rating, prediction)

##Print RMSE
cat('RMSE for Rating that include best Lambda regularization in Movie and User Bias is', regularization_rmse)

#### Summary Table ####

#build the column values
c1<-c("Average",
      "Effect included Movie Bias",
      "Effect included Movie and User Bias",
      "Effect included regulerization on Movie and User Bias")
c2<-c(avg_rmse,
      movie_rmse,
      user_rmse,
      regularization_rmse )
#add it to dataframe
df <- data.frame(c1,c2)

#name the heading
names(df) <- c('Method Description', 'RMSE')

#print the result data frame
head(df)



