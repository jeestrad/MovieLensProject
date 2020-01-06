################################
# HERE STARTS THE CODE PROVIDED ON THE INSTRUCTIONS
################################

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

library(data.table)
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

#set.seed(1, sample.kind="Rounding") # if using a later version than R 3.5
#if using R 3.5 or earlier, use `set.seed(1)` instead
set.seed(1)
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

################################
# HERE ENDS THE CODE PROVIDED ON THE INSTRUCTIONS
################################


################################
# LOAD IMPORTANT LIBRARIES
################################
library(dslabs)
library(tidyverse)
library(caret)
library(lubridate)
library(purrr)
library(knitr)

################################
# DATA EXPLORATION AND DATA CLEANING
################################

#DETERMINE THE DIMENSIONS OF THE EDX (TRAINING) AND VALIDATION (TESTING) DATASETS

dim(validation)
dim(edx)

#EXPLORE THE VARIABLES OR INFORMATION INCLUDED ON THE DATASETS
head(validation)
head(edx)

#DETERMINE IF THERE IS NA IS THE DATASET
mean(is.na(edx))
mean(is.na(validation))


################################
# DEFINING RMSE
################################

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

################################
# CREATING A SUBSET FROM EDX FOR TRAINING AND TEST, INDEPENDENT FROM VALIDATION
################################

#THE APPROACH WILL BE USING A SUBSET OF EDX AS TRAINING THEN
#TO SELECT THE BEST METHOD. THEN, APPLY THE BEST METHOD TO
#THE VALIDATION DATASET

#set.seed(1, sample.kind="Rounding") # if using a later version than R 3.5
# if using R 3.5 or earlier, use `set.seed(1)` instead
set.seed(1)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

# Make sure userId and movieId in test_set are also in train set
test_set <- test_set %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

################################
# FIT MODELS
################################

#NOW LETS EVALUATE THE SIMPLEST MODEL (NAIVE) TO SEE HOW JUST BY 
#DETERMININING THE AVERAGE RATING AMONG ALL THE MOVIES MU_RATINGS
#REGARDLESS OF USER SPECIFIC EFFECT AND MOVIE BIAS

mu_hat <- mean(train_set$rating)
mu_hat

naive_rmse <- RMSE(test_set$rating, mu_hat)
naive_rmse

rmse_results <- tibble(method = "Just the rating average", RMSE = naive_rmse)
rmse_results

#MOVIE BIASED

#SHOW THE RATING VARIABILITY
mu <- mean(train_set$rating) 
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 20, data = ., color = I("black"))


#THEN WE CAN AND THE MOVIE BIAS EFFECT
mu <- mean(train_set$rating) 
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
movie_avgs

predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

model_1_rmse <- RMSE(predicted_ratings, test_set$rating)
model_1_rmse

rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie Bias Effect Model",
                                 RMSE = model_1_rmse ))
rmse_results

#USER SPECIFIC EFFECT
user_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))
user_avgs %>% qplot(b_u, geom ="histogram", bins = 30, data = ., color = I("black"))

predicted_ratings <- test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
model_2_rmse <- RMSE(predicted_ratings, test_set$rating)
model_2_rmse
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie + User Effects Model",
                                 RMSE = model_2_rmse))

rmse_results

#LETS DETERMINE IF THERE IS ENOUGH MOTIVATION FOR REGULARIZATION OF THE MOVIE EFFECT
#LETS DETERMINE THE 10 BEST AND 10 WORST ESTIMATES OF MOVIE EFFECT (BI) AND THE NUMBER
#OF ITS OCCURANCE

#LINK THE movieID to the Title
movie_titles <- train_set %>%
  select(movieId, title) %>%
  distinct()

#BEST MOVIE ESTIMATES
train_set %>% count(movieId) %>%
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>%
  select(title, b_i, n) %>%
  slice(1:10)

#LOWER MOVIE ESTIMATES
train_set %>% count(movieId) %>%
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>%
  select(title, b_i, n) %>%
  slice(1:10)

#PENALIZED LEAST SQUARES. DETERMINE WHICH LAMBDA MINIMIZES THE RMSE
lambdas <- seq(0, 10, 0.25)
mu <- mean(train_set$rating)
just_the_sum <- train_set %>%
  group_by(movieId) %>%
  summarize(s = sum(rating - mu), n_i = n())
rmses <- sapply(lambdas, function(l){
  predicted_ratings <- test_set %>%
    left_join(just_the_sum, by='movieId') %>%
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_set$rating))
})
qplot(lambdas, rmses)
lambdas[which.min(rmses)]

lambda <- lambdas[which.min(rmses)]
mu <- mean(train_set$rating)
movie_reg_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n())

predicted_ratings <- test_set %>%
  left_join(movie_reg_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)
model_3_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Regularized Movie Effect Model",
                                 RMSE = model_3_rmse))
rmse_results

####REGULARIZED MOVIE AND USER EFFECTS
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <-
    test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_set$rating))
})
qplot(lambdas, rmses)
lambda <- lambdas[which.min(rmses)]
lambda
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Regularized Movie + User Effect Model",
                                 RMSE = min(rmses)))
rmse_results


#AS A MATTER OF THE LIMITED TIME, I HAD TO STOP HERE. FURTHER MODEL WOULD INCLUDE
#THE GENRE EFFECT AS PRESENTED AT THE INTRODUCTION OF THE ASSIGNMENT

edx %>% group_by(genres) %>% summarize(avg_rating = mean(rating)) %>% arrange(desc(avg_rating)) 

################################
# FIT MODELS ON THE VALIDATION DATASET
################################

#APPLY REGULARIZED MOVIE AND USER EFFECT MODEL TO TRAIN THE EDX DATA
#AND TO TEST IT ON THE VALIDATE USING THE LAMBDA THAT MINIMIZED THE 
#RMSE ON THE TRAINING DATA

l <- lambda 
mu <- mean(edx$rating)
b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+l))
b_u <- edx %>%
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+l))
predicted_ratings <-
  validation %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

RMSE <- RMSE(predicted_ratings, validation$rating)
RMSE
