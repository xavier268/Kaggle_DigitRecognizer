# loadData.R

train <- read.csv("../Dataset/train.csv")
test  <- read.csv("../Dataset/test.csv")

train <- data.matrix(train)
test <- data.matrix(test)

train.X <- train[,-1]/255
train.y <- train[,1]
test.X <- test / 255
