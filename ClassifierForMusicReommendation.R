###Machine Learning Final Project###
###Spotify Song Attributes###
library(ISLR)
library(MASS)
library(class)
require(boot)
require(pROC)
require(verification)
rm(list = ls())
data <- read.csv(file = "data.csv", header = T)
data <- data[,-c(1)]
n <- nrow(data)
#Logistic Regression
glm.fit <- glm(target ~ acousticness+danceability+duration_ms+energy+instrumentalness+key+liveness+loudness+mode+speechiness+tempo+time_signature+valence,
               family = binomial, data = data)
glm.probs <- predict(glm.fit, type = "response")
glm.pred <- rep(0,n)
glm.pred[glm.probs > 0.5] = 1
mytable <- table(data$target, glm.pred)
mytable

my.table <- matrix(ncol = 6, nrow=0, dimnames = list(NULL, c("Method", "Correct", "Type I", "Type II", 
                                                     "Power", "Precision")))
methods <- c('Log.Reg', 'Modified.Log.Reg', 'LDA', 'QDA', 'Best K', 'k Fold', 'Lasso', 'Ridge', 'QDA Modified')


#Overall fraction of correct prediction
correct <- mean(glm.pred == data$target)
#Overall error rate
mean(glm.pred != data$target)
#Type I error
type1 <- mytable[1, 2] / sum(mytable[1, ])
#Type II error
type2 <- mytable[2, 1] / sum(mytable[2, ])
#Power of the model
power <- mytable[2, 2] / sum(mytable[2, ])
#Precision of the model
prec<- mytable["1", "1"] / sum(mytable[, "1"])

my.table <- rbind(my.table, c(methods[1], correct, type1, type2, power, prec))

#Predictors that are significant
coef(glm.fit)
summary(glm.fit)

summary(glm.fit)$coefficients[,4]

which(summary(glm.fit)$coefficients[,4] < 0.05)

#Logistic Regression Using Statistically Significant Predictors
glm.fit2 <- glm(target ~ acousticness+danceability+duration_ms+instrumentalness+loudness+speechiness+tempo+valence,
                family = binomial, data = data)
glm.probs2 <- predict(glm.fit2, type = "response")
glm.pred2 <- rep(0,n)
glm.pred2[glm.probs2 > 0.5] = 1
mytable2 <- table(data$target, glm.pred2)
mytable2
#Overall fraction of correct prediction
correct2 <- mean(glm.pred2 == data$target)
#Overall error rate
mean(glm.pred2 != data$target)
#Type I error
type12 <- mytable2[1, 2] / sum(mytable2[1, ])
#Type II error
type22 <- mytable2[2, 1] / sum(mytable2[2, ])
#Power of the model
power2 <- mytable2[2, 2] / sum(mytable2[2, ])
#Precision of the model
prec2 <- mytable2["1", "1"] / sum(mytable2[, "1"])

my.table <- rbind(my.table, c(methods[2], correct2, type12, type22, power2, prec2))

###LDA###
lda.fit <- lda(target ~ acousticness+danceability+duration_ms+instrumentalness+loudness+speechiness+tempo+valence,
               data = data)
lda.pred <- predict(lda.fit, data)
lda.class <- lda.pred$class
mytable3 <- table(data$target,lda.class)
mytable3
#Overall fraction of correct prediction
correct3 <- mean(lda.class == data$target)
#Overall error rate
mean(lda.class != data$target)
#Type I error
type13 <- mytable3[1, 2] / sum(mytable[1, ])
#Type II error
type23 <- mytable3[2, 1] / sum(mytable[2, ])
#Power of the model
power3 <- mytable3[2, 2] / sum(mytable[2, ])
#Precision of the model
prec3 <- mytable3["1", "1"] / sum(mytable[, "1"])

my.table <- rbind(my.table, c(methods[3], correct3, type13, type23, power3, prec3))

###QDA###
qda.fit <- qda(target ~ acousticness+danceability+duration_ms+instrumentalness+loudness+speechiness+tempo+valence,
               data = data)
qda.pred <- predict(qda.fit, data, type='response')
qda.class <- qda.pred$class
mytable4 <- table(data$target,qda.class)
mytable4
#Overall fraction of correct prediction
correct4 <- mean(qda.class == data$target)
#Overall error rate
mean(qda.class != data$target)
#Type I error
type14 <- mytable4[1, 2] / sum(mytable[1, ])
#Type II error
type24 <- mytable4[2, 1] / sum(mytable[2, ])
#Power of the model
power4 <- mytable4[2, 2] / sum(mytable[2, ])
#Precision of the model
prec4 <- mytable4["1", "1"] / sum(mytable[, "1"])

my.table <- rbind(my.table, c(methods[4], correct4, type14, type24, power4, prec4))

###Best K###
m.train  <-  sample(nrow(data), 0.8 * nrow(data))
m.test  <-  setdiff(1:nrow(data), m.train) 
m.trainset <- data[m.train,]
m.testset <- data[m.test,]

train.x <- data.frame(m.trainset)[,-c(14:16)]
train.y <- as.factor(m.trainset$target)

test.x <- data.frame(m.testset)[,-c(14:16)]
test.y <- as.factor(m.testset$target)

test.errors <- c()
for(i in 1:20) {
  knn.pred <- knn(train.x, test.x, train.y, k = i)
  test.errors[i] <- mean(test.y != knn.pred)
}

bestk <- which.min(test.errors)
cat("Best data k =", bestk)

knnbest.pred <- knn(train.x, test.x, train.y, k=bestk)
knnbest.table <- table(test.y, knnbest.pred)
knnbest.table

## Fraction of correct predictions
correct5 <- mean(knnbest.pred == test.y)
## Overall error rate
mean(knnbest.pred != test.y)
## Type I error rates (false positive)
type15 <-knnbest.table[1,2] / sum(knnbest.table[1,])
## Type II error rates (false negative)
type25 <- knnbest.table[2,1] / sum(knnbest.table[2,])
## Power
power5 <- 1 - (knnbest.table[2,1] / sum(knnbest.table[2,]))
## Precision
prec5 <- knnbest.table[2,2] / sum(knnbest.table[,2])

my.table <- rbind(my.table, c(methods[5], correct5, type15, type25, power5, prec5))


###kFold Cross Validation###
cv.error <- c()
for (i in 1:10){
  glm.kfold <- glm(target ~ poly(acousticness+danceability+duration_ms+energy+
                                 instrumentalness+key+liveness+loudness+mode+
                                 speechiness+tempo+time_signature+valence, i), family = binomial, data=data)
  cv.error[i] <- cv.glm(data, glm.kfold, K=10)$delta[1]
}
#plot(cv.error, type="b", xlab='Degree', ylab='10-Fold Cross Validation Error', main='k-Fold Cross Validation ')

glm.fit3 <- glm(target ~  poly(acousticness+danceability+duration_ms+energy+
                                 instrumentalness+key+liveness+loudness+mode+
                                 speechiness+tempo+time_signature+valence, 8), family = binomial, data=data)
glm.probs3 <- predict(glm.fit3, type = "response")
glm.pred3 <- rep(0,n)
glm.pred3[glm.probs3 > 0.5] = 1
mytable6 <- table(data$target, glm.pred3)
mytable6
#Overall fraction of correct prediction
correct6 <- mean(glm.pred3 == data$target)
#Overall error rate
mean(glm.pred3 != data$target)
#Type I error
type16 <- mytable6[1, 2] / sum(mytable6[1, ])
#Type II error
type26 <- mytable6[2, 1] / sum(mytable6[2, ])
#Power of the model
power6 <- mytable6[2, 2] / sum(mytable6[2, ])
#Precision of the model
prec6 <- mytable6["1", "1"] / sum(mytable6[, "1"])

my.table <- rbind(my.table, c(methods[6], correct6, type16, type26, power6, prec6))


###Lasso###
x <- model.matrix(target ~ ., data = data)[, -ncol(data)]
y <- data$target
train <- sample(1:nrow(data), nrow(data)/2)
test <- (-train)
x.train <- x[train, ]
x.test <- x[test, ]
y.train <- y[train]
y.test <- y[test]

cv.out <- cv.glmnet(x.train, y.train, alpha=1, family='binomial')
bestlam <- cv.out$lambda.min

lasso.prob <- predict(cv.out, newx=x.test, s = bestlam, type="response")
lasso.predict <- rep(0, nrow(data))
lasso.predict[lasso.prob > .5] <- 1
lasso.table <- table(data$target, lasso.predict)
lasso.table

#Overall fraction of correct prediction
correct7 <- mean(lasso.predict == data$target)
#Overall error rate
mean(lasso.predict != data$target)
#Type I error
type17 <- lasso.table[1, 2] / sum(lasso.table[1, ])
#Type II error
type27 <- lasso.table[2, 1] / sum(lasso.table[2, ])
#Power of the model
power7 <- lasso.table[2, 2] / sum(lasso.table[2, ])
#Precision of the model
prec7 <- lasso.table["1", "1"] / sum(lasso.table[, "1"])

my.table <- rbind(my.table, c(methods[7], correct7, type17, type27, power7, prec7))


###Ridge###
cv.out <- cv.glmnet(x.train, y.train, alpha=0, family='binomial')
bestlam <- cv.out$lambda.min

ridge.prob <- predict(cv.out, newx=x.test, s = bestlam, type="response")
ridge.predict <- rep(0, nrow(data))
ridge.predict[ridge.prob > .5] <- 1
ridge.table <- table(data$target, ridge.predict)
ridge.table

#Overall fraction of correct prediction
correct8 <- mean(ridge.predict == data$target)
#Overall error rate
mean(ridge.predict != data$target)
#Type I error
type18 <- ridge.table[1, 2] / sum(ridge.table[1, ])
#Type II error
type28 <- ridge.table[2, 1] / sum(ridge.table[2, ])
#Power of the model
power8 <- ridge.table[2, 2] / sum(ridge.table[2, ])
#Precision of the model
prec8 <- ridge.table["1", "1"] / sum(ridge.table[, "1"])

my.table <- rbind(my.table, c(methods[8], correct8, type18, type28, power8, prec8))


###Modifying QDA using ROC###
actuals <- factor(data$target)
predicted <- qda.pred$posterior

aucc <- roc.area(as.numeric(actuals)-1,predicted[,2])$A
aucc
roc.plot(as.integer(as.numeric(actuals))-1,predicted[,2], 
         main="ROC Curve")

###Modified QDA###
qda.mod.pred <- rep(0,n)
qda.mod.pred[predicted[,2] > 0.3] = 1
qda.mod.table <- table(data$target, qda.mod.pred)
qda.mod.table

#Overall fraction of correct prediction
correct9 <- mean(qda.pred$class == data$target)
#Overall error rate
mean(qda.pred$class != data$target)
#Type I error
type19 <- qda.mod.table[1, 2] / sum(qda.mod.table[1, ])
#Type II error
type29 <- qda.mod.table[2, 1] / sum(qda.mod.table[2, ])
#Power of the model
power9 <- qda.mod.table[2, 2] / sum(qda.mod.table[2, ])
#Precision of the model
prec9 <- qda.mod.table["1", "1"] / sum(qda.mod.table[, "1"])

my.table <- rbind(my.table, c(methods[9], correct9, type19, type29, power9, prec9))
