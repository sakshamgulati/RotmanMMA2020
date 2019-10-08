library(MASS)
library(ISLR)
fix(Boston)
names(Boston)
#fitting a linear model
lm.fit=lm(Boston$medv~Boston$lstat, data=Boston)
plot(lm.fit)
summary(lm.fit)
#to find out what all other information is stored in lm.fit
names(lm.fit)
#to find the coefficients
coef(lm.fit)
#to find out the confidence intervals
confint(lm.fit)
#we use predict function to produce CI and other prediction intervals
?predict
predict(lm.fit,data.frame(lstat=c(5,10,15)), interval = "confidence")

predict(lm.fit,data.frame(lstat=c(5,10,15)), interval = "prediction")
attach(Boston)
#when we plot it looks kinda polynomial
plot(lstat,medv)
abline(lm.fit,lwd=3)
summary(lm.fit)
#to view the plot functions for lm.fit
#this function divides the screen into 4 parts
par(mfrow=c(2,2))
plot(lm.fit)

#to plot the residuals(a pattern in residuals tells us that we need to go for polynomial reg)
par(mfrow=c(1,1))
plot(predict(lm.fit),residuals(lm.fit))
#to produce teh studentized residuals.Studentized residuals are more effective in detecting outliers and in assessing the equal variance assumption. The Studentized Residual by Row Number plot essentially conducts a t test for each residual. Studentized residuals falling outside the red limits are potential outliers.
plot(predict(lm.fit),rstudent(lm.fit))
#we use leverage functions to odentify any outliers
plot(hatvalues(lm.fit))
#to identify the outlier
which.max(hatvalues(lm.fit))

#multiple linear regression
lm.fit=lm(medv~ lstat+age, data=Boston)
summary(lm.fit)


lm.fit=lm(medv~., data=Boston)
summary(lm.fit)
#to get individualized metrics suchas r-sq
summary(lm.fit)$r.sq
#the RSE
summary(lm.fit)$sigma

#we can also use VIF to find out the colleanearity between vectors. it shouldnt exceed 5
install.packages("car")
library(car)
vif(lm.fit)
#to run the lm again but without a specific predictor
lm.fit1=lm(medv~.-age,data = Boston)
summary(lm.fit1)
#or even update function can be used for the same
lm.fit1=update(lm.fit,~.-age)
#when predictors are collinear, we include an interaction term.(to tackle collinearity) 
summary(lm(medv~ lstat*age,data = Boston))

#we can create polynomial reg using i() func
lm.fit2=lm(medv~lstat+I(lstat^2))
summary(lm.fit2)
#to further check ,use anove model(null hypo- both models fit equally well, alternative hypo- full model is superior)

anova(lm.fit,lm.fit2)
plot(lm.fit2)

#to further go for poly func regressions
lm.fit5=lm(medv~poly(lstat,5))
summary(lm.fit5)
summary(lm(medv~log(rm),data = Boston))
#now lets try to use qualitative variables as well
fix(Carseats)
attach(Carseats)
lm.fit=lm(Sales~.+Income:Advertising+Price:Age,data = Carseats)
summary(lm.fit)
#to see the encoding for dummy variables
contrasts(ShelveLoc)

Loadlib=function(){
  library(MASS)
  library(ISLR)
  print("Libraries loaded!")
}

Loadlib()
