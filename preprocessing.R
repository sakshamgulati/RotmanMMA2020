x=c(1,2,3,4,5)
length(x)
#to see all the functions and vectors that you have saved
ls()
#to remove all the functions and vectors that you have saved
rm(x)
#to remove all the functions in the R enviornment
rm(list=ls())
#to create  a matrix
?matrix()
x=matrix(data = c(1,2,3,45),nrow=2,ncol=2,byrow = TRUE)
xran=matrix(data=rnorm(4),nrow = 2,ncol = 2)
xran
#used to create a randome number
y=rnorm(100000,mean=0,sd=1)
hist(y)
pdf("figure.pdf")
dev.off()
#the dev.off indicated R that we are done creating the pdf
#function of set seed
set.seed(1)
rnorm(40)
#now remove the set seed and run the rnorm(again)
3:11
seq(3,11,length=11)
Auto
#to view it in a spreadsheet like cwindow use fix()
fix(Auto)
#rows vs cols
dim(Auto)
#to omit missing variables
auto=na.omit(Auto)
#to check which variable has a missing value
sum(is.na(Auto))
dim(Auto)

plot(Auto$cylinders,Auto$mpg)
Auto$cylinders=as.factor(Auto$cylinders)
#hist only works with numberic variables
hist(Auto$mpg)

#to create a scatter plot of all the variables check for 
pairs(Auto)
#to create a scatter plot of some variables
pairs(~mpg+displacement+horsepower+weight+acceleration,Auto)
identify(Auto$horsepower,Auto$mpg,Auto$name)
library(psych)
#pysch library has a function which does the job. So instead of writing all the seperate functions, we can just use this one line to do the job.
describe(Auto_new)
summary(Auto)

#checking for correlations
rcorr(as.matrix(Auto_num))

