library(fpp)
plot(a10)
getwd()
setwd("/Users/seanhegarty/Desktop")
getwd()
dir()
tute1 <- read.csv("tute1.csv")
tute1[,2]
names(tute1)
tute1$Sales
tute1 <- ts(tute1[,-1],start=1981,frequency=4)
seasonplot(tute1[,"Sales"])
#seasonplot(tute1$Sales)
# does not work with this type of subsetting
monthplot(tute1[,"Sales"])
summary(tute1)
pairs(as.data.frame(tute1))
cor.test(tute1[,"Sales"],tute1[,"AdBudget"])

#### Graphics Package ####
#Time Plots
plot(melsyd)
plot(melsyd[,"Economy.Class"])
plot(melsyd[,"Economy.Class"], main= "Economy class passengers: Melbourne-Sydney",
     xlab="Year",ylab="Passengers")
plot(a10,ylab="$ million", xlab="Year", main= "Antidiabetic drug sales")

# Seasonal plots
seasonplot(a10,ylab="$ million",xlab="Year",
           main="Seasonal plot: antidiabetic drug sales", 
           year.labels=TRUE, year.labels.left=TRUE,col=1:20,pch=19)

View(a10)
class(a10)

# Seasonal subseries plots
monthplot(a10,ylab= "$ million",xlab="Month",xaxt="n",
          main= "Seasonal deviation plot: antidiabetic drug sales")
axis(1,at =1:12,labels=month.abb, cex=0.8)

# Scatterplots
summary(fuel)
plot(jitter(fuel[,5]),jitter(fuel[,8]))
plot(fuel[,5],fuel[,8])
plot(jitter(fuel[,5]), jitter(fuel[,8]),
     xlab="City mpg",ylab="Carbon footprint")

# Scatterplot matrices
pairs(fuel[,-c(1,2,4,7)],pch=19)

#### Numeric data summaries ####
fuel2 <- fuel[fuel$Litres<2,]
summary(fuel2[,"Carbon"])
sd(fuel2[,"Carbon"])
#Basic Statistics
xbar <- sum(fuel$City)/length(fuel$City)
mean(fuel$City)
xbar
sdx <- sqrt(sum((fuel$City-xbar)^2)/(length(fuel$City)-1))
sd(fuel$City)
sdx
ybar <- sum(fuel$Carbon)/length(fuel$Carbon)
mean(fuel$Carbon)
ybar
sdy <- sqrt(sum((fuel$Carbon-ybar)^2)/(length(fuel$Carbon)-1))
sd(fuel$Carbon)
sdy
covXY <- sum((fuel$City-xbar)*(fuel$Carbon-ybar))/length(fuel$Carbon)
cov(fuel$City,fuel$Carbon)
covXY
corXY <- covXY/(sdy*sdx)
corXYn <- sum((fuel$City-xbar)*(fuel$Carbon-ybar))/(
        sqrt(sum((fuel$City-xbar)^2))*sqrt(sum((fuel$Carbon-ybar)^2))
        )
cor(fuel$City,fuel$Carbon)
corXY
corXYn

# Autocorrelation
beer2 <- window(ausbeer,start=1992,end=2006-0.1)
lag.plot(beer2,lags=9,do.lines=FALSE)

# White noise are Time Series that show no autocorrelation
set.seed(30)
x <- ts(rnorm(50))
plot(x,main="White Noise")
Acf(x)
Acf(beer2)

#### Simple Forecasting Methods #####
mean(x=a10)
mean(a10) + c(-1,1)*qt(0.8,length(a10))*sd(a10)/sqrt(length(a10))
#Mean method
meanf(x=a10,h=4)
# Naive methods
naive(x=a10,h=4)
rwf(a10,4)
# Seasonal Naive methods
snaive(a10,4)
rwf(a10,4,drift=T)

# Plottng
beer2 <- window(ausbeer,start= 1992,end=2006-0.1)
beerfit1 <- meanf(beer2,h=11)
beerfit2 <- naive(beer2,h=11)
beerfit3 <- snaive(beer2,h=11)

plot(beerfit1,plot.conf=F, main= "Forecasts for quarterly beer production")
lines(beerfit2$mean,col=2)
lines(beerfit3$mean,col=3)
legend("topleft",lty=1,col=c(4,2,3),
       legend= c("Mean method","Naive method","Seasonal naive method"))

dj2 <- window(dj,end=250)
plot(dj2, main= "Dow Jones Index (daily ending 15 Jul 94)",
     ylab="",xlab="Day", xlim=c(3,290))
lines(meanf(dj2,42)$mean,col=4)
lines(rwf(dj2,h=42)$mean, col=2)
lines(rwf(dj2,h=42,drift=T)$mean,col=3)
legend("topleft",lty=1, col= c(4,2,3),
       legend=c("Mean Method","Naive method","Drift method"))

#### Transforming data ####
plot(elec,ylab="Electricity demand",
     xlab="Year", main="Monthly electricity demand")

plot(log(elec),ylab="Transformed electricity demand",
     xlab="Year", main="Transformed monthly electricity demand")

# The BoxCox.lambda() function will choose a value of lambda for you.
lambda <- BoxCox.lambda(elec) # =0.27
plot(BoxCox(elec,lambda))

#Calendar Adjustments
monthdays <- rep(c(31,28,31,30,31,30,31,31,30,31,30,31),14)
monthdays[26+ (4*12)*(0:2)] <- 29
par(mfrow=c(2,1))
plot(milk,main="Monthly milk production per cow",
     ylab="Pounds",xlab="Years")
plot(milk/monthdays, main= "Average milk production per cow per day",
     ylab="Pounds",xlab="Years")
par(mfrow=c(1,1))

# Population adjustments
# consider per person, or per thousand people, or per million people, etc.

# Inflation adjustments
# Data affected by money value are best adjusted before modeling.
# Use a price index to adjust money value over time.

#### Evaluating forecast accuracy ####
# mean absolute error (MAE) = mean(abs(yi-yiest))
# RMSE = root mean squared error = exactly what is said in the order it is said
# percent error = pi= 100*(yi-yiest)/yi
# Mean abs percent error (MAPE) = mean(abs(pi)) ## This is bad for a few reasons
# Use sMAPE = symmetric MAPE = mean(200*abs(yi-yiest)/(yi+yiest))
## Apparently we shouldn't use any kind of MAPE

# Scaled errors
# qj = ej / [1/(T-1) * sum(abs(yt-ytminus1))]
# if qj < 1, then the forecast is better than the average naive forecast.
# if qj > 1, then the forecast is worse than the average naive forecast.
# MASE = mean(abs(qj))

beer2 <- window(ausbeer,start=1992,end=2006-0.1)

beerfit1 <- meanf(beer2,h=11)
beerfit2 <- rwf(beer2,h=11)
beerfit3 <- snaive(beer2,h=11)

plot(beerfit1,plot.con=FALSE,
     main="Forecasts for quarterly beer production")
lines(beerfit2$mean,col=2)
lines(beerfit3$mean,col=3)
lines(ausbeer)
legend("bottomleft",lty=1,col=c(4,2,3),
       legend=c("Mean method","Naive method","Seasonal naive method"))

beer3 <- window(ausbeer,start=2006)
accuracy(beerfit1,beer3)
accuracy(beerfit2,beer3)
accuracy(beerfit3,beer3)

dj2 <- window(dj,end=250)
plot(dj2, main="Dow Jones Index (daily ending 15 Jul 94)",
     ylab="",xlab="Day", xlim=c(2,290))
lines(meanf(dj2,h=42)$mean,col=4)
lines(rwf(dj2,h=42)$mean,col=2)
lines(rwf(dj2,drift=TRUE,h= 42)$mean,col=3)
legend("topleft",lty=1,col=c(4,2,3),
       legend= c("Mean method","Naive method","Drift method"))
lines(dj)

dj3 <- window(dj,start=251)
accuracy(meanf(dj2,h=42),dj3)
accuracy(rwf(dj2,h=42),dj3)
accuracy(rwf(dj2,drift=TRUE,h=42),dj3)

#### Residual Diagnostics ####
# 1. Good forecasting will produce residuals that are uncorrelated
# 2. The residuals will have zero mean. Otherwise the forecasts are biased.
# Forecasts that do not satify these properties can be improved. Forecasts
# that satisfy these properties may also be improved.
# Residuals are a good way to check that your model is using the information
# well, but not a good way to select a model.

# for stock market indexes, the best forecasting method is often the naive method.
dj2 <- window(dj, end=250)
plot(dj2, main="Dow Jones Index (daily ending 15 Jul 94)",
     ylab="", xlab="Day")
res <- residuals(naive(dj2))
plot(res, main="Residuals from naive method",
     ylab="",xlab="Day")
Acf(res, main="ACF of residuals")
# Autocorrelation function measures the correlation between lagged variables
hist(res,nclass="FD",main= "Histogram of residuals")

#Portmanteau tests for autocorrelation
# grouping aurtocorrelations ri to rk
#' Box-Pierce test
#'      Q = T * sum(ri^2) from ri to rk
#'      suggested k=10 for non-seasonal data and k=2m for
#'      seasonal data where m = period of seasonality
#'      if values > T/5 then k=T/5
#' Ljung-Box test
#'      Q* = T * (T+2) * sum((T-i)^-1 * ri^2) from i to k
#'      Q* suggests autocorrelation does not come from white noise series.
#'      How large is too large>
#'              if autoCor comes from white noise series, then Q and Q* would
#'              have x^2 distribution with (k-p) degfree (p=predictors)

Box.test(res,lag=10,fitdf=0)
Box.test(res,lag=10,fitdf=0, type= "Lj")

#### 2.8 Exercises ####
data(package="fma")
data("dole")
summary(dole)
dole2 <- window(dole,end=1993-0.5)
monthplot(dole2)
data("usdeaths")
usdeaths
usdeaths2 <- window(usdeaths)
monthplot(usdeaths2)
data("bricksq")
bricksq
bricksq2 <- window(bricksq,start=1956+0.25)
bricksq2
seasonplot(bricksq2)

data("dowjones")
plot(dowjones)
lines(meanf(dowjones)$mean)
lines(rwf(dowjones,drift=TRUE)$mean)
lines(snaive(dowjones)$mean)
lines(x=c(1,78),y=c(dowjones[1],dowjones[length(dowjones)]))

data("ibmclose")
ibmclose2 <- window(ibmclose,start=1,end=300)
plot(ibmclose2,xlim=c(0,370))
ibmclose3 <- window(ibmclose,start=301)
lines(ibmclose3,col="red")
lines(meanf(ibmclose2,h=69)$mean,col="blue")
lines(snaive(ibmclose2,h=69)$mean,col="green")
lines(rwf(ibmclose2,h=69,drift=TRUE)$mean,col="purple")
lines(x=c(1,300),y=c(ibmclose2[1],ibmclose2[300]))

data("hsales")
plot(hsales)
hsales
hsales2 <- window(hsales,end=1994-0.05)
hsales2
plot(hsales2,xlim=c(1973,1996))
lines(x=c(1973,1994-0.05),y=c(hsales2[1],hsales2[length(hsales2)]))
lines(meanf(hsales2,h=23)$mean,col="blue")
lines(rwf(hsales2,h=23,drift=TRUE)$mean,col="red")
lines(window(hsales,start=1994-0.05),col="green")

#### Regression and Correlation ####
plot(jitter(Carbon) ~ jitter(City), xlab="City (mpg)",
     ylab="Carbon footprint (tons per year)",data=fuel)
fit <- lm(Carbon~City,data=fuel)
abline(fit)
summary(fit)

#### Evaluating the regression model ####
# ei = yi-ypredi  <- residuals
res <- residuals(fit)
plot(jitter(res)~jitter(City),ylab="Residuals",xlab="City",data=fuel)
abline(0,0)
abline(h=0,col="red")
# goodness of fit R^2
# R^2 <- ∑(ypredi-yave)^2/∑(yi-yave)^2
# Standard error of regression Se
# Se = √(∑ei^2/(n-2))

#### Forecasting with regression ####
# ypred ± 1.96 * Se * sqrt(1+ (1/N) + (x-xave)^2/((N-1)*Sx^2))
fitted(fit)[1]
fcast <- forecast(fit,newdata=data.frame(City=30))
plot(fcast,xlab-"City (mpg)",ylab="Carbon footprint (tons per year)")
confint(fit,level=0.95)

#### Non-linear functional forms ####
par(mfrow=c(1,2))
fit2 <- lm(log(Carbon)~log(City),data=fuel)
plot(jitter(Carbon) ~ jitter(City), xlab="City (mpg)",
     ylab="Carbon footprint (tonnes per year)",data=fuel)
lines(1:50,exp(fit2$coef[1]+fit2$coef[2]*log(1:50)))
lines(1:50,exp(coef(fit2) %*% t(cbind(1,log(1:50)))))
plot(log(jitter(Carbon)) ~ log(jitter(City)), xlab= "log City mpg",
     ylab="log carbon footprint", data= fuel)
abline(fit2)
par(mfrow=c(1,1))

res <- residuals(fit2)
plot(jitter(res,amount=0.005)~ jitter(log(City)),
     ylab="Residuals",xlab="log(City)", data = fuel)
abline(h=0)

#### Regression w/ time series data ####
par(mfrow=c(1,2))
fit.ex3 <- lm(consumption ~ income, data=usconsumption)
plot(usconsumption,ylab="% change in consumption and income",
     plot.type="single", col=1:2,xlab="Year")
legend("topright", legend=c("Consumption", "Income"),
       lty=1,col=c(1,2),cex=0.9)
plot(consumption ~ income, data= usconsumption,
     ylab="% change in consumption", xlab="% change in income")
abline(fit.ex3)
summary(fit.ex3)
par(mfrow=c(1,1))

#### Scenario based forecasting ####
fcast <- forecast(fit.ex3,newdata=data.frame(income=c(-1,1)))
plot(fcast,ylab="% change in consumption",xlab="% change in income")

#### Forecasting time series w/ Linear trend ####
fit.ex4 <- tslm(austa ~ trend)
f <- forecast(fit.ex4,h=5, level=c(80,95))
plot(f,ylab="International tourist arrivals to Australia (millions)",
     xlab="t")
lines(fitted(fit.ex4),col="blue")
summary(fit.ex4)

#### Residual Autocorrelation ####
par(mfrow=c(2,2))
res3 <- ts(resid(fit.ex3),s=1970.25,f=4)
plot.ts(res3,ylab="res(Consumption)")
abline(0,0)
Acf(res3)
res4 <- resid(fit.ex4)
plot(res4,ylab="res(Tourism")
abline(h=0)
Acf(res4)
par(mfrow=c(1,1))

#### 5.1 Intro to Multiple Regression ####
panel.hist <- function(x, ...)
{
        usr <- par("usr"); on.exit(par(usr))
        par(usr = c(usr[1:2], 0, 1.5) )
        h <- hist(x, plot = FALSE)
        breaks <- h$breaks; nB <- length(breaks)
        y <- h$counts; y <- y/max(y)
        rect(breaks[-nB], 0, breaks[-1], y, col = "cyan", ...)
}
pairs(credit[,-(4:5)],diag.panel=panel.hist)
creditlog <- data.frame(score=credit$score,
                        log.savings=log(credit$savings+1),
                        log.income=log(credit$income+1),
                        log.address=log(credit$time.address+1),
                        log.employed=log(credit$time.employed+1),
                        fte=credit$fte,single=credit$single)
pairs(creditlog[,1:5],diag.panel=panel.hist)

fit <- step(lm(score ~ log.savings + log.income + log.address + 
                       log.employed + single,data=creditlog))
summary(fit)

plot(fitted(fit),creditlog$score,ylab="Score",
     xlab="Predicted Score")

Rsqrd <- sum((fitted(fit)-mean(creditlog$score))^2)/
        sum((creditlog$score-mean(creditlog$score))^2)
Rsqrd

#### 5.2 Dummy Variables ####
beer2 <- window(ausbeer,start=1992,end=2006-0.1)
fit <- tslm(beer2 ~ trend + season)
summary(fit)

plot(beer2, xlab="Year",ylab="",main="Quarterly Beer Production")
lines(fitted(fit),col=2)
legend("topright",lty=1,col=c(1,2),legend=c("Actual","Predicted"))

plot(fitted(fit),beer2,xy.lines=FALSE,xy.labels=FALSE,
     xlab="Predicted values",ylab="Actual values",
     main="Quarterly Beer Production")
abline(0,1,col="gray")

fcast <- forecast(fit)
plot(fcast,main= "Forecasts of beer production using linear regression")

#### 5.3 Selecting Predictors ####
CV(fit)

#### 5.4 Residual ####
fit <- lm(score ~ log.savings + log.income _ log.address + log.employed, 
          data = creditlog)
par(mfrow = c(2,2))
plot(creditlog$log.savings, residuals(fit), xlab="log(savings)")
plot(creditlog$log.income, residuals(fit), xlab="log(income)")
plot(creditlog$log.address, residuals(fit), xlab = "log(address)")
plot(creditlog$log.emploted, residuals(fit), xlab = "log(employed)")
par(mfrow = c(1,1))

plot(fitted(fit),residuals(fit), xlab="Predicted scores", ylab="Residuals")

fit <- tslm(beer2~ trend + season)
res <- residuals(fit)
par(mfrow=c(1,2))
plot(res,ylab="Residuals",xlab="Year")
Acf(res,main = "ACF of residuals")
par(mfrow=c(1,1))

dwtest(fit, alt="two.sided")

bgtest(fit,5)

hist(res, breaks="FD", xlab="Residuals", 
     main="Histogram of residuals", ylim=c(0,22))
x <- -50:50
lines(x, 560*dnorm(x,0,sd(res)), col=2)

?pmax
