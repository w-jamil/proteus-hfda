require(caret)
require(MASS)

Adam<- function(x,y){
  x <- as.matrix((x))
  y <- as.numeric(y)
  N<- ncol(x)
  T<- nrow(x)
  w<- matrix(0,ncol=N,nrow=1)
  g<- matrix(0,ncol=N,nrow=1)
  m<- matrix(0,ncol=N,nrow=1)
  v<- matrix(0,ncol=N,nrow=1)
  b1<-0.9
  eta<-1/T
  pred<-matrix(0,ncol=1,nrow=T)
  for(t in 1:T){
    x1<-as.matrix(x[t,])
    pred[t]<- t(x1)%*%t(w)
    alph<-(y[t]-pred[t])
    g<- g+(as.numeric(alph)*t(x1))
    m<- b1*m+(1-b1)*g
    mt<-m/(1-b1^t)
    v<- b1*v+(1-b1)*(g*g)
    vt<-v/(1-b1^t)
    w <- w - eta*mt/(sqrt(vt)+0.000000001)
  }
  res<-postResample(pred = pred, obs = Y)
  stats<- as.matrix(res)
  quant<-quantile(Y-pred,probs=c(.25,.50,.75))
  
  return(list(predictions=pred,performance=stats,quantiles=quant))
}


NGD<- function(x,y){
  x <- as.matrix((x))
  y <- as.numeric(y)
  N<- ncol(x)
  T<- nrow(x)
  w<- matrix(0,ncol=N,nrow=1)
  pred<-matrix(0,ncol=1,nrow=T)
  for(t in 1:T){
    x1<-as.matrix(x[t,])
    pred[t]<- t(x1)%*%t(w)
    alph<-(y[t]-pred[t])/(t(x1)%*%x1)
    w<- w+(as.numeric(1/T*alph)*t(x1))
  }
  res<-postResample(pred = pred, obs = y)
  stats<- as.matrix(res)
  quant<-quantile(as.matrix(y)-as.matrix(pred),probs=c(.25,.50,.75))
  
  return(list(predictions=pred,performance=stats,quantiles=quant))
}

CNLS<- function(x,y){
  x <- as.matrix((x))
  y <- as.numeric(y)
  N<- ncol(x)
  T<- nrow(x)
  w<- matrix(0,ncol=N,nrow=1)
  pred<-matrix(0,ncol=1,nrow=T)
  for(t in 1:T){
    x1<-as.matrix(x[t,])
    pred[t]<- t(x1)%*%t(w)
    alp<- y[t]-pred[t]
    n<-as.numeric(alp/(1/T+t(x1)%*%x1))
    w<- t(t(w)+(n*x1))
  }
  res<-postResample(pred = pred, obs = y)
  stats<- as.matrix(res)
  quant<-quantile(as.matrix(y)-as.matrix(pred),probs=c(.25,.50,.75))
  
  return(list(predictions=pred,performance=stats,quantiles=quant))
}

AROWR<- function(X,Y){
    X<- as.matrix(X)
    Y<-as.matrix(Y)
    N<- ncol(X)
    T<- nrow(X)
    bt<-matrix(0,ncol=1,nrow=N)
    At<-diag(1, N)
    pred<-matrix(0,ncol=1,nrow=T)
    a<-1/T
    for(t in 1:T){
      xt<-X[t,]
      pred[t,]<- tcrossprod(crossprod(bt,At),xt)
      At<- At + (tcrossprod(xt,xt) * 1/a)
      At<- chol2inv(chol(At))
      bt<- bt + (Y[t,]*xt)
      theta<-crossprod(At,bt)
    }
    res<-postResample(pred = pred, obs = Y)
    stats<- as.matrix(res)
    quant<-quantile(Y-pred,probs=c(.25,.50,.75))
    
    return(list(predictions=pred,performance=stats,quantiles=quant))
}


RLS<- function(X,Y){
    X<- as.matrix(X)
    Y<-as.matrix(Y)
    N<- ncol(X)
    T<- nrow(X)
    bt<-matrix(0,ncol=1,nrow=N)
    At<-diag(1, N)
    pred<-matrix(0,ncol=1,nrow=T)
    a<- 1/T
    for(t in 1:T){
      xt<-X[t,]
      pred[t,]<- tcrossprod(crossprod(bt,At),xt)
      At<- (a*At) + tcrossprod(xt,xt)
      At<- chol2inv(chol(At))
      bt<- bt + (Y[t,]*xt)
      theta<-crossprod(At,bt)
    }
    res<-postResample(pred = pred, obs = Y)
    stats<- as.matrix(res)
    quant<-quantile(Y-pred,probs=c(.25,.50,.75))
    
    return(list(predictions=pred,performance=stats,quantiles=quant))
}


ORR<- function(X,Y){
    X<- as.matrix(X)
    Y<-as.matrix(Y)
    N<- ncol(X)
    T<- nrow(X)
    bt<-matrix(0,ncol=1,nrow=N)
    a<- 1/T
    At<-diag(a, N)
    pred<-matrix(0,ncol=1,nrow=T)
    for(t in 1:T){
      xt<-X[t,]
      pred[t,]<- tcrossprod(crossprod(bt,At),xt)
      At<- At + tcrossprod(xt,xt)
      At<- chol2inv(chol(At))
      bt<- bt + (Y[t,]*xt)
      theta<-crossprod(At,bt)
    }
    res<-postResample(pred = pred, obs = Y)
    stats<- as.matrix(res)
    quant<-quantile(Y-pred,probs=c(.25,.50,.75))
    
    return(list(predictions=pred,performance=stats,quantiles=quant))
}

AAR<- function(X,Y){
    X<- as.matrix(X)
    Y<-as.matrix(Y)
    N<- ncol(X)
    T<- nrow(X)
    bt<-matrix(0,ncol=1,nrow=N)
    a<- 1/T
    At<-diag(a, N)
    pred<-matrix(0,ncol=1,nrow=T)
    for(t in 1:T){
      xt<-X[t,]
      pred[t,]<- tcrossprod(crossprod(bt,At),xt) / as.numeric(crossprod(xt,crossprod(At,xt))+1)
      At<- At + tcrossprod(xt,xt)
      InvA<- chol2inv(chol(At))
      bt<- bt + (Y[t,]*xt)
      theta<-crossprod(InvA,bt)
    }
    res<-postResample(pred = pred, obs = Y)
    stats<- as.matrix(res)
    quant<-quantile(Y-pred,probs=c(.25,.50,.75))
    
    return(list(predictions=pred,performance=stats,quantiles=quant))
  
}

ONS<- function(x,y){
  x<- as.matrix(x)
  T<- nrow(x)
  N<- ncol(x)
  w<- matrix(0,1,N)
  p<- matrix(0,T,1)
  A<- diag(1/0.1,N)
  eta<- 1/T
  for(t in 1:T){
    p[t,]<- w%*%x[t,]
    loss<- y[t] - p[t,]
    g<- eta*(loss)*x[t,]
    A<- A + g%*%t(g)
    w<- w + t(chol2inv(chol(A))%*%g)*1/eta
  }
  res<-postResample(pred = p, obs = y)
  stats<- as.matrix(res)
  quant<-quantile(y-p,probs=c(.25,.50,.75))
  
  return(list(predictions=p,performance=stats,quantiles=quant))
}

OSLOG<-function(X,Y){
    X<-as.matrix(X)
    Y<-as.matrix(Y)
    T<-nrow(X)
    N<-ncol(X)
    bt<- matrix(0,ncol=1,nrow=N)
    At<- diag(0,N)
    pred<- matrix(0,nrow=T,ncol=1)
    theta0<- rep(1,N)
    a<- 1/T
    for (t in 1:T){
      xt<-X[t,]
      pred[t] <- crossprod(as.matrix(theta0), xt)
      Dt <- diag(sqrt(abs(c(theta0))))
      D <- outer(diag(Dt),diag(Dt))
      At <- At + tcrossprod(xt,xt)
      InvA <-  chol2inv(chol(diag(a,N) + D * At))
      AAt<- D * InvA 
      bt <- bt + (Y[t] * xt)
      theta0 <- crossprod(AAt,bt) 
    }
    res<-postResample(pred = pred, obs = Y)
    stats<- as.matrix(res)
    quant<-quantile(as.matrix(Y)-as.matrix(pred),probs=c(.25,.50,.75))
    
    return(list(predictions=pred,performance=stats,quantiles=quant))
  
}

CIRR<-function(X,Y){
    X<-as.matrix(X)
    Y<-as.matrix(Y)
    T<-nrow(X)
    N<-ncol(X)
    bt<- matrix(0,ncol=1,nrow=N)
    At<- diag(0,N)
    pred<- matrix(0,nrow=T,ncol=1)
    theta0<- rep(1,N)
    a<- 1/T
    for (t in 1:T){
      xt<-X[t,]
      Dt <- diag(sqrt(abs(c(theta0))))
      D <- outer(diag(Dt),diag(Dt)) 
      At <- At + tcrossprod(xt,xt)
      InvA <-  chol2inv(chol(diag(a,N) + D * At))
      AAt<- D * InvA 
      theta0<- crossprod(AAt,bt) 
      pred[t] <- crossprod(as.matrix(theta0), xt)
      bt <- bt + (Y[t] * xt)
      theta0 <- crossprod(AAt,bt)
    }
    res<-postResample(pred = pred, obs = Y)
    stats<- as.matrix(res)
    quant<-quantile(as.matrix(Y)-as.matrix(pred),probs=c(.25,.50,.75))
    
    return(list(predictions=pred,performance=stats,quantiles=quant))
}

BLUE<- function(X,Y){
  pred<-predict(lm(Y~as.matrix(X)))
  res<-postResample(pred = pred, obs = Y)
  stats<- as.matrix(res)
  quant<-quantile(as.matrix(Y)-as.matrix(pred),probs=c(.25,.50,.75))
  return(list(predictions=pred,performance=stats,quantiles=quant))
}


#1-Gaze

train<- read.table("gaze_train.inputs",F)
valid<- read.table("gaze_valid.inputs",F)
trainTarget<-read.table("gaze_train.targets",F)
validTarget<-read.table("gaze_valid.targets",F)

names(train) <- names(valid)
names(trainTarget) <- names(validTarget)
X<-as.matrix(rbind(train,valid))
Y<-as.matrix(rbind(trainTarget,validTarget))

#Gaze Data
p<-c(0,Y[1:449])
res<-postResample(pred = p, obs = Y)
stats<- as.matrix(res)
quant<-quantile(Y-p,probs=c(.25,.50,.75))


m1 <- lm(Y ~X )  #Create a linear model
resid(m1) #List of residuals
plot(density(resid(m1))) 
qqnorm(resid(m1)) # A quantile normal plot - good for checking normality
qqline(resid(m1))

one<- NGD(X,Y)
two<- CNLS(X,Y)
three<- AROWR(X,Y)
four<-RLS(X,Y)
five<-ORR(X,Y)
six<-AAR(X,Y)
seven<-ONS(X,Y)
eight<-OSLOG(X,Y)
nine<- CIRR(X,Y)
ten<-BLUE(X,Y)
#2-F16

load(file = "F16.rda")
f16<- F16
X<- as.matrix((f16[,1:40]))
Y<- as.matrix(f16[,41])

p<-c(0,Y[1:length(Y)-1])
res<-postResample(pred = p, obs = Y)
stats<- as.matrix(res)
quant<-quantile(Y-p,probs=c(.25,.50,.75))


m1 <- lm(Y ~X )  #Create a linear model
resid(m1) #List of residuals
plot(density(resid(m1))) 

one<- NGD(X,Y)
two<- CNLS(X,Y)
three<- AROWR(X,Y)
four<-RLS(X,Y)
five<-ORR(X,Y)
six<-AAR(X,Y)
seven<-ONS(X,Y)
eight<-OSLOG(X,Y)
nine<- CIRR(X,Y)
ten<-BLUE(X,Y)


#3-NO2

data<- read.table("NO2.dat",F)
data<- data[order(data[,8],data[,7]),]
data<- as.matrix(data[,-c(8)])
data<- scale(data)
#data <- data[-c(169), ]
X<- as.matrix(data[,-c(1)])
#bias<- rep(1,500)
#X<-cbind(X,bias)
X<-data.frame(X)
X<-cbind(rep(1,500),X)
Y<- as.matrix(data[,1])

p<-c(0,Y[1:length(Y)-1])
res<-postResample(pred = p, obs = Y)
stats<- as.matrix(res)
quant<-quantile(Y-p,probs=c(.25,.50,.75))


m1 <- lm(Y ~X )  #Create a linear model
resid(m1) #List of residuals
plot(density(resid(m1))) 

one<- NGD(X,Y)
two<- CNLS(X,Y)
three<- AROWR(X,Y)
four<-RLS(X,Y)
five<-ORR(X,Y)
six<-AAR(X,Y)
seven<-ONS(X,Y)
eight<-OSLOG(X,Y)
nine<- CIRR(X,Y)
ten<-BLUE(X,Y)



#4-ISE
load(file = "ISE.rda")
data<- ISE
X<- as.matrix(data[,-c(1,3)])
Y<- as.matrix(data[,3])

p<-c(0,Y[1:length(Y)-1])
res<-postResample(pred = p, obs = Y)
stats<- as.matrix(res)
quant<-quantile(Y-p,probs=c(.25,.50,.75))


m1 <- lm(Y ~X )  #Create a linear model
resid(m1) #List of residuals
plot(density(resid(m1))) 

one<- NGD(X,Y)
two<- CNLS(X,Y)
three<- AROWR(X,Y)
four<-RLS(X,Y)
five<-ORR(X,Y)
six<-AAR(X,Y)
seven<-ONS(X,Y)
eight<-OSLOG(X,Y)
nine<- CIRR(X,Y)
ten<-BLUE(X,Y)


#5-Brest Cancer

Data<- read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data",header = F,sep = ",")
X <- as.matrix(Data[,c(-1,-2,-34,-35)])
Y <-as.matrix(Data[,34])

mean(Y)
var(Y)


one<- NGD(X,Y)
two<- CNLS(X,Y)
three<- AROWR(X,Y)
four<-RLS(X,Y)
five<-ORR(X,Y)
six<-AAR(X,Y)
seven<-ONS(X,Y)
eight<-OSLOG(X,Y)
nine<- CIRR(X,Y)
ten<-BLUE(X,Y)

ncol(X)

# 6 - Fraidman 
load(file = "Fried.rda")
fried<- Fried
X<- as.matrix((fried[,1:10]))
Y<- as.matrix(fried[,11])

mean(Y)
var(Y)


m1 <- lm(Y ~X )  #Create a linear model
resid(m1) #List of residuals
plot(density(resid(m1))) 
summary(m1)$sigma^2

median(cooks.distance(m1))
min(cooks.distance(m1))
max(cooks.distance(m1))

one<- NGD(X,Y)
two<- CNLS(X,Y)
three<- AROWR(X,Y)
four<-RLS(X,Y)
five<-ORR(X,Y)
six<-AAR(X,Y)
seven<-ONS(X,Y)
eight<-OSLOG(X,Y)
nine<- CIRR(X,Y)
ten<-BLUE(X,Y)
eleven<- Adam(X,Y)

#7 - Weather - kaggle 

weather<- read.csv("weatherHistory.csv", header = T)

w<- weather[,2:11]
X<-w[,-4]
Y<-w[,4]
x1<- as.numeric(factor(X[,1]))
x2<- as.numeric(factor(X[,2]))
X<- cbind(x1,x2,X[,3:9])
m1 <- lm(Y ~as.matrix(X) )  #Create a linear model
summary(m1)$sigma^2

resid(m1) #List of residuals
plot(density(resid(m1))) 


median(cooks.distance(m1))
min(cooks.distance(m1))
max(cooks.distance(m1))

mean(Y)
var(Y)

x1<- as.numeric(factor(X[,1]))
x2<- as.numeric(factor(X[,2]))
X<- cbind(x1,x2,X[,3:9])


one<- NGD(X,Y)
two<- CNLS(X,Y)
#three<- AROWR(X,Y)
#four<-RLS(X,Y)
five<-ORR(X,Y)
#six<-AAR(X,Y)
seven<-ONS(X,Y)
#eight<-OSLOG(X,Y)
#nine<- CIRR(X,Y)
ten<-BLUE(X,Y)
eleven<- Adam(X,Y)


##############################
#Code doesn't improve
###########################
AROWR<-function(X,Y){
  X<- as.matrix(X)
  Y<-as.matrix(Y)
  N<- ncol(X)
  T<- nrow(X)
  bt<-matrix(0,ncol=1,nrow=N)
  At<-diag(1, N)
  pred<-matrix(0,ncol=1,nrow=T)
  a<- 1/T
  for (t in 1:T){
    xt<-X[t,]
    pred[t,]<- tcrossprod(crossprod(bt,At),xt)
    At<- At + (tcrossprod(xt,xt) * 1/a)
    InvA <- At - ((t(xt%*%At)%*%(as.matrix(xt%*%At)))
                  /as.numeric(xt%*%At%*%xt+a))
    bt <- bt + (Y[t] * xt)
    theta<- InvA %*% bt
  }
  res<-postResample(pred = pred, obs = Y)
  stats<- as.matrix(res)
  quant<-quantile(Y-pred,probs=c(.25,.50,.75))
  
  return(list(predictions=pred,performance=stats,quantiles=quant))
}


RLS<-function(X,Y){
  X<- as.matrix(X)
  Y<-as.matrix(Y)
  N<- ncol(X)
  T<- nrow(X)
  bt<-matrix(0,ncol=1,nrow=N)
  At<-diag(1, N)
  pred<-matrix(0,ncol=1,nrow=T)
  a<- 1/T
  for (t in 1:T){
    xt<-X[t,]
    pred[t,]<- tcrossprod(crossprod(bt,At),xt)
    At<- (a*At) + tcrossprod(xt,xt)
    InvA <- At - ((t(xt%*%At)%*%(as.matrix(xt%*%At)))
                  /as.numeric(xt%*%At%*%xt+a))
    bt <- bt + (Y[t] * xt)
    theta<- InvA %*% bt
  }
  res<-postResample(pred = pred, obs = Y)
  stats<- as.matrix(res)
  quant<-quantile(Y-pred,probs=c(.25,.50,.75))
  
  return(list(predictions=pred,performance=stats,quantiles=quant))
}
###Misc

train_set<- read.csv("train_2v.csv")
test_set<- read.csv("test_2v.csv")

factors<-sapply(train_set,is.factor)       
train_set2<-sapply(train_set[,factors],unclass)  
train_set<-cbind(train_set[,!factors],train_set2)



train_set<- na.omit(train_set)
X <- train_set[,-c(7)]
Y <- train_set[,7]


one<- NGD(X,Y)
two<- CNLS(X,Y)
three<- AROWR(X,Y)
four<-RLS(X,Y)
five<-ORR(X,Y)
six<-AAR(X,Y)
seven<-ONS(X,Y)
eight<-OSLOG(X,Y)
nine<- CIRR(X,Y)
ten<-BLUE(X,Y)


############################
#Loss function plots
l2<- function(x){x*x}
l1<-function(x){abs(x)}
normalisel2<- function(x){x*x/norm(x,"2")}
parameterl2<- function(x,ell){x*x/(norm(x,"2")+ell)}
x<-seq(-1.9,1.9,0.1)
library(ggplot2)
data<- data.frame(l2(x),l1(x),normalisel2(x),x)
colnames(data)<-c("SquareLoss","AbsoluteLoss","NormalisedSquareLoss","x")
write.csv(data,"lossfun.csv")
p<-ggplot(data, aes(x)) + 
  geom_line(aes(y = AbsoluteLoss,linetype = "Absolute Loss")) + 
  geom_line(aes(y = SquareLoss,linetype = "Squared Loss"))+
  geom_line(aes(y = NormalisedSquareLoss,linetype = "Normalised Squared Loss")) +
  labs(title="Loss Functions",y = "f(x)")+theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                                                             panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))
data1<- data.frame(parameterl2(x,-5),parameterl2(x,0),parameterl2(x,5),x)
colnames(data1)<-c("parameterm20","parameter0","parameter20","x")  
write.csv(data1,"Myloss.csv")
p1<-ggplot(data1, aes(x)) + 
  geom_line(aes(y = parameterm20, linetype = "eta = -0.9||x||^2")) + 
  geom_line(aes(y = parameter0, linetype = "eta = 0"))+
  geom_line(aes(y = parameter20, linetype= "eta = ||x||^2")) +
  labs(title="Tunable Normalised Squared Loss",y = "f(x)")+theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                                                                              panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))+ylim(0,3)

require(gridExtra)
grid.arrange(p, p1, ncol=1,nrow=2)

