# Set-up -----------------------------------------------------------------------
library(ggplot2)
set.seed(405)

# Functions used for fitting ---------------------------------------------------
## Define kernel functions
k<-function(x, type=c("gaussian", "epanechnikov", "logistic")){
  type<-match.arg(type)
  switch(type,
         gaussian=dnorm(x),
         epanechnikov=0.75*(1-x^2)*(abs(x)<=1),
         logistic=dlogis(x)
  )
}

# Regression estimator
r<-function(X, Y, at, h, type="gaussian"){
  X<-as.matrix(X)
  at<-as.matrix(at)
  stopifnot(ncol(X)==ncol(at), length(h) %in% c(1, ncol(X)), nrow(X)==length(Y))
  h<-rep_len(h, ncol(X))
  vapply(seq_len(nrow(at)),
         function(i){
           K<-rep(1, nrow(X))
           for(j in seq_len(ncol(X))){
             K<-K*k((X[,j]-at[i,j])/h[j], type)
           }
           s<-sum(K)
           if(s>0) sum(K*Y)/s else NA
         },
         numeric(1))
}

## Function for kernel density estimation
kde<-function(Y, h, type="gaussian", grid=NULL, n_grid=512){
  if(is.null(grid)) grid<-seq(min(Y)-3*h, max(Y)+3*h, length.out=n_grid)
  p_hat<-vapply(grid, function(y) mean(k((y-Y)/h, type))/h, numeric(1))
  list(x=grid, p_hat=p_hat)
}

## Function for leave-one-out CV 
cv_bandwidth<-function(X, Y, hs){
  D<-outer(X, X, "-")
  cv<-vapply(hs, function(h){
    Kmat<-dnorm(D/h)
    W<-Kmat/rowSums(Kmat)
    rhat<-as.vector(W%*%Y)
    mean(((Y-rhat)/(1-diag(W)))^2)
  }, numeric(1))
  list(h=hs[which.min(cv)], hs=hs, cv=cv)
}

## Function to plot regression fits
plot_fit<-function(x, y, grid, fit, truth, title, ylab="Y"){
  ggplot()+geom_point(aes(x=x, y=y), alpha=.5)+
    geom_line(aes(x=grid, y=truth, color="True Mean"), linewidth=.8)+
    geom_line(aes(x=grid, y=fit, color="Kernel Regression"), linewidth=.8)+
    scale_color_manual(name="Regression Type",
                       values=c("True Mean"="blue", "Kernel Regression"="orange"))+
    labs(title=title)+xlab("X")+ylab(ylab)
}

# Example 1 --------------------------------------------------------------------
n<-100
X<-seq(1, 20, length.out=n)
m1<-function(x) sqrt(x)+x*exp(-x)*sin(x)
Y<-m1(X)+rnorm(n, 0, .25)

cv1<-cv_bandwidth(X, Y, hs=seq(.1, 2, length.out=60))
h1<-cv1$h

grid1<-seq(min(X), max(X), length.out=200)
fit1<-r(X, Y, grid1, h1)
plot_fit(X, Y, grid1, fit1, m1(grid1), "Kernel Regression on Gaussian Data")

h_kde1<-bw.nrd0(Y)
pk1<-kde(Y, h_kde1)
mix1<-vapply(pk1$x, function(y) mean(dnorm(y, m1(X), .25)), numeric(1))

ggplot()+geom_line(aes(x=pk1$x, y=mix1, color="true"))+
  geom_line(aes(x=pk1$x, y=pk1$p_hat, color="kernel"))+
  labs(title="Kernel Density Estimate of Gaussian Data",
       subtitle="with mixture density overlay")+
  xlab("Y")+ylab("Density")+scale_color_manual(name="Density Type",
                                               values=c("kernel"="orange",
                                                        "true"="blue"))


# Example 2 --------------------------------------------------------------------
n<-100
x<-seq(-2, 2, length.out=n)
prob<-plogis(5.01*x)
Y<-rbinom(n, 1, prob)

h2<-cv_bandwidth(x, Y, hs=seq(.05, 1, length.out=40))$h
h2
grid2<-seq(min(x), max(x), length.out=200)
fit2<-r(x, Y, grid2, h2)
plot_fit(x, Y, grid2, fit2, plogis(5.01*grid2), "Kernel Regression on Binary Data")

h_kde2<-bw.nrd0(Y)
pk2<-kde(Y, h_kde2)
pmf2<-c(1-mean(prob), mean(prob))

ggplot()+geom_line(aes(x=pk2$x, y=pk2$p_hat, color="kernel"))+
  geom_segment(aes(x=c(0, 1), xend=c(0, 1), y=0, yend=pmf2, color="true"),
               linewidth=1.2)+
  labs(title="Kernel Density Estimate of Binary Data",
       subtitle="with mixture mass overlay (scales not comparable)")+
  xlab("Y")+ylab("Density / Mass")+scale_color_manual(name="Type",
                                                      values=c("kernel"="orange",
                                                               "true"="blue"))


# Example 3 --------------------------------------------------------------------
n<-100
x<-seq(-2, 2, length.out=n)
lam_fun<-function(x) exp(.922-.501*x)
lam<-lam_fun(x)
Y<-rpois(n, lam)

h3<-cv_bandwidth(x, Y, hs=seq(.05, 1.5, length.out=40))$h
h3
grid3<-seq(min(x), max(x), length.out=200)
fit3<-r(x, Y, grid3, h3)
plot_fit(x, Y, grid3, fit3, lam_fun(grid3), "Kernel Regression on Poisson Data")

h_kde3<-bw.nrd0(Y)
pk3<-kde(Y, h_kde3)
support3<-0:max(Y)
pmf3<-vapply(support3, function(y) mean(dpois(y, lam)), numeric(1))

ggplot()+geom_line(aes(x=pk3$x, y=pk3$p_hat, color="kernel"))+
  geom_point(aes(x=support3, y=pmf3, color="true"))+
  labs(title="Kernel Density Estimate of Poisson Data",
       subtitle="with mixture mass overlay (scales not comparable)")+
  xlab("Y")+ylab("Density / Mass")+scale_color_manual(name="Type",
                                                      values=c("kernel"="orange",
                                                               "true"="blue"))


# Example 4 --------------------------------------------------------------------
n<-100
b<-c(5.01, 9.22, -12.22, 1, 2)
x1<-seq(1, 2, length.out=n)
x2<-seq(5.01, 9.22, length.out=n)
x3<-x1+rgamma(n, 1, 8)
x4<-x2+rgamma(n, 1, 4)
Z<-cbind(x1=x1, x2=x2, x3=x3, x4=x4)
m4<-function(Z) b[1]+b[2]*Z[,1]^2+b[3]*Z[,2]^2+b[4]*Z[,3]+b[5]*Z[,4]^2
Y<-m4(Z)+rnorm(n, 0, 50)

h4<-apply(Z, 2, sd)*n^(-1/(ncol(Z)+4))

G<-apply(Z, 2, function(z) seq(min(z), max(z), length.out=100))
fit4<-r(Z, Y, G, h4)
truth4<-m4(G)

vars<-colnames(Z)
df_pts<-data.frame(covariate=rep(vars, each=n), x=as.vector(Z), Y=rep(Y, 4))
df_line<-data.frame(covariate=rep(vars, each=100), x=as.vector(G),
                    fit=rep(fit4, 4), truth=rep(truth4, 4))

ggplot()+geom_point(data=df_pts, aes(x=x, y=Y), alpha=.4)+
  geom_line(data=df_line, aes(x=x, y=truth, color="True Mean"), linewidth=.8)+
  geom_line(data=df_line, aes(x=x, y=fit, color="Kernel Regression"), linewidth=.8)+
  facet_wrap(~covariate, scales="free_x", nrow=2)+
  labs(title="Multivariate Kernel Regression on Gaussian Data",
       subtitle="fit and truth along the diagonal slice")+
  xlab("Covariate")+ylab("Y")+scale_color_manual(name="Regression Type",
                                                 values=c("True Mean"="blue",
                                                          "Kernel Regression"="orange"))



# Example 5 --------------------------------------------------------------------
n<-100
x1<-seq(1, 2, length.out=n)
x2<-seq(5.01, 9.22, length.out=n)
Z<-cbind(x1=x1, x2=x2)
m5<-function(Z) exp(-1+.3*Z[,1]^2+.2*Z[,2])
Y<-rpois(n, m5(Z))

h5<-apply(Z, 2, sd)*n^(-1/(ncol(Z)+4))
G<-apply(Z, 2, function(z) seq(min(z), max(z), length.out=100))
fit5<-r(Z, Y, G, h5)
truth5<-m5(G)

vars<-colnames(Z)
df_pts<-data.frame(covariate=rep(vars, each=n), x=as.vector(Z), Y=rep(Y, 2))
df_line<-data.frame(covariate=rep(vars, each=100), x=as.vector(G),
                    fit=rep(fit5, 2), truth=rep(truth5, 2))

ggplot()+geom_point(data=df_pts, aes(x=x, y=Y), alpha=.4)+
  geom_line(data=df_line, aes(x=x, y=truth, color="True Mean"), linewidth=.8)+
  geom_line(data=df_line, aes(x=x, y=fit, color="Kernel Regression"), linewidth=.8)+
  facet_wrap(~covariate, scales="free_x")+
  labs(title="Multivariate Kernel Regression on Poisson Data",
       subtitle="fit and truth along the diagonal slice")+
  xlab("Covariate")+ylab("Y")+scale_color_manual(name="Regression Type",
                                                 values=c("True Mean"="blue",
                                                          "Kernel Regression"="orange"))










