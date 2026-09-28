Kernel Density Estimation and Nonparametric Regression
================
Patrick Mellady

- [Kernels](#kernels)
  - [Multivariate Kernels](#multivariate-kernels)
- [Kernel density estimation](#kernel-density-estimation)
- [Kernel regression](#kernel-regression)
  - [Bandwidth selection by leave-one-out
    cross-validation](#bandwidth-selection-by-leave-one-out-cross-validation)
- [Implementation](#implementation)
  - [Example 1: Gaussian errors](#example-1-gaussian-errors)
  - [Example 2: Binary responses](#example-2-binary-responses)
  - [Example 3: Poisson responses](#example-3-poisson-responses)
  - [Example 4: Multiple covariates (Gaussian
    errors)](#example-4-multiple-covariates-gaussian-errors)
  - [Example 5: Multiple covariates (Poisson
    responses)](#example-5-multiple-covariates-poisson-responses)

This repository derives kernel density estimators and the
Nadaraya–Watson kernel regression estimator, implements both from
scratch, and checks the implementation against R’s `density` and
`ksmooth` functions.

# Kernels

A *kernel* is a function $K:\mathbb{R}\to\mathbb{R}$ such that

$$K(x)\ge 0,\qquad K(-x)=K(x),\qquad \int K(x)\,dx=1,\qquad \int x^2K(x)\,dx<\infty .$$

Since $|x|\le (1+x^2)/2$, the finite second moment implies
$\int |x|K(x)\,dx<\infty$, and symmetry then gives $\int xK(x)\,dx=0$.

Given a *bandwidth* $h>0$, define the rescaled kernel

$$K_h(x)=\frac{1}{h}K\!\left(\frac{x}{h}\right).$$

Two facts about $K_h$ that we will repeatedly use are presented below:

$$\int K_h(x)\,dx=\int K(u)\,du=1,
\qquad
\int xK_h(x)\,dx=h\int uK(u)\,du=0 .$$

## Multivariate Kernels

For $x\in\mathbb{R}^d$ and a symmetric positive definite *bandwidth
matrix* $H$, define

$$K_H(x)=\frac{1}{|\det H|}K\!\left(H^{-1}x\right).$$

Here $K:\mathbb{R}^d\to\mathbb{R}$ must itself be a $d$-dimensional
kernel. A common choice is a product kernel built from a univariate
kernel $K$, with a diagonal bandwidth matrix
$H=\text{diag}(h_1,\dots,h_d)$:

$$K_H(x)=\prod_{j=1}^{d}\frac{1}{h_j}K\!\left(\frac{x_j}{h_j}\right).$$

With $H=hI$ this reduces to $h^{-d}\prod_{j=1}^d K(x_j/h)$.

# Kernel density estimation

Suppose $Y_1,\dots,Y_n$ are i.i.d. from a distribution with unknown
density $p$. The kernel density estimate is

$$\hat p(y)=\frac1n\sum_{i=1}^n K_h(Y_i-y).$$

To see that this is a valid estimator for the density, see that
$\hat p\ge0$ because $K\ge 0$, and

$$\int\hat p(y)\,dy=\frac1n\sum_{i=1}^n\int K_h(Y_i-y)\,dy=\frac1n\sum_{i=1}^n 1=1,$$

so $\hat p$ satisfies the definition of a density function.

# Kernel regression

We want to estimate the regression defined by $r(x)=E(Y\mid X=x)$. We
start by expanding the conditional expectation as follows

$$r(x)=\int y\,f(y\mid x)\,dy=\int y\,\frac{f(x,y)}{f(x)}\,dy .$$

If we then estimate the joint density with a product kernel:

$$\hat f(x,y)=\frac1n\sum_{i=1}^nK_h(X_i-x)K_h(Y_i-y).$$

and determine the marginal of $x$ by Integrating out $y$:

$$\int\hat f(x,y)\,dy=\frac1n\sum_{i=1}^nK_h(X_i-x)\int K_h(Y_i-y)\,dy=\frac1n\sum_{i=1}^nK_h(X_i-x)=\hat f(x),$$

We obtain the following

$$\int y\frac{\hat f(x,y)}{\hat f(x)}dy=\frac{\sum_{i=1}^nK_h(X_i-x)\int yK_h(Y_i-y)\,dy}{\sum_{i=1}^nK_h(X_i-x)}$$

By making the substitution $z=y-Y_i$ and using kernel properties:

$$\int yK_h(Y_i-y)\,dy=\int (z+Y_i)K_h(z)\,dz=\underbrace{\int zK_h(z)\,dz}_{=0}+Y_i\underbrace{\int K_h(z)\,dz}_{=1}=Y_i .$$

Which yields the final form of the estimator below:

$$\hat r(x)=\sum_{i=1}^nw_i(x)Y_i,
\qquad
w_i(x)=\frac{K_h(X_i-x)}{\sum_{j=1}^nK_h(X_j-x)} .$$

## Bandwidth selection by leave-one-out cross-validation

Let $K_{ij}=K_h(X_i-X_j)$ and $w_{ij}=K_{ij}/\sum_{l}K_{il}$, so that
$\hat r(X_i)=\sum_jw_{ij}Y_j$. The leave-one-out fit at $X_i$ drops
observation $i$ and renormalizes:

$$\hat r_{-i}(X_i)=\frac{\sum_{j\ne i}K_{ij}Y_j}{\sum_{j\ne i}K_{ij}}
=\frac{\sum_{j\ne i}w_{ij}Y_j}{1-w_{ii}}
=\frac{\hat r(X_i)-w_{ii}Y_i}{1-w_{ii}} .$$

We then calculate the residual by subtracting the above from $Y_i$,

$$Y_i-\hat r_{-i}(X_i)=\frac{Y_i(1-w_{ii})-\hat r(X_i)+w_{ii}Y_i}{1-w_{ii}}=\frac{Y_i-\hat r(X_i)}{1-w_{ii}} .$$

Hence the cross-validation score can be computed from a single fit, with
no refitting:

$$\mathrm{CV}(h)=\frac1n\sum_{i=1}^n\left(\frac{Y_i-\hat r(X_i)}{1-w_{ii}}\right)^2,
\qquad \hat h=\arg\min_h \mathrm{CV}(h).$$

For densities we will use Silverman’s rule while for regression we will
use Scott’s rule, $h_j=\hat\sigma_j\,n^{-1/(d+4)}$.

# Implementation

Below, we implement a general purpose kernel function capable of
switching between three different popular choices of kernels.

``` r
# Kernel functions
k<-function(x, type=c("gaussian", "epanechnikov", "logistic")){
  type<-match.arg(type)
  switch(type,
    gaussian=dnorm(x),
    epanechnikov=0.75*(1-x^2)*(abs(x)<=1),
    logistic=dlogis(x)
  )
}
```

Next, we implement the kernel regression estimator, the kernel density
estimator, and cross-validation for the bandwidth. The regression
function accepts a vector or a matrix of covariates and uses a product
kernel with one bandwidth per coordinate. It returns `NA` at any point
where no observation has positive kernel weight.

``` r
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

# Kernel density estimator on grid
kde<-function(Y, h, type="gaussian", grid=NULL, n_grid=512){
  if(is.null(grid)) grid<-seq(min(Y)-3*h, max(Y)+3*h, length.out=n_grid)
  p_hat<-vapply(grid, function(y) mean(k((y-Y)/h, type))/h, numeric(1))
  list(x=grid, p_hat=p_hat)
}

# Leave-one-out CV 
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

# Function to plot regression fits
plot_fit<-function(x, y, grid, fit, truth, title, ylab="Y"){
  ggplot()+geom_point(aes(x=x, y=y), alpha=.5)+
    geom_line(aes(x=grid, y=truth, color="True Mean"), linewidth=.8)+
    geom_line(aes(x=grid, y=fit, color="Kernel Regression"), linewidth=.8)+
    scale_color_manual(name="Regression Type",
                       values=c("True Mean"="blue", "Kernel Regression"="orange"))+
    labs(title=title)+xlab("X")+ylab(ylab)
}
```

## Example 1: Gaussian errors

We first examine normal data of the form $Y_i=\mu(x_i)+\varepsilon_i$
with $\mu(x)=\sqrt{x}+xe^{-x}\sin x$ and $\epsilon_i\sim N(0,.5)$.

``` r
n<-100
X<-seq(1, 20, length.out=n)
m1<-function(x) sqrt(x)+x*exp(-x)*sin(x)
Y<-m1(X)+rnorm(n, 0, .25)
```

We choose the bandwidth by leave-one-out cross-validation.

``` r
cv1<-cv_bandwidth(X, Y, hs=seq(.1, 2, length.out=60))
h1<-cv1$h
```

The kernel regression then gives the following estimates

``` r
grid1<-seq(min(X), max(X), length.out=200)
fit1<-r(X, Y, grid1, h1)
plot_fit(X, Y, grid1, fit1, m1(grid1), "Kernel Regression on Gaussian Data")
```

![](kernel_regression_files/figure-gfm/ex1-reg-1.png)<!-- -->

and the kernel density estimate of the $Y_i$s is shown below, along with
the mixture density $\bar p(y)=\frac1n\sum_i N(y;\,m(x_i),0.25^2)$.

``` r
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
```

![](kernel_regression_files/figure-gfm/ex1-kde-1.png)<!-- -->

## Example 2: Binary responses

We now do an example with binary data, $Y_i\sim\text{Bernoulli}(p_i)$
with $p_i=\text{expit}(5.01\,x_i)$.

``` r
n<-100
x<-seq(-2, 2, length.out=n)
prob<-plogis(5.01*x)
Y<-rbinom(n, 1, prob)
```

The kernel regression estimator is

``` r
h2<-cv_bandwidth(x, Y, hs=seq(.05, 1, length.out=40))$h
h2
```

    ## [1] 0.2935897

``` r
grid2<-seq(min(x), max(x), length.out=200)
fit2<-r(x, Y, grid2, h2)
plot_fit(x, Y, grid2, fit2, plogis(5.01*grid2), "Kernel Regression on Binary Data")
```

![](kernel_regression_files/figure-gfm/ex2-reg-1.png)<!-- -->

Although densities and mass functions are different quantities, we
compare the kernel density estimator to the true mass function below.

``` r
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
```

![](kernel_regression_files/figure-gfm/ex2-kde-1.png)<!-- -->

## Example 3: Poisson responses

In this example, we look at Poisson distributed data,
$Y_i\sim\text{Poisson}(\lambda_i)$ with
$\lambda_i=\exp(0.922-0.501\,x_i)$. The kernel regression targets
$E(Y\mid x)=\lambda(x)$.

``` r
n<-100
x<-seq(-2, 2, length.out=n)
lam_fun<-function(x) exp(.922-.501*x)
lam<-lam_fun(x)
Y<-rpois(n, lam)
```

The regression estimator gives

``` r
h3<-cv_bandwidth(x, Y, hs=seq(.05, 1.5, length.out=40))$h
h3
```

    ## [1] 0.4589744

``` r
grid3<-seq(min(x), max(x), length.out=200)
fit3<-r(x, Y, grid3, h3)
plot_fit(x, Y, grid3, fit3, lam_fun(grid3), "Kernel Regression on Poisson Data")
```

![](kernel_regression_files/figure-gfm/ex3-reg-1.png)<!-- -->

and, as in Example 2, we compare the density estimate of the $Y_i$s with
the mass function, despite the incompatibilities between the two.

``` r
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
```

![](kernel_regression_files/figure-gfm/ex3-kde-1.png)<!-- -->

## Example 4: Multiple covariates (Gaussian errors)

We now implement a multivariate kernel regression estimator with $p=5$
covariates. We define the mean function below

$$\mu(z)=5.01+9.22z_1^2-12.22z_2^2+z_3+2z_4^2,\qquad Y=\mu(z)+\epsilon,\ \ \epsilon\sim N(0,50^2).$$

``` r
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
```

We visualize the fit along a diagonal slice through the covariates,
where each coordinate moves from its minimum to its maximum
simultaneously. The curves below are the estimate and the truth along
that path, plotted against each coordinate in turn.

``` r
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
```

![](kernel_regression_files/figure-gfm/ex4-fit-1.png)<!-- -->

## Example 5: Multiple covariates (Poisson responses)

Further, we can examine how Poisson distributed data is fit with several
covariates. Here $\lambda(z)=\exp(-1+0.3z_1^2+0.2z_2)$. As in the last
example, we evaluate along a diagonal slice of the covariates.

``` r
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
```

![](kernel_regression_files/figure-gfm/ex5-1.png)<!-- -->
