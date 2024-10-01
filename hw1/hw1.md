# 1.Poisson Regression

## (a)
$$
\begin{align*}
p(y; \lambda) &= \frac{e^{-\lambda} \lambda^y}{y!} \\
              &= \frac{1}{y!} \exp(\log(\lambda^y) - \lambda) \\
              &= \frac{1}{y!} \exp(y \log \lambda - \lambda)
\end{align*}
$$
取$$b(y)=\frac{1}{y},\eta=log\lambda,\lambda=a(\eta)=exp(\eta)$$
则有$$p(y;\eta)=\frac{1}{y!}exp(\eta^Ty-exp(\eta))$$
## (b)
$$
\begin{align*}
g(\eta) &= E[T(y); \eta] \\
        &= E[p(y; \lambda)] \\
        &= \lambda \\
        &= \exp(\eta)
\end{align*}
$$
## (c)
$$
\begin{align*}
\ell(\theta) &= \log(p(y^{(i)}|x^{(i)}, \theta)) \\
             &= \log(p(y^{(i)}; \eta)) \\
             &= \log\left(\frac{1}{y^{(i)}!}\right) + \eta^T y^{(i)} - \exp(\eta) \\
             &= \log\left(\frac{1}{y^{(i)}!}\right) + \theta^T x^{(i)} y^{(i)} - \exp(\theta^T x^{(i)})
\end{align*}
$$
$$
\begin{align*}\frac{\partial \ell(\theta)}{\partial \theta}&=\frac{\partial}{\partial \theta}(log(\frac{1}{y^{(i)}!})+\theta^Tx^{(i)}y^{(i)}-exp(\theta^Tx^{(i)}))\\&=x^{(i)}y^{(i)}-x^{(i)}exp(\theta^Tx^{(i)})\\&=y^{(i)} - h_\theta (x^{(i)})\end{align*}$$

$$\begin{align*}\therefore\theta &:= \theta + \alpha \ \big( y^{(i)} - h_\theta (x^{(i)}) \big) \ x^{(i)}\\&:=\theta + \alpha \ \big( x^{(i)}y^{(i)}-x^{(i)}exp(\theta^Tx^{(i)})) \ x^{(i)}\end{align*}$$

#  2. Convexity of Generalized Linear Models
## (a)
由
$$\begin{align*}
E(Y;\eta)&=\int yp(y;\eta)dy\\
\end{align*}$$
得
$$\begin{align*}
\frac{\partial}{\partial \eta}\int p(y;\eta)dy&=\int \frac{\partial}{\partial \eta} p(y;\eta)dy\\
&=\int \frac{\partial}{\partial \eta} \left( b(y)exp(\eta y −a(\eta))\right)dy\\
&=\int b(y)exp(\eta y −a(\eta))\left( y - \frac{\partial a(\eta)}{\partial \eta}\right)dy\\
&=\int p(y;\eta)\left( y - \frac{\partial a(\eta)}{\partial \eta}\right)dy\\
&=E(Y;\eta)-\int p(y;\eta)\frac{\partial a(\eta)}{\partial \eta}dy
\end{align*}$$
又由
$$\int p(y;\eta)dy=1$$
$$\begin{align*}\int p(y;\eta)\frac{\partial a(\eta)}{\partial \eta}dy&=\frac{\partial a(\eta)}{\partial \eta}\int p(y;\eta)dy\\
&=\frac{\partial a(\eta)}{\partial \eta}
\end{align*}$$
得
$$E(Y;\eta)=\frac{\partial a(\eta)}{\partial \eta}$$
## (b)
由(a)得
$$E(Y;\eta)=\frac{\partial a(\eta)}{\partial \eta}$$
$$\begin{align*}
\frac {\partial E(Y;\eta)}{\partial \eta} &=\frac {\partial^2 a(\eta)}{\partial \eta^2} \\
\end{align*}$$
又由
$$\begin{align*}
\frac {\partial E(Y;\eta)}{\partial \eta} &= \frac {\partial}{\partial \eta} \int yp(y;\eta)dy\\
&=\int y\frac {\partial p(y;\eta)}{\partial \eta}dy\\
&=\int yp(y;\eta)\left( y - \frac{\partial a(\eta)}{\partial \eta}\right)dy\\
&=\int p(y;\eta)\left( y^2 - y\frac{\partial a(\eta)}{\partial \eta}\right)dy\\
&=E(Y^2;\eta)-E(Y;\eta)\frac{\partial a(\eta)}{\partial \eta}\\
&=E(Y^2;\eta)-E^2(Y;\eta)\\
&=Var(Y;\eta)
\end{align*}$$
得
$$Var(Y;\eta)=\frac {\partial^2 a(\eta)}{\partial \eta^2}$$

## (c)
由
$$
\begin{align*}
\ell (\theta) & = - \log p(y; \ \eta) \\
              & = - \log(b(y) \exp (\eta y - a(\eta)) )\\
              & = a(\eta) - \eta y - \log b(y) \\
              & = a(\theta^T x) - \theta^T x y - \log b(y)
\end{align*}
$$
得
$$\nabla_\theta \ell (\theta) = x \frac{\partial}{\partial \eta} a(\theta^T x) - xy$$

$$H = \nabla_\theta^2 \ell (\theta) = x x^T \frac{\partial^2}{\partial \eta^2} a(\theta^T x)$$

对  $\forall z \in \mathbb{R}^n$, 有
$$\begin{align*}
z^T H z &= z^T x x^T \frac{\partial^2}{\partial \eta^2} a(\theta^T x) z \\
        &= (x^T z)^T \frac{\partial^2}{\partial \eta^2} a(\eta) (x^T z) \\
        &= (x^T z)^2 Var(Y;\eta)\\
        &\geq 0
\end{align*}
$$
$$\therefore H \succeq 0\text{, GLM的NLL损失是凸函数}$$
#  3. Multivariate Least Squares
## (a)
$$\begin{align*}
J(\Theta)& =\frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{p}((\Theta^Tx^{(i)})_j-y^{(i)}_j)^2\\
&= \frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{p}(X\Theta-Y)_{ij}^2\\
&= \frac{1}{2}\sum_{i=1}^{p}[(X\Theta-Y)^T(X\Theta-Y)]_{ii}\\
&= \frac{1}{2}tr[(X\Theta-Y)^T(X\Theta-Y)]\\
\end{align*}
$$

## (b)
