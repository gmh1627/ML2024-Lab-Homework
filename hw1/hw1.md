# 1. Poisson Regression

## (a)
$$
\begin{align*}
p(y; \lambda) &= \frac{e^{-\lambda} \lambda^y}{y!} \\
              &= \frac{1}{y!} \exp(\log(\lambda^y) - \lambda) \\
              &= \frac{1}{y!} \exp(y \log \lambda - \lambda)
\end{align*}
$$
取 
$$
b(y)=\frac{1}{y}, \eta=\log\lambda, \lambda=a(\eta)=\exp(\eta)
$$
则有 
$$
p(y;\eta)=\frac{1}{y!}\exp(\eta^T y - \exp(\eta))
$$

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
\begin{align*}
\frac{\partial \ell(\theta)}{\partial \theta} &= \frac{\partial}{\partial \theta}\left(\log\left(\frac{1}{y^{(i)}!}\right) + \theta^T x^{(i)} y^{(i)} - \exp(\theta^T x^{(i)})\right) \\
&= x^{(i)} y^{(i)} - x^{(i)} \exp(\theta^T x^{(i)}) \\
&= y^{(i)} - h_\theta (x^{(i)})
\end{align*}
$$
$$
\begin{align*}
\therefore \theta &:= \theta + \alpha \ \big( y^{(i)} - h_\theta (x^{(i)}) \big) \ x^{(i)} \\
&:= \theta + \alpha \ \big( x^{(i)} y^{(i)} - x^{(i)} \exp(\theta^T x^{(i)}) \big) \ x^{(i)}
\end{align*}
$$

# 2. Convexity of Generalized Linear Models

## (a)
由
$$
\begin{align*}
E(Y;\eta) &= \int y p(y;\eta) dy
\end{align*}
$$
得
$$
\begin{align*}
\frac{\partial}{\partial \eta} \int p(y;\eta) dy &= \int \frac{\partial}{\partial \eta} p(y;\eta) dy \\
&= \int \frac{\partial}{\partial \eta} \left( b(y) \exp(\eta y - a(\eta)) \right) dy \\
&= \int b(y) \exp(\eta y - a(\eta)) \left( y - \frac{\partial a(\eta)}{\partial \eta} \right) dy \\
&= \int p(y;\eta) \left( y - \frac{\partial a(\eta)}{\partial \eta} \right) dy \\
&= E(Y;\eta) - \int p(y;\eta) \frac{\partial a(\eta)}{\partial \eta} dy
\end{align*}
$$
又由
$$
\int p(y;\eta) dy = 1
$$
$$
\begin{align*}
\int p(y;\eta) \frac{\partial a(\eta)}{\partial \eta} dy &= \frac{\partial a(\eta)}{\partial \eta} \int p(y;\eta) dy \\
&= \frac{\partial a(\eta)}{\partial \eta}
\end{align*}
$$
得
$$
E(Y;\eta) = \frac{\partial a(\eta)}{\partial \eta}
$$

## (b)
由(a)得
$$
E(Y;\eta) = \frac{\partial a(\eta)}{\partial \eta}
$$
$$
\begin{align*}
\frac{\partial E(Y;\eta)}{\partial \eta} &= \frac{\partial^2 a(\eta)}{\partial \eta^2}
\end{align*}
$$
又由
$$
\begin{align*}
\frac{\partial E(Y;\eta)}{\partial \eta} &= \frac{\partial}{\partial \eta} \int y p(y;\eta) dy \\
&= \int y \frac{\partial p(y;\eta)}{\partial \eta} dy \\
&= \int y p(y;\eta) \left( y - \frac{\partial a(\eta)}{\partial \eta} \right) dy \\
&= \int p(y;\eta) \left( y^2 - y \frac{\partial a(\eta)}{\partial \eta} \right) dy \\
&= E(Y^2;\eta) - E(Y;\eta) \frac{\partial a(\eta)}{\partial \eta} \\
&= E(Y^2;\eta) - E^2(Y;\eta) \\
&= Var(Y;\eta)
\end{align*}
$$
得
$$
Var(Y;\eta) = \frac{\partial^2 a(\eta)}{\partial \eta^2}
$$

## (c)
由
$$
\begin{align*}
\ell (\theta) &= - \log p(y; \eta) \\
              &= - \log(b(y) \exp (\eta y - a(\eta))) \\
              &= a(\eta) - \eta y - \log b(y) \\
              &= a(\theta^T x) - \theta^T x y - \log b(y)
\end{align*}
$$
得
$$
\nabla_\theta \ell (\theta) = x \frac{\partial}{\partial \eta} a(\theta^T x) - xy
$$
$$
H = \nabla_\theta^2 \ell (\theta) = x x^T \frac{\partial^2}{\partial \eta^2} a(\theta^T x)
$$
对 $\forall z \in \mathbb{R}^n$, 有
$$
\begin{align*}
z^T H z &= z^T x x^T \frac{\partial^2}{\partial \eta^2} a(\theta^T x) z \\
        &= (x^T z)^T \frac{\partial^2}{\partial \eta^2} a(\eta) (x^T z) \\
        &= (x^T z)^2 Var(Y;\eta) \\
        &\geq 0
\end{align*}
$$
$$
\therefore H \succeq 0\text{, GLM的NLL损失是凸函数}
$$

# 3. Multivariate Least Squares

## (a)
$$
\begin{align*}
J(\Theta) &= \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{p} ((\Theta^T x^{(i)})_j - y^{(i)}_j)^2 \\
          &= \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{p} (X \Theta - Y)_{ij}^2 \\
          &= \frac{1}{2} \sum_{i=1}^{p} [(X \Theta - Y)^T (X \Theta - Y)]_{ii} \\
          &= \frac{1}{2} \operatorname{tr}[(X \Theta - Y)^T (X \Theta - Y)]
\end{align*}
$$

## (b)
$$
\begin{align*}
\frac{\partial J(\Theta)}{\partial \Theta} &= \frac{\partial}{\partial \Theta} \frac{1}{2} \operatorname{tr}[(X \Theta - Y)^T (X \Theta - Y)] \\
&= \frac{1}{2} \frac{\partial}{\partial \Theta} \operatorname{tr}[\Theta^T X^T X \Theta - \Theta^T X^T Y - Y^T X \Theta + Y^T Y] \\
&= \frac{1}{2} \frac{\partial}{\partial \Theta} \operatorname{tr}[\Theta^T X^T X \Theta - 2 \Theta^T X^T Y + Y^T Y] \\
&= \frac{1}{2} (2X^T X\Theta - 2X^T Y) \\
&= X^T X\Theta - X^T Y
\end{align*}
$$

$
\nabla_\Theta J(\Theta) = \mathbf{0}
$时，有
$$
X^T X\Theta - X^T Y = \mathbf{0}
$$
即
$$
\Theta = (X^T X)^{-1} X^T Y
$$

## (c)
由题意得
$$
Y=X(\theta_1, \theta_2, \ldots, \theta_p)
$$
记
$$
Y_j=(y^{(1)}_j, y^{(2)}_j, \ldots, y^{(m)}_j)^T
$$
则
$$
Y_j=X\theta_j
$$
$$
\theta_j= (X^T X)^{-1} X^T Y_j
$$
$$
(\theta_1, \theta_2, \ldots, \theta_p)=(X^T X)^{-1} X^T Y=\Theta
$$
$\therefore$ 与上题结论一致

# 4. Incomplete, Positive-Only Labels
## (a)
$$
\begin{align*}
p(t^{(i)} = 1 | y^{(i)} = 1,x^{(i)}) &= \frac{p(y^{(i)} = 1 | t^{(i)} = 1,x^{(i)}) p(t^{(i)} = 1 , x^{(i)})}{p(y^{(i)} = 1 , x^{(i)})} \\
&= \frac{p(y^{(i)} = 1 | t^{(i)} = 1,x^{(i)}) p(t^{(i)} = 1 , x^{(i)})}{p(y^{(i)} = 1 | t^{(i)} = 1,x^{(i)}) p(t^{(i)} = 1 , x^{(i)}) + p(y^{(i)} = 1 | t^{(i)} = 0,x^{(i)}) p(t^{(i)} = 0 , x^{(i)})}\\
&= \frac{\alpha p(t^{(i)} = 1 , x^{(i)})}{\alpha p(t^{(i)} = 1 , x^{(i)}) + 0}\\
&= 1
\end{align*}
$$

## (b)
$$
\begin{align*}
p(y^{(i)} =1\ \vert \ x^{(i)})
& = p(y^{(i)} = 1, t^{(i)} = 1 \ \vert \ x^{(i)}) + p(y^{(i)} = 1, t^{(i)} = 0 \ \vert \ x^{(i)}) \\
& = p(y^{(i)} = 1 \ \vert \ t^{(i)} = 1, x^{(i)}) \ p(t^{(i)} = 1 \ \vert \ x^{(i)}) + p(y^{(i)} = 1 \ \vert \ t^{(i)} = 0, x^{(i)}) \ p(t^{(i)} = 0 \ \vert \ x^{(i)}) \\
& = p(y^{(i)} = 1 \ \vert \ t^{(i)} = 1, x^{(i)}) \ p(t^{(i)} = 1 \ \vert \ x^{(i)}) \\
& = \alpha \ p(t^{(i)} = 1 \ \vert \ x^{(i)})
\end{align*}
$$
即
$$
p(t^{(i)} = 1 \ \vert \ x^{(i)}) = \frac{1}{\alpha} \ p(y^{(i)} = 1 \ \vert \ x^{(i)})
$$
## (c)
$$
\begin{align*}
h(x^{(i)}) &= p(y^{(i)} = 1 | x^{(i)})\\
&= \alpha p(t^{(i)} = 1 | x^{(i)})\\
&= \alpha 
\end{align*}
$$

则
$$
\begin{align*}
E[h(x^{(i)}) | y^{(i)} = 1]&=E[\alpha | y^{(i)} = 1]\\
&=\alpha
\end{align*}
$$