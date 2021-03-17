# General Linear Model

### Exponential Family

广义线性模型的核心是Exponential Family，它包含了机器学习中最重要的分布形式（伯努利分布，高斯分布，均匀分布，gamma分布，t分布）。Exponential Family是指一种分布的概率密度函数满足以下条件，
$$
P(x|\eta)=h(x)\exp[\eta^TT(x)-A(\eta)]
$$
在广义线性模型中，假设

- 变量y的分布满足ExpFamily，即$P(y|\eta)=h(y)\exp[\eta^TT(y)-A(\eta)]$，在二分类问题中y满足伯努利分布
- 对于给定的输入x，目标是输出$h(x)=E(T(y)|x)$。
- 令$\eta=\Theta^TX$，代入目标函数

### 对数几率回归

基于ExpFamily，广义线性模型可以处理二分类问题，假设分类的输出$y=\{0,1\},\phi(y=1)=\phi,\phi(y=0)=1-\phi$。显然输出y满足伯努利分布，
$$
P(y)=\phi^y(1-\phi)^{1-y}=\exp\left[y\ln\phi+(1-y)\ln(1-\phi)\right]\\
=\exp[y\ln\frac{\phi}{1-\phi}+\ln(1-\phi)]
$$
对比ExpFamily形式，$\eta=\ln\frac{\phi}{1-\phi},\phi(y)=y,a(y)=\ln(1-\phi)$，可以得到sigmoid函数，
$$
\phi=\frac{1}{1+e^{-\eta}}=\frac{1}{1+e^{-\Theta^TX}}
$$
我们的目标函数是
$$
h(y|x)=E(y|x)=\prod_{i=1}^my^iP(y^i)
$$
得到他的对数似然函数，并用极大似然法求解系数，
$$
\ell(\Theta)=\sum_{i=1}^m[y^i\Theta^TX^i-\ln(1+e^{\Theta^TX})]\\
\Theta^*=arg_{\Theta}\max\ell(\Theta)
$$
对对数似然函数求梯度
$$
\frac{\partial \ell(\Theta)}{\partial\Theta}=\sum_{k}\frac{\partial \ell(\Theta)}{\partial\Theta_k}=\sum_{i=1}^m[-y^iX^i+\frac{X^ie^{\Theta^TX^i}}{1+e^{\Theta^TX^i}}]
$$

### Softmax(多分类问题)

Softmax是用于求解多分类问题（类间无交集）的模型，输出y满足多项分布$y^i=\{1,2,...,k\};p(y^i=t)=\phi_t$最后一项满足$p(y^i=k)=1-\sum_{t=1}^{k-1}\phi_t$

我们定义ExpFamily中的T(y)为这样的形式，
$$
T(1)=\left(\begin{array}{c1}1\\0\\0\\\vdots\end{array}\right)\quad T(k-1)=\left(\begin{array}{c1}0\\0\\\vdots\\1\end{array}\right)\quad T(k)=\left(\begin{array}{c1}0\\0\\0\\\vdots\end{array}\right)
$$
即等价于对这样的结果分类，各项的元素可以表示成$T(y^i)_t=\mathbb{I}(y^i=t),t=\{1,\cdots,k-1\}$。显然y满足多项分布，
$$
P(y)=\prod_{i=1}^k\phi_i^{\mathbb{I}(y^i=i)}=\exp[\eta^TT(y)-A(\eta)]
$$
其中
$$
\eta=\left(\begin{array}{c1}\ln(\frac{\phi_1}{\phi_k})\\\ln(\frac{\phi_2}{\phi_k})\\\vdots\\\ln(\frac{\phi_{k-1}}{\phi_k})\end{array}\right)\quad A(\eta)=-\ln(\phi_k)
$$
得到各分类的概率函数
$$
\phi_i=\frac{e^{\eta_i}}{1+\sum_{j=1}^{k-1}e^{\eta_j}},i=\{1,2,...,k-1\}\quad \phi_k=\frac{1}{1+\sum_{j=1}^{k-1}e^{\eta_j}}
$$
令$\eta_k=0$,所以,这里$X^i=[x^i_1,x^i_2,\cdots,x^i_d,1]$，
$$
\phi_t=\frac{e^{\eta_t}}{\sum_{j=1}^{k}e^{\eta_j}}=\frac{e^{\Theta_t^TX^i}}{\sum_{j=1}^{k}e^{\Theta_j^TX^i}},i=\{1,2,...,k\}
$$
同理目标函数是
$$
h(y|x)=E(y|x)=\prod_{i=1}^mP^{\mathbb{I}(y^i=k)}(y^i=k)
$$
对数似然函数，并用极大似然法求解系数
$$
\ell(\Theta)=\sum_{i=1}^m\mathbb{I}(y^i=k)\ln\phi_k=\sum_{i=1}^m\mathbb{I}(y^i=k)\ln\frac{e^{\Theta_k^TX^i}}{\sum_{j=1}^{k}e^{\Theta_j^TX^i}}
$$
求梯度
$$
\frac{\partial \ell(\Theta)}{\partial\Theta_t}=\sum_{l}\frac{\partial \ell(\Theta)}{\partial\Theta_{tl}}=\sum_{l}\sum_{i=1}^m\mathbb{I}(y^i=k)\frac{1}{\phi_k}\frac{\mathbb{I}(k=t)X_le^{\Theta_t^TX^i}\sum_{j=1}^{k}e^{\Theta_j^TX^i}-e^{\Theta_k^TX^i}e^{\Theta_t^TX^i}X^i_l}{(\sum_{j=1}^{k}e^{\Theta_j^TX^i})^2}\\
=\sum_{i=1}^m\left[\mathbb{I}(y^i=k)\mathbb{I}(k=t)X^i-\mathbb{I}(y^i=k)\phi_tX^i\right]
$$

### 其他方法

对于类间有交集的模型，可以使用One VS One或One VS Rest，他们都可以直接从二分类法拓展出来。