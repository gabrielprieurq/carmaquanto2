I checked both the attached PDF and the attached ZIP source. The model implemented there is not a single bare SDE from the start. It is a **daily, discrete-time solar-radiation model** built in layers, and then the package constructs a **continuous-time diffusion representation** in the `radiationModel` class. The package itself describes this as a stochastic model for solar radiation using seasonal models and Gaussian mixtures, with the main fitted object `solarModel` containing a clear-sky model, a transformed-variable seasonal model, an ARMA model, a seasonal variance model, a GARCH model, and a Gaussian-mixture model.  

I will write the model in mathematical notation and keep the notation aligned with the code.

## 1. Observed process and deterministic clear-sky envelope

Let (R_t) denote the observed daily global horizontal irradiance/radiation, called `GHI` in the code. The model first constructs a deterministic seasonal upper envelope (C_t), the fitted clear-sky radiation. The class `seasonalClearsky` is used for this. Its regressors include extraterrestrial radiation (H_0(t)), optional powers of (H_0(t)), optional trend, and Fourier seasonal terms. 

The solar-geometry layer uses the day-of-year (n\in{1,\dots,365}) and, by default, the **Spencer** formulas. The package defines
$$
B(n)=\frac{2\pi}{365},n,
$$
and the corrected solar constant
$$
G_{on}(n)=G_{sc}\Bigl(1.000110+0.034221\cos B+0.001280\sin B+0.000719\cos(2B)+0.000077\sin(2B)\Bigr),
$$
with (G_{sc}=1367). It also defines the solar declination
$$
\delta(n)=\frac{180}{\pi}\Bigl(0.006918-0.399912\cos B+0.070257\sin B-0.006758\cos(2B)+0.000907\sin(2B)-0.002697\cos(3B)+0.00148\sin(3B)\Bigr),
$$
and the sunset hour angle
$$
\omega_s(n,\phi)=\cos^{-1}!\bigl(-\tan(\delta(n))\tan(\phi)\bigr),
$$
where (\phi) is latitude in degrees. The daily extraterrestrial radiation on a horizontal surface is then
$$
H_0(n,\phi)=G_{on}(n)\frac{24\cdot 3600}{\pi}
\left(\cos\phi,\cos\delta(n),\sin\omega_s(n,\phi)+\frac{\pi}{180}\sin\phi,\sin\delta(n)\right),
$$
exactly as documented in `seasonalSolarFunctions`.   

The fitted clear-sky regression in the ZIP source is
$$
\widehat C_t^{(0)}
==================

\delta_0
+\sum_{j=1}^{J_H}\delta^{(H)}*j,H_0(t)^j
+\mathbf{1}*{{\text{trend}}}\delta_t,\tau_t
+\sum_{k=1}^{m}\Bigl(
\delta^{(s)}_k\sin\frac{2\pi k n_t}{P}
+
\delta^{(c)}_k\cos\frac{2\pi k n_t}{P}
\Bigr),
$$
where (J_H=\texttt{order_H0}), (m=\texttt{order}), (P=\texttt{period}) and (\tau_t=\text{Year}(t)-\max\text{Year}) in the training sample. In the code, the base formula is `clearsky ~ H0`, then powers (H0^2,\dots,H0^{J_H}) are added, then optional trend, then Fourier terms. The default controls are (J_H=1), (m=1), (P=365), with intercept included and no trend.  

After that first regression, the package rescales the whole fitted curve by a multiplicative factor. The code searches over a grid of (\delta) values and picks the **smallest** one such that the number of violations of the inequality
$$
R_t \le \delta,\widehat C_t^{(0)}
$$
is at most `ntol`. The final clear-sky curve is therefore
$$
C_t=\Delta,\widehat C_t^{(0)},
$$
with (\Delta=\delta,\delta_0^{\text{init}}), where `delta0` is the initial inflation factor used before the grid search. The defaults are `delta0 = 1.4`, `lower = 0`, `upper = 3`, `by = 0.001`, `ntol = 0`. The source code is explicit that this step enforces the clear-sky envelope.  

## 2. Risk-driver transform

The model does not work directly on (R_t). It first defines the **solar risk driver**
$$
X_t = 1-\frac{R_t}{C_t}.
$$
This is exactly `solarTransform$X(Rt,Ct)`. The inverse is
$$
R_t = C_t(1-X_t).
$$
Then (X_t) is assumed to lie in a bounded interval
$$
X_t\in(\alpha,\alpha+\beta),\qquad \alpha\ge 0,\quad \beta>0,\quad \alpha+\beta\le 1.
$$
The normalized variable is
$$
X_t'=\frac{X_t-\alpha}{\beta}\in(0,1).
$$
All of these mappings are given explicitly in the PDF documentation for `boundTransform` and `solarTransform`.  

The transformed latent state is then
$$
Y_t = g(X_t').
$$
The package allows four link functions (g): inverse Gumbel, Gumbel, logistic, and Gaussian quantile. The **default** in the code and spec is `invgumbel`, for which
$$
g(x)=\log(-\log x),\qquad
g^{-1}(y)=e^{-e^y},\qquad
g'(x)=\frac{1}{x\log x}.
$$
Therefore under the default link
$$
X_t' = e^{-e^{Y_t}},
$$
so
$$
X_t=\alpha+\beta e^{-e^{Y_t}},
$$
and the clearness index (K_t:=R_t/C_t) becomes
$$
K_t = 1-X_t
= 1-\alpha-\beta e^{-e^{Y_t}}.
$$
Hence the observed radiation is
$$
R_t = C_t\Bigl(1-\alpha-\beta e^{-e^{Y_t}}\Bigr).
$$
These are exactly the transformations documented in `dsolarX`, `dsolarK`, `dsolarGHI`, and `solarTransform$iRY`.  

The parameters ((\alpha,\beta)) are fitted from the empirical (X_t) sample in the source code as follows. Let (X_{(1)}) be the selected lower order statistic and (X^{(1)}) the selected upper order statistic, controlled by `min_pos` and `max_pos`. Then
$$
\varepsilon = X_{(1)}\times \texttt{threshold},
\qquad
\alpha = X_{(1)}-\varepsilon,
\qquad
\beta = \bigl(X^{(1)}-X_{(1)}\bigr)+2\varepsilon.
$$
After estimating these bounds, the code shrinks extreme (X_t) values slightly inward by factors (1+\delta) and (1-\delta) before computing (Y_t), to avoid numerical problems at the boundaries of the transform. The default transform settings are `link = "invgumbel"`, `threshold = 0.01`, `delta = 0.05`, `min_pos = 1`, `max_pos = 1`. 

## 3. Seasonal mean of the transformed process

Once (Y_t) is built, the package fits a `seasonalModel` to capture deterministic seasonality in (Y_t). The package describes `seasonalModel` as a linear combination of sine and cosine terms, optionally with intercept and trend. In the source, the Fourier basis added is exactly
$$
\sin!\Bigl(\frac{2\pi k n_t}{P}\Bigr),\qquad
\cos!\Bigl(\frac{2\pi k n_t}{P}\Bigr),\qquad k=1,\dots,m.
$$
So the seasonal mean is
$$
\bar Y_t
========

a_0
+\mathbf 1_{{\text{trend}}}a_t,\tau_t
+\sum_{k=1}^{m}\left(
a^{(s)}_k\sin\frac{2\pi k n_t}{P}
+
a^{(c)}_k\cos\frac{2\pi k n_t}{P}
\right).
$$
The deseasonalized series is
$$
\widetilde Y_t = Y_t-\bar Y_t.
$$
This is precisely what `fit_seasonal_model_Yt()` does. The default is (m=1), (P=365), intercept included, no trend.  

There is an optional **monthly mean correction**. If enabled, the code computes
$$
m_M = \mathbb E$$\widetilde Y_t\mid \text{Month}(t)=M$$,\qquad M=1,\dots,12,
$$
estimated empirically on the training sample month by month, and then replaces
$$
\widetilde Y_t \leftarrow \widetilde Y_t - m_{\text{Month}(t)}.
$$
So with monthly centering on, the exact decomposition is
$$
Y_t = \bar Y_t + m_{\text{Month}(t)} + Y_t^{c},
$$
where (Y_t^{c}) is the centered stochastic part passed to ARMA. This monthly centering is off by default. 

## 4. ARMA dynamics for the centered transformed process

The next layer is an ARMA model on the centered deseasonalized series. The PDF says `solarModel` fits `ARMA` on `Yt_tilde` and computes residuals `eps`, and the `ARMA_modelR6` class is the standard ARMA((p,q)) wrapper. 

Mathematically, the fitted centered process satisfies
$$
\Phi(L),Y_t^{c}
===============

\phi_0 + \Theta(L),\varepsilon_t,
$$
where
$$
\Phi(L)=1-\phi_1L-\cdots-\phi_pL^p,\qquad
\Theta(L)=1+\theta_1L+\cdots+\theta_qL^q.
$$
Equivalently,
$$
Y_t^{c}
=======

\phi_0+\sum_{i=1}^{p}\phi_i Y_{t-i}^{c}
+\varepsilon_t
+\sum_{j=1}^{q}\theta_j\varepsilon_{t-j}.
$$
The code defaults are (p=1), (q=0), no intercept. The fitted ARMA residual is
$$
\varepsilon_t = Y_t^{c} - \widehat Y_t^{c}.
$$
The implementation also stores the ARMA companion matrix (A) and shock-loading vector (b) to compute multi-step conditional means and variances.  

## 5. Seasonal variance of the ARMA residuals

The residual variance is not assumed constant. The package fits a second `seasonalModel`, now to (\varepsilon_t^2). The code does
$$
\varepsilon_t^2 \approx
c_0
+\mathbf 1_{{\text{trend}}}c_t,\tau_t
+\sum_{k=1}^{m_\sigma}\left(
c^{(s)}*k\sin\frac{2\pi k n_t}{P*\sigma}
+
c^{(c)}*k\cos\frac{2\pi k n_t}{P*\sigma}
\right),
$$
and defines the seasonal standard deviation by
$$
\bar\sigma_t = \sqrt{\widehat{\mathbb E}$$\varepsilon_t^2\mid n_t$$}.
$$
Then it standardizes the ARMA residuals as
$$
\widetilde\varepsilon_t = \frac{\varepsilon_t}{\bar\sigma_t}.
$$
This is exactly the `fit_seasonal_variance()` step described in `solarModel`. 

There is again an optional monthly correction. If enabled, the code computes a month-specific unconditional standard deviation
$$
s_M = \operatorname{sd}(\widetilde\varepsilon_t\mid \text{Month}(t)=M),
$$
and then uses
$$
\widetilde\varepsilon_t
\leftarrow
\frac{\varepsilon_t}{\bar\sigma_t,s_{\text{Month}(t)}}.
$$
There is also an optional correction factor that rescales the seasonal variance coefficients so that the standardized residuals have empirical variance exactly one on the training sample. Both monthly correction and variance correction are off by default. 

## 6. GARCH layer

After removing deterministic seasonal variance, the package optionally fits a standard GARCH model to (\widetilde\varepsilon_t). The `solarModel` documentation states that `fit_GARCH()` fits a GARCH model on `eps_tilde` and computes the standardized residuals `u_tilde`. The `sGARCH` class is a standard GARCH((p,q)) wrapper.  

The variance recursion is
$$
\sigma_t^2
==========

\omega
+\sum_{i=1}^{p_\sigma}\alpha_i,\widetilde\varepsilon_{t-i}^2
+\sum_{j=1}^{q_\sigma}\beta_j,\sigma_{t-j}^2,
$$
and the GARCH-standardized innovation is
$$
u_t = \frac{\widetilde\varepsilon_t}{\sigma_t}.
$$
Hence the full residual decomposition is
$$
\varepsilon_t
=============

\bar\sigma_t,s_{\text{Month}(t)},\sigma_t,u_t.
$$
The default is GARCH((1,1)), active by default. If GARCH is switched off, the code simply sets (\sigma_t\equiv 1) and (u_t=\widetilde\varepsilon_t). The unconditional GARCH variance used by the class is
$$
\sigma_\infty^2=\frac{\omega}{1-\sum_i\alpha_i-\sum_j\beta_j}.
$$


## 7. Monthly Gaussian-mixture innovations

The final innovation law is **not Gaussian**. The package fits a 2-component Gaussian mixture month by month to the GARCH-standardized residuals (u_t). The `solarModel` and `radiationModel` documentation refer to this as the `NM_model` / `solarMixture` layer and to a Bernoulli-type component classification.  

So, conditionally on the month (M=\text{Month}(t)),
$$
u_t \mid M
\sim
p_M,\mathcal N(\mu_{1,M},,s_{1,M}^2)
+
(1-p_M),\mathcal N(\mu_{2,M},,s_{2,M}^2).
$$
Equivalently, introducing a latent component indicator (B_t\in{1,2}),
$$
\Pr(B_t=1\mid M)=p_M,\qquad \Pr(B_t=2\mid M)=1-p_M,
$$
and
$$
u_t\mid(B_t=b,M)\sim \mathcal N(\mu_{b,M},s_{b,M}^2).
$$

Therefore the **full discrete-time model** can be written compactly as
$$
R_t = C_t\Bigl(1-\alpha-\beta e^{-e^{Y_t}}\Bigr),
$$
$$
Y_t = \bar Y_t + m_{\text{Month}(t)} + Y_t^c,
$$
$$
Y_t^c
=====

\phi_0+\sum_{i=1}^{p}\phi_i Y_{t-i}^c
+\varepsilon_t
+\sum_{j=1}^{q}\theta_j\varepsilon_{t-j},
$$
$$
\varepsilon_t
=============

\bar\sigma_t,s_{\text{Month}(t)},\sigma_t,u_t,
$$
$$
\sigma_t^2
==========

\omega+\sum_i\alpha_i\widetilde\varepsilon_{t-i}^2+\sum_j\beta_j\sigma_{t-j}^2,
\qquad
\widetilde\varepsilon_t=\frac{\varepsilon_t}{\bar\sigma_t,s_{\text{Month}(t)}}=\sigma_t u_t,
$$
$$
u_t\mid \text{Month}(t)=M
\sim
p_M\mathcal N(\mu_{1,M},s_{1,M}^2)
+(1-p_M)\mathcal N(\mu_{2,M},s_{2,M}^2).
$$
If monthly mean correction is off, take (m_M\equiv 0). If monthly variance correction is off, take (s_M\equiv 1).  

## 8. Continuous-time diffusion representation in `radiationModel`

This is the SDE part. The PDF states that the `radiationModel` class has a **mean-reversion parameter** `theta`, seasonal mean (\bar Y_t), seasonal volatility (\bar\sigma_t), monthly mixture drifts (\mu_B), monthly mixture diffusions (\sigma_B), and componentwise drifts/diffusions (\mu_Y,\sigma_Y,\mu_R,\sigma_R).  

There is an important notation clash in the code: the name `theta` is used both for the **continuous-time mean-reversion speed** in `radiationModel` and elsewhere for a **mixture shift parameter** in forecasting/pricing routines. To avoid confusion, I will denote the mean-reversion speed by (\kappa), even though the code stores it as `theta`. The PDF confirms that `radiationModel$theta` is the mean-reversion parameter, while `solarModel$Moments(..., theta = 0)` uses `theta` as a shift parameter.  

The ZIP source estimates (\kappa) from the transformed series by a martingale-type method and then replaces the discrete AR coefficient by
$$
\phi_1 = e^{-\kappa}.
$$
So the continuous-time reinterpretation is an OU-type mean-reverting dynamics around the seasonal mean.

### 8.1 SDE for the transformed process (Y_t)

Fix one mixture component (B\in{1,2}). The code-defined drift and diffusion are, in continuous notation,
$$
dY_t
====

\mu_Y(Y_t,t,B),dt+\sigma_Y(t,B),dW_t,
$$
with
$$
\mu_Y(Y_t,t,B)
==============

\dot{\bar Y}(t)
-\kappa\bigl(Y_t-\bar Y(t)\bigr)
+\bar\sigma(t),\mu_B(t)
+\bar\sigma(t),\sigma_B(t),\lambda,\mathbf 1_{{Q}},
$$
and
$$
\sigma_Y(t,B)=\bar\sigma(t),\sigma_B(t).
$$

Here:

* (\bar Y(t)) is the seasonal mean of (Y_t);
* (\bar\sigma(t)) is the deterministic seasonal standard deviation;
* (\mu_B(t)) and (\sigma_B(t)) are the month-specific mean and standard deviation of mixture component (B);
* (\lambda) is the market-price-of-risk term used when the package switches from (P) to (Q).

This is exactly what the ZIP source implements in `mu_Y()` and `sigma_Y()`: finite differences for (\dot{\bar Y}(t)), linear mean reversion, plus the mixture drift and optional (Q)-measure correction. The PDF documents the existence of these methods and their meaning.   

Equivalently, for a fixed component (B), the mild solution is
$$
Y_T
===

e^{-\kappa (T-t)}Y_t
+
\int_t^T e^{-\kappa (T-s)}
\Bigl(
\dot{\bar Y}(s)+\kappa \bar Y(s)+\bar\sigma(s)\mu_B(s)+\bar\sigma(s)\sigma_B(s)\lambda\mathbf 1_{{Q}}
\Bigr),ds
+
\int_t^T e^{-\kappa (T-s)}\bar\sigma(s)\sigma_B(s),dW_s.
$$
Hence
$$
Y_T\mid (Y_t,B)
\sim
\mathcal N!\bigl(M_Y^{(B)}(t,T),,S_Y^{(B)}(t,T)\bigr),
$$
with
$$
M_Y^{(B)}(t,T)
==============

e^{-\kappa (T-t)}Y_t
+
\int_t^T e^{-\kappa (T-s)}
\Bigl(
\dot{\bar Y}(s)+\kappa \bar Y(s)+\bar\sigma(s)\mu_B(s)+\bar\sigma(s)\sigma_B(s)\lambda\mathbf 1_{{Q}}
\Bigr),ds,
$$
$$
S_Y^{(B)}(t,T)
==============

\int_t^T e^{-2\kappa (T-s)}\bar\sigma(s)^2\sigma_B(s)^2,ds.
$$
The package methods `integral_expectation()`, `integral_variance()`, `M_Y()`, `S_Y()`, `pdf_Y()`, and `cdf_Y()` are exactly the numerical/closed-form implementation of these quantities.   

### 8.2 Continuous-time seasonal variance used in that SDE

In the continuous-time wrapper, the code reparametrizes the first-harmonic seasonal variance from the discrete coefficients
$$
a_0+a_1\sin(\omega t)+a_2\cos(\omega t),\qquad \omega=\frac{2\pi}{365},
$$
into continuous-time coefficients (c_0,c_1,c_2). This reparametrization is explicit in the ZIP source and gives
$$
c_0 = \frac{2\kappa a_0}{1-e^{-2\kappa}},
$$
$$
\alpha_* = 1-e^{-2\kappa}\cos\omega,\qquad
\beta_* = e^{-2\kappa}\sin\omega,\qquad
D=\alpha_*^2+\beta_*^2,
$$
$$
c_1=
\frac{(2\kappa\alpha_*+\omega\beta_*)a_1+(2\kappa\beta_*-\omega\alpha_*)a_2}{D},
$$
$$
c_2=
\frac{(\omega\alpha_*-2\kappa\beta_*)a_1+(\omega\beta_*+2\kappa\alpha_*)a_2}{D}.
$$
Then the seasonal variance entering the diffusion is
$$
\bar\sigma(t)^2 = c_0+c_1\sin(\omega t)+c_2\cos(\omega t).
$$
The code also stores exact integral coefficients for
$$
\int_t^T e^{-2\kappa (T-s)}\bar\sigma(s)^2,ds,
$$
which is why the `radiationModel` can compute conditional variances analytically once the mixture component is fixed. This level of detail is in the ZIP source rather than the API PDF. The PDF only states that the class reparametrizes seasonal variance and computes the corresponding integrals.  

## 9. SDE for the observed radiation (R_t)

Now apply Itô to
$$
R_t = f(t,Y_t)=C_t\Bigl(1-\alpha-\beta e^{-e^{Y_t}}\Bigr).
$$
Define
$$
K_t = 1-\alpha-\beta e^{-e^{Y_t}} = \frac{R_t}{C_t}.
$$
Then
$$
\frac{\partial f}{\partial y}(t,y)
==================================

C_t,\beta,e^{y-e^y},
$$
and
$$
\frac{\partial^2 f}{\partial y^2}(t,y)
======================================

C_t,\beta,e^{y-e^y}(1-e^y).
$$
Therefore the code-implied SDE for (R_t), conditional on component (B), is
$$
dR_t = \mu_R(R_t,t,B),dt+\sigma_R(R_t,t,B),dW_t,
$$
with
$$
\mu_R(R_t,t,B)
==============

\dot C_t,K_t
+
C_t\beta e^{Y_t-e^{Y_t}}
\left(
\mu_Y(Y_t,t,B)+\frac12(1-e^{Y_t})\sigma_Y(t,B)^2
\right),
$$
and
$$
\sigma_R(R_t,t,B)=
C_t\beta e^{Y_t-e^{Y_t}}\sigma_Y(t,B).
$$
This is exactly the formula coded in `mu_R()` and `sigma_R()`. The PDF documents these methods, while the exact algebra is in the attached ZIP source.  

So the package’s continuous-time model is a **componentwise mean-reverting diffusion for (Y_t)**, transformed nonlinearly through the inverse-Gumbel map and the clear-sky envelope to obtain (R_t).

## 10. Conditional densities and moments

Because (Y_T\mid(Y_t,B)) is Gaussian in the continuous-time wrapper, and (B) is mixed with monthly probability (p_{M(T)}), the terminal law of (Y_T) is
$$
f_{Y_T\mid Y_t}(y)
==================

p_{M(T)},\varphi(y;M_Y^{(1)},S_Y^{(1)})
+
(1-p_{M(T)}),\varphi(y;M_Y^{(2)},S_Y^{(2)}),
$$
where (\varphi(\cdot;m,v)) is the Gaussian density with mean (m) and variance (v). The package implements this in `pdf_Y()` and `cdf_Y()`.  

Since
$$
R_T = C_T\Bigl(1-\alpha-\beta e^{-e^{Y_T}}\Bigr),
$$
the density of (R_T) is obtained by change of variables:
$$
f_{R_T\mid R_t}(r)
==================

f_{Y_T\mid Y_t}!\bigl(y(r)\bigr),
\left|\frac{dy}{dr}\right|,
$$
where
$$
y(r)=g!\left(\frac{1-\alpha-r/C_T}{\beta}\right)
================================================

\log!\left(
-\log\left(\frac{1-\alpha-r/C_T}{\beta}\right)
\right)
$$
under the default inverse-Gumbel link. Since
$$
\frac{dr}{dy}=C_T\beta e^{y-e^y},
$$
we get
$$
f_{R_T\mid R_t}(r)
==================

\frac{
f_{Y_T\mid Y_t}(y(r))
}{
C_T\beta e^{,y(r)-e^{y(r)}}
}.
$$
This is exactly the structure behind `dsolarGHI`, `pdf_R()`, `cdf_R()`, `e_GHI()`, and `v_GHI()`.  

The same transformation formulas give the package-level random-variable definitions
$$
X(Y)=\alpha+\beta e^{-e^Y},\qquad
K(Y)=1-\alpha-\beta e^{-e^Y},\qquad
R(Y)=C_t\bigl(1-\alpha-\beta e^{-e^Y}\bigr),
$$
which are exactly the formulas documented in `dsolarX`, `dsolarK`, and `dsolarGHI`. 

## 11. What the fitted model is, in one line

If I compress the entire attached implementation into one mathematical statement, it is this:

$$
\boxed{
R_t
===

C_t\Bigl(1-\alpha-\beta e^{-e^{Y_t}}\Bigr)
}
$$

with

$$
\boxed{
Y_t
===

\bar Y_t + m_{\text{Month}(t)} + Y_t^c
}
$$

and

$$
\boxed{
\Phi(L)Y_t^c = \phi_0 + \Theta(L)\varepsilon_t,
\qquad
\varepsilon_t = \bar\sigma_t,s_{\text{Month}(t)},\sigma_t,u_t,
\qquad
u_t\mid \text{Month}(t)=M \sim p_M\mathcal N(\mu_{1,M},s_{1,M}^2)+(1-p_M)\mathcal N(\mu_{2,M},s_{2,M}^2)
}
$$

plus the optional continuous-time reinterpretation

$$
\boxed{
dY_t
====

\Bigl$$\dot{\bar Y}(t)-\kappa(Y_t-\bar Y(t))+\bar\sigma(t)\mu_B(t)+\bar\sigma(t)\sigma_B(t)\lambda\mathbf 1_{{Q}}\Bigr$$dt
+\bar\sigma(t)\sigma_B(t)dW_t
}
$$

and the induced (R_t)-diffusion obtained by Itô. This is the precise mathematical content of the attached package implementation.  

One subtle point is worth emphasizing. The **core `solarModel` is fundamentally discrete-time**: seasonal regression (+) ARMA (+) seasonal variance (+) GARCH (+) monthly Gaussian mixture. The `radiationModel` then builds the continuous-time mean-reverting diffusion representation on top of that fitted discrete object. So the SDE is not the starting point of estimation; it is the continuous-time representation of the fitted transformed model.

If you want, I can next turn this into a **paper-style model section in LaTeX**, with proposition-style notation and all equations numbered.
