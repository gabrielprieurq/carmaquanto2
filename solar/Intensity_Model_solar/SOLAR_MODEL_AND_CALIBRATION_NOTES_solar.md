# Spot-Solar Intensity Model Notes

## Objective

This folder contains a standalone solar extension of the `Intensity_Model` workflow.  
Nothing in the original `Intensity_Model` folder is modified.

The model keeps the same structure as the wind notebook for:

1. spot spikes and spot continuous dynamics,
2. nonparametric jump-intensity calibration,
3. AR(24) calibration of the continuous factors,
4. full bivariate Wishart calibration on the joint VAR innovation covariance.

The solar side is changed in one place only: the renewable state is no longer a raw logit of the capacity factor.  
Instead, it follows a solar-specific deterministic decomposition inspired by the `solarr` package.

## Model

### Spot block

The spot block is unchanged:

\[
S_t = \Lambda_t^{P} + X_t^{P} + Y_t,
\]

where:

- \(\Lambda_t^{P}\) is deterministic spot seasonality,
- \(X_t^{P}\) is the continuous spot factor,
- \(Y_t\) is the spike factor.

Spikes are detected exactly as in the original notebook, year by year, using the paper threshold on first differences.

The spike state is reconstructed with exponential decay:

\[
Y_{t+\Delta} = e^{-\beta \Delta} Y_t + J_{t+\Delta}.
\]

### Solar deterministic decomposition

Let \(Q_t \in [0,1]\) be the observed solar capacity factor.

The solar package suggests separating:

1. a deterministic clear-sky-type envelope,
2. a bounded cloudiness/risk-driver transform,
3. a deterministic seasonal mean in transformed coordinates,
4. a short-memory residual process.

This is implemented as follows.

#### 1. Clear-sky proxy

For each calendar pair \((d,h)\), where \(d\) is day-of-year and \(h\) is hour-of-day, define

\[
\widehat C_{d,h}^{\mathrm{raw}}
=
\mathrm{Quantile}_{0.98}\!\left(Q_t \mid \mathrm{doy}(t)=d,\ \mathrm{hour}(t)=h\right).
\]

The resulting \(366 \times 24\) table is then smoothed cyclically across day-of-year to obtain the clear-sky proxy

\[
\widehat C_{d,h}.
\]

For simulation, this gives a deterministic hourly envelope

\[
\widehat C_t = \widehat C_{\mathrm{doy}(t),\mathrm{hour}(t)}.
\]

This is the solar analogue of the `solarr` clear-sky component \(C_t\), but built directly from the German capacity-factor data.

#### 2. Solar risk driver and bounded transform

Following `solarr`, define a cloudiness-type risk driver

\[
X_t = 1 - \frac{Q_t}{\max(\widehat C_t,\varepsilon_C)} \in [0,1].
\]

The empirical support is expanded slightly by fitted bounds \(\alpha\) and \(\beta\):

\[
X_t' = \frac{X_t - \alpha}{\beta} \in (0,1).
\]

The latent solar state is then the logit transform

\[
Y_t^{Q} = \log\!\left(\frac{X_t'}{1-X_t'}\right).
\]

This mirrors the `solarr` mapping

\[
R_t = C_t \bigl(1-\alpha-\beta g^{-1}(Y_t)\bigr),
\]

with \(g^{-1}\) chosen here as the logistic inverse.

#### 3. Deterministic solar seasonality in latent coordinates

The deterministic latent mean is estimated by OLS:

\[
\Lambda_t^{Q} = z_t^\top \gamma,
\]

where \(z_t\) contains:

- hour-of-day dummies,
- month dummies,
- annual Fourier terms,
- hour-by-annual-Fourier interactions.

This is the key solar-specific improvement over the wind notebook.  
It allows sunrise, sunset, and intraday profile changes to move smoothly through the year.

#### 4. Solar continuous component

Define the deseasonalized latent solar factor

\[
X_t^{Q} = Y_t^{Q} - \Lambda_t^{Q}.
\]

This is fitted with the same AR(24) machinery used in the wind notebook:

\[
X_t^{Q} = \sum_{j=1}^{24}\phi_j^{Q} X_{t-j}^{Q} + \varepsilon_t^{Q}.
\]

### Inverse solar map

When simulating, the solar latent state is mapped back to physical capacity factor via

\[
Q_t
=
\widehat C_t\left(1-\alpha-\beta \sigma\!\left(\Lambda_t^{Q}+X_t^{Q}\right)\right),
\]

where \(\sigma(x) = (1+e^{-x})^{-1}\).

This guarantees:

- \(Q_t \in [0,1]\),
- night-time output is forced to zero when \(\widehat C_t = 0\),
- seasonal daylight variation is handled deterministically rather than by the AR residual.

## Spike-intensity calibration

The jump-intensity calibration follows the wind notebook exactly in methodology.

Let \(I_t^{+}\) and \(I_t^{-}\) denote positive and negative spot-jump indicators.
Using lagged solar capacity factor \(Q_{t-1}\) as conditioning variable, the kernel estimator is

\[
\widehat\lambda^\pm(q)
=
\frac{\sum_{t} K_h(q-Q_{t-1}) I_t^\pm}
{\Delta \sum_t K_h(q-Q_{t-1})}.
\]

Implementation details:

- Epanechnikov kernel,
- data-driven solar support grid,
- fixed bandwidth selected from the same style of paper criterion,
- positive intensity treated as constant in simulation,
- negative intensity projected to a two-state increasing function:

\[
\lambda^{-}(q)=
\begin{cases}
\lambda^{-}_{\mathrm{low}}, & q \le q^\star,\\
\lambda^{-}_{\mathrm{high}}, & q > q^\star.
\end{cases}
\]

The monotonicity restriction is economically natural for solar cannibalization:
high solar output should be associated with more negative-price pressure.

## Joint spot-solar covariance model

The stochastic-covariance block is unchanged in structure from the corrected intensity notebook.

The continuous pair is

\[
U_t = \begin{pmatrix} X_t^{P} \\ X_t^{Q} \end{pmatrix}.
\]

Rather than fitting two separate AR filters, the code estimates the true joint VAR(24):

\[
U_t = \sum_{\ell=1}^{24}\Phi_\ell U_{t-\ell} + \eta_t,
\qquad
\eta_t \sim (0,\Sigma_t).
\]

The Wishart block is calibrated to a rolling covariance proxy of the joint innovation pair \(\eta_t\), exactly as in the corrected notebook:

\[
\widehat\Sigma_t(W)
=
\frac{1}{W-1}\sum_{j=0}^{W-1} \eta_{t-j}\eta_{t-j}^\top .
\]

Candidate windows are compared, and the selected fit is the one minimizing the same correlation-aware score used in the base notebook.

## Model variants in this folder

Two solar variants are kept in parallel:

1. `solar_clearsky`
   This is the main model and should be treated as the solar analogue of the corrected renewable specification.

2. `mle_notebook_raw`
   This is a naive benchmark using a raw logit transform of the solar capacity factor and the generic notebook calendar regression.

The comparison is useful because it shows how much of the solar structure is missed if one ignores the clear-sky-style envelope.
