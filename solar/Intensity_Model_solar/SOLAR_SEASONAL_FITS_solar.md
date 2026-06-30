# Solar Seasonal Fit Notes

## Why solar needs a different seasonal treatment

Wind can be modeled reasonably well by applying a logit transform directly to the capacity factor and then fitting calendar seasonality.

Solar cannot.

The reason is structural:

- night-time hours force exact zeros,
- sunrise and sunset move continuously through the year,
- the amplitude of the daytime profile is strongly annual,
- a raw logit on \(Q_t\) confounds deterministic daylight mechanics with stochastic weather.

For this reason, the solar branch uses a two-layer deterministic decomposition.

## Layer 1: clear-sky proxy

The fitted clear-sky proxy is a smooth calendar envelope \(\widehat C_t\) built from the upper tail of the observed hourly capacity factor.

Interpretation:

- it is not a physical irradiance model,
- it is a data-driven proxy for the maximum feasible aggregate German solar capacity factor at each calendar hour,
- it absorbs the daylight geometry before the stochastic layer is fitted.

The proxy should be inspected in the notebook against:

1. realized solar capacity factor,
2. the naive raw-logit fit,
3. the transformed latent seasonal mean.

## Layer 2: latent seasonal mean

After normalizing by the clear-sky proxy, the code models the transformed cloudiness/risk-driver series in latent coordinates.

The deterministic regression contains:

- hour dummies,
- month dummies,
- annual Fourier terms,
- interactions between hour dummies and annual Fourier terms.

Economically and statistically this means:

- the baseline solar shape differs by hour,
- the annual cycle is smooth rather than a step function,
- the shape of the intraday profile itself is allowed to evolve over the year.

This is the part most directly inspired by the `solarr` decomposition.

## Benchmark fit kept for comparison

The notebook also includes `mle_notebook_raw`, which applies the generic notebook idea directly to a raw logit of the solar capacity factor.

That benchmark is intentionally weaker.  
It is kept to document why the solar-specific decomposition is needed.

The most informative comparison metrics are:

- latent-space RMSE,
- physical-space RMSE after inverse mapping,
- maximum absolute difference between fitted latent curves,
- residual ACF/PACF after the AR(24) step,
- marginal fit of the simulated solar capacity factor.

## Recommended interpretation of the plots

When the `solar_clearsky` specification is behaving correctly, the notebook should show:

1. a clear-sky envelope that tracks the annual daylight structure smoothly;
2. latent residuals with much weaker deterministic day/night structure than the raw-logit benchmark;
3. lower physical-space fitting error than the naive benchmark;
4. more stable AR(24) residual diagnostics;
5. a better simulated marginal law for the solar capacity factor.

## Relation to the Wishart block

The Wishart model is **not** calibrated to the raw spot-solar dependence.
It is calibrated to the covariance of the joint VAR innovation pair after:

1. spot deseasonalization and spike removal,
2. solar clear-sky normalization,
3. solar latent seasonality removal,
4. joint VAR(24) filtering.

That is the correct dependence object for the stochastic-covariance extension.
