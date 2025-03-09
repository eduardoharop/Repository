# === Constants ===================================================================================
npoints = 30
ntrials = 10
x0 = 20
dy = 1000
x = 1:npoints + rnorm(npoints, 0, 0.5)
set.seed(42)

y = x^3 + x + 500 + rnorm(npoints, 0, 2500)
plot(x, y, pch = 16, col="darkorange", ylim = c(-5000,30000))
curve(x^3 + x + 500, lwd = 3, col = "grey", add = TRUE)
points(x0, x0^3 + x0 - dy, col = "blue", pch = 16, cex = 1.2)

# === Linear regression ===========================================================================
Sys.sleep(3)
for(i in 1:ntrials) {
  y = x^3 + x + 500 + rnorm(npoints, 0, 2500)
  plot(x, y, pch = 16, col="darkorange", ylim = c(-5000,30000))
  curve(x^3 + x + 500, lwd = 3, col = "grey", add = TRUE)
  lm = lm(y ~ x)
  curve(lm$coefficients[1] + lm$coefficients[2] * x, lwd = 3, col = "black", add = TRUE)
  points(x0, x0^3 + x0 - dy, col = "blue", pch = 16, cex = 1.2)
  points(x0, predict(lm, data.frame('x' = x0)), pch = 16, cex = 1.2)
  segments(x0, x0^3 + x0 - dy, x1 = x0, y1 = predict(lm, data.frame('x' = x0)), col = "blue", lwd = 2)
  Sys.sleep(2)
}

# === Polynomial ==================================================================================
Sys.sleep(3)
df = data.frame('x' = x)
for(i in 1:ntrials) {
  y = x^3 + x + 500 + rnorm(npoints, 0, 3000)
  df$y = y
  plot(x, y, pch = 16, col="darkorange", ylim = c(-5000,30000))
  curve(x^3 + x + 500, lwd = 3, col = "grey", add = TRUE)
  pol = lm(y ~ poly(x, 20), data = df)
  xx = seq(0.5,npoints,0.01)
  yy = predict(pol, data.frame('x' = xx))
  points(xx, yy, type = 'l', lwd = 3)
  points(x0, x0^3 + x0 - dy, col = "blue", pch = 16, cex = 1.2)
  points(x0, predict(pol, data.frame('x' = x0)), pch = 16, cex = 1.2)
  segments(x0, x0^3 + x0 - dy, x1 = x0, y1 = predict(pol, data.frame('x' = x0)), col = "blue", lwd = 2)
  Sys.sleep(2)
}

# === Increasing no. of observations (x10) decreases overfitting ===================================
Sys.sleep(3)
npoints2 = npoints * 10
ntrials = 10
x = (1:npoints2) / 10 + rnorm(npoints2, 0, 0.5)
df = data.frame('x' = x)
Sys.sleep(2)
for(i in 1:ntrials) {
  y = x^3 + x + 500 + rnorm(npoints, 0, 3000)
  df$y = y
  plot(x, y, pch = 16, col="darkorange", ylim = c(-5000,30000))
  curve(x^3 + x + 500, lwd = 3, col = "grey", add = TRUE)
  pol = lm(y ~ poly(x, 20), data = df)
  xx = seq(0.5,npoints,0.01)
  yy = predict(pol, data.frame('x' = xx))
  points(xx, yy, type = 'l', lwd = 3)
  points(x0, x0^3 + x0 - dy, col = "blue", pch = 16, cex = 1.2)
  points(x0, predict(pol, data.frame('x' = x0)), pch = 16, cex = 1.2)
  segments(x0, x0^3 + x0 - dy, x1 = x0, y1 = predict(pol, data.frame('x' = x0)), col = "blue", lwd = 2)
  Sys.sleep(2)
}


