# Load necessary libraries
library(tidyverse)
library(caret)
library(neuralnet)

# Load data
data <- read.csv("sensor_data.csv")

# Characterize each sensor
sensor_summary <- data %>%
        group_by(sensor) %>%
        summarize(mean = mean(pm2.5),
                  sd = sd(pm2.5),
                  min = min(pm2.5),
                  max = max(pm2.5))

# Plot time series of low-cost sensors against high-cost sensor
ggplot(data, aes(x = timestamp, y = pm2.5, color = sensor)) +
        geom_line() +
        scale_color_discrete(name = "Sensor")

# Measure degree of difference in behavior with 2 metrics
rmse <- function(y_true, y_pred) {
        sqrt(mean((y_true - y_pred)^2))
}

r_squared <- function(y_true, y_pred) {
        1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)
}

results <- data %>%
        group_by(sensor) %>%
        summarize(rmse = rmse(pm2.5, reference),
                  r_squared = r_squared(pm2.5, reference))

# Calibrate each sensor with the reference sensor
# Method 1: Mean offset correction
calibrated_mean_offset <- data %>%
        group_by(sensor) %>%
        mutate(calibrated = pm2.5 - (mean(pm2.5) - mean(reference)))

# Method 2: Linear regression
fit_linear <- lm(reference ~ pm2.5, data = data)
calibrated_linear <- data %>%
        mutate(calibrated = predict(fit_linear, newdata = data))

# Method 3: Polynomial regression
fit_poly <- lm(reference ~ poly(pm2.5, 2), data = data)
calibrated_poly <- data %>%
        mutate(calibrated = predict(fit_poly, newdata = data))

# Method 4: Generalized Additive Model (GAM)
library(mgcv)
fit_gam <- gam(reference ~ s(pm2.5), data = data)
calibrated_gam <- data %>%
        mutate(calibrated = predict(fit_gam, newdata = data))

# Method 5: Random Forest
set.seed(123)
fit_rf <- train(reference ~ pm2.5, data = data, method = "rf")
calibrated_rf <- data %>%
        mutate(calibrated = predict(fit_rf, newdata = data))

# Method 6: Support Vector Regression (SVR)
fit_svr <- train(reference ~ pm2.5, data = data, method = "svmRadial")
calibrated_svr <- data %>%
        mutate(calibrated = predict(fit_svr, newdata = data))


# Method 7: Artificial Neural Network (ANN)
fit_ann <- neuralnet(reference ~ pm2.5, data = data, hidden = c(5))
calibrated_ann <- data %>%
        mutate(calibrated = compute(fit_ann, data[, "pm2.5"])$net.result)

# Method 8: Convolutional Neural Network (CNN)
library(keras)
model_cnn <- keras_model_sequential() %>%
        layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu",
                      input_shape = c(length(data$pm2.5), 1)) %>%
        layer_max_pooling_1d(pool_size = 2) %>%
        layer_flatten() %>%
        layer_dense(units = 10, activation = "relu") %>%
        layer_dense(units = 1)

model_cnn %>% compile(loss = "mean_squared_error", optimizer = "adam")
model_cnn %>% fit(array(data$pm2.5), array(data$reference), epochs = 100)

calibrated_cnn <- data %>%
        mutate(calibrated = as.vector(predict(model_cnn, array(data$pm2.5, dim = c(nrow(data), 1, 1)))))

# Method 9: Recurrent Neural Network (RNN)
model_rnn <- keras_model_sequential() %>%
        layer_lstm(units = 10, input_shape = c(1, length(data$pm2.5))) %>%
        layer_dense(units = 1)

model_rnn %>% compile(loss = "mean_squared_error", optimizer = "adam")
model_rnn %>% fit(array(data$pm2.5, dim = c(nrow(data), 1, 1)), array(data$reference), epochs = 100)

calibrated_rnn <- data %>%
        mutate(calibrated = as.vector(predict(model_rnn, array(data$pm2.5, dim = c(nrow(data), 1, 1)))))

# Method 10: Deep Belief Network (DBN)
library(RBM)
fit_dbn <- rbm(reference ~ pm2.5, data = data, n_hidden = 10)
calibrated_dbn <- data %>%
        mutate(calibrated = predict(fit_dbn, newdata = data))

# Measure quality of calibration with 2 metrics
calibrated_data <- list(mean_offset = calibrated_mean_offset,
                        linear = calibrated_linear,
                        poly = calibrated_poly,
                        gam = calibrated_gam,
                        rf = calibrated_rf,
                        svr = calibrated_svr,
                        ann = calibrated_ann,
                        cnn = calibrated_cnn,
                        rnn = calibrated_rnn,
                        dbn = calibrated_dbn)

results_calibrated <- lapply(calibrated_data, function(x) {
        data.frame(method = names(calibrated_data)[which(calibrated_data == x)],
                   r2 = round(cor(x$reference, x$calibrated)^2, 4),
                   rmse = round(sqrt(mean((x$reference - x$calibrated)^2)), 4))
}) %>%
        do.call(rbind, .)


# QUALITY OF ADJUSTMENT

# Quality of adjustment metrics

# 1. Mean Absolute Error
mae <- mean(abs(reference_data - low_cost_data))

# 2. Root Mean Squared Error
rmse <- sqrt(mean((reference_data - low_cost_data)^2))


# DATA STRUCTURE

# Load the data into R
data <- read.csv("pm25_data.csv")

# View the first few rows of the data
head(data)

# Output
  Timestamp  GRIMM   PMS1   PMS2   PMS3   PMS4
1   00:00:00  17.2   16.0   15.1   15.7   15.9
2   00:01:00  17.3   15.8   14.9   15.5   15.7
3   00:02:00  17.1   16.1   15.2   15.8   15.9
4   00:03:00  17.2   16.0   15.1   15.7   15.9
5   00:04:00  17.3   15.9   14.9   15.6   15.8
6   00:05:00  17.2   16.0   15.1   15.7   15.9

# Summary statistics for the data
summary(data)

# Output
 Timestamp           GRIMM           PMS1           PMS2           PMS3           PMS4      
 Min.   :00:00:00   Min.   :14.0   Min.   :14.0   Min.   :14.0   Min.   :14.0   Min.   :14.0  
 1st Qu.:06:30:00   1st Qu.:16.0   1st Qu.:15.0   1st Qu.:15.0   1st Qu.:15.0   1st Qu.:15.0  
 Median :13:00:00   Median :17.0   Median :16.0   Median :16.0   Median :16.0   Median :16.0  
 Mean   :13:00:00   Mean   :17.1   Mean   :15.9   Mean   :15.9   Mean   :15.9   Mean   :15.9  
 3rd Qu.:19:30:00   3rd Qu.:18.0   3rd Qu.:16.8   3rd Qu.:16.8   3rd Qu.:16.8   3rd Qu.:16.8  
 Max.   :01:00:00   Max.   :19.0   Max.   :18.0   Max.   :18.0   Max.   :18.0   Max.   :18.0  
