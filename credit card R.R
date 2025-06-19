library(tidyverse)
library(randomForest)
library(caret)
library(ROSE)
library(pROC)

data <- read.csv("C:\\Users\\HP\\Downloads\\credit card.csv")

# Check structure
str(data)

# Summary statistics
summary(data)

# Check class imbalance (0 = Legitimate, 1 = Fraud)
table(data$Class)
prop.table(table(data$Class)) * 100  # Fraud percentage

# Normalize 'Amount' (important for Random Forest)
data$Amount <- scale(data$Amount)

# Remove 'Time' (often not useful in static models)
data <- data %>% select(-Time)

# Convert 'Class' to factor (required for classification)
data$Class <- as.factor(data$Class)
set.seed(113)  # For reproducibility
split <- createDataPartition(data$Class, p = 0.7, list = FALSE)
train_data <- data[split, ]
test_data <- data[-split, ]

# Verify distribution
table(train_data$Class)
table(test_data$Class)

# First, check your data structure
str(train_data)

# Convert all predictors to numeric (if they aren't already)
train_data <- train_data %>%
  mutate(across(-Class, as.numeric))

# Ensure Class is a factor
train_data$Class <- as.factor(train_data$Class)

# Now try ROSE again
train_balanced <- ROSE(Class ~ ., data = train_data, seed = 123)$data

# Check new balance
table(train_balanced$Class)


set.seed(113)
rf_model <- randomForest(
  Class ~ ., 
  data = train_balanced,
  ntree = 100,            # Number of trees
  mtry = sqrt(ncol(train_balanced) - 1),  # Features per split
  importance = TRUE,       # Track feature importance
  strata = train_balanced$Class  # Stratified sampling
)

# Print model summary
print(rf_model)

# Ensure both predictions and test_data$Class have the same factor levels
predictions <- factor(predictions, levels = levels(test_data$Class))

# Now compute confusion matrix
conf_matrix <- confusionMatrix(predictions, test_data$Class)

print(conf_matrix)

# ROC-AUC curve
prob_predictions <- predict(rf_model, test_data, type = "prob")[, 2]
roc_curve <- roc(test_data$Class, prob_predictions)
plot(roc_curve, main = "ROC Curve")
auc(roc_curve)  # Higher AUC = better model

# Plot feature importance
varImpPlot(rf_model, main = "Feature Importance")

# Extract importance scores
importance_scores <- importance(rf_model)
importance_scores[order(-importance_scores[, "MeanDecreaseGini"]), ]

# Save model
saveRDS(rf_model, "fraud_detection_rf_model.rds")

# Load model later
loaded_model <- readRDS("fraud_detection_rf_model.rds")

# Example: Predict a single transaction
new_data <- data.frame(
  V1 = -1.36, V2 = -0.96, V3 = 1.19, V4 = -0.42, V5 = -0.33,
  V6 = -0.43, V7 = -0.43, V8 = 0.16, V9 = -0.05, V10 = -0.53,
  V11 = -0.77, V12 = -0.55, V13 = -0.24, V14 = -0.21, V15 = 0.01,
  V16 = -0.21, V17 = -0.33, V18 = -0.28, V19 = 0.01, V20 = -0.09,
  V21 = -0.35, V22 = -0.19, V23 = -0.33, V24 = -0.01, V25 = 0.02,
  V26 = -0.01, V27 = 0.01, V28 = -0.01, Amount = 1.23
)

# Predict fraud probability
predict(rf_model, new_data, type = "prob")[, 2]  # Fraud probability


# Load required packages
library(ggplot2)
library(dplyr)

# Prepare data for visualization
fraud_counts <- data %>%
  count(Class) %>%
  mutate(Class = ifelse(Class == 1, "Fraud", "Legitimate"),
         percentage = n/sum(n)*100)

# Create professional pie chart
ggplot(fraud_counts, aes(x = "", y = n, fill = Class)) +
  geom_bar(width = 1, stat = "identity", color = "white") +
  coord_polar("y", start = 0) +
  scale_fill_manual(values = c("#FF5A5F", "#25A18E")) +  # Red for fraud, teal for legitimate
  geom_text(aes(label = paste0(round(percentage, 2), "%")), 
            position = position_stack(vjust = 0.5),
            size = 6, color = "white", fontface = "bold") +
  labs(title = "Extreme Class Imbalance in Fraud Data",
       subtitle = "Only 0.17% of transactions are fraudulent",
       fill = "Transaction Type") +
  theme_void() +
  theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 12, hjust = 0.5),
        legend.position = "bottom",
        legend.title = element_text(face = "bold"))













