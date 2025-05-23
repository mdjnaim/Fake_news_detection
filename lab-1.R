View(fake_news_dataset)

install.packages("readr")
install.packages("dplyr")
install.packages("FSelector")
install.packages("infotheo")
install.packages("tm")
install.packages("NLP")
install.packages("psych")
library(readr) 
library(dplyr)
library(FSelector)
library(infotheo)
library(tm)
library(NLP)
library(psych)

df <- read_csv("fake_news_dataset.csv") %>%
  select(text, label, source, author, category) %>%
  na.omit() %>%
  mutate(
    label = as.factor(label),
    category = as.factor(category),
    source = as.factor(source)
  )

corpus <- VCorpus(VectorSource(df$text))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stripWhitespace)


dtm <- DocumentTermMatrix(corpus)
dtm <- removeSparseTerms(dtm, 0.999)

if (nrow(dtm) > 0 && ncol(dtm) > 0) {
  dtm_df <- as.data.frame(as.matrix(dtm), stringsAsFactors = FALSE)

if(nrow(dtm_df) == nrow(df)) {
  df_ready <- cbind(dtm_df, label = df$label)
} else {
  min_rows <- min(nrow(dtm_df), nrow(df))
  df_ready <- cbind(dtm_df[1:min_rows, ], label = df$label[1:min_rows])
} 
}


anova_scores <- apply(df_ready[, !names(df_ready) %in% "label"], 2, function(x) {
  if (length(unique(x)) <= 1) return(1)
  anova_result <- summary(aov(x ~ df_ready$label))[[1]]
  if ("Pr(>F)" %in% colnames(anova_result)) {
    return(anova_result$`Pr(>F)`[1])
  } else {
    return(1)
  }
})
anova_scores_adjusted <- p.adjust(anova_scores, method = "BH")
anova_selected <- names(sort(anova_scores_adjusted))[1:10]
print(summary(aov(df_ready[, 1] ~ df_ready$label))[[1]])


df_numeric <- df_ready
df_numeric$label_num <- as.numeric(df_numeric$label)
kendall_scores <- apply(df_numeric[, !(names(df_numeric) %in% c("label", "label_num"))], 2, function(x) {
  cor(x, df_numeric$label_num, method = "kendall")
})
kendall_selected <- names(sort(abs(kendall_scores), decreasing = TRUE))[1:10]
cat("Top 10 features by Kendall's Tau correlation:\n")
for (feature in kendall_selected) {
  cat(feature, ": ", round(kendall_scores[feature], 2), "\n")
}


df_chi <- data.frame(lapply(df_numeric, function(x) 
  if(is.numeric(x)) cut(x, 3) else x))
chi_scores <- chi.squared(label ~ ., df_chi)
chi_selected <- cutoff.k(chi_scores, 10)
cat("Top 10 features by Chi-Squared test:\n")
for (feature in chi_selected) {
  cat(feature, ":", chi_scores[feature," "], "\n")
}

df_mi <- discretize(df_numeric[, -which(names(df_numeric) %in% c("label", "label_num"))])
mi_scores <- sapply(df_mi, function(x) mutinformation(x, df_numeric$label))
mi_selected <- names(sort(mi_scores, decreasing = TRUE))[1:10]
cat("\nTop 10 features by Mutual Information:\n")
for (feature in mi_selected) {
  cat(feature, ": ", mi_scores[feature], "\n")
}




