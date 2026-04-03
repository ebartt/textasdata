library(tidyverse)
library(tokenizers)
library(quanteda)
library(quanteda.textplots)
library(devtools)
library(quanteda.corpora)
library(irr)
library(xtable)

###########Loading and cleaning data################
setwd("~/Desktop/UCSD_CSS/CSS classes/POLI176/finalproj")
undf <- read_csv('UNGDspeeches-1.csv')
vdem <- readRDS('V-Dem-CD-v15_rds/V-Dem-CD-v15.rds')

#cleaning up the vdem dataset to keep the years we need
keep_years = seq(from = 1970, to = 2018, by = 1)
vdem = vdem[(vdem$year %in% keep_years) & grepl("12-31",vdem$historical_date), ] %>% select(country_text_id, year, v2x_polyarchy)
vdem = vdem %>% 
  rename(
    country = country_text_id,
  )

#cleaning the countrycodes (UN Methodology dataset) to make it easier to work with
countrycodes = read_csv('countrycodes.csv') %>% select(`Region Name`, `Sub-region Name`, `Intermediate Region Name`, `Country or Area`, `ISO-alpha3 Code`, `Least Developed Countries (LDC)`, `Land Locked Developing Countries (LLDC)`, `Small Island Developing States (SIDS)`)
countrycodes = countrycodes %>% 
  rename(
    region_name = `Region Name`,
    subregion = `Sub-region Name`,
    intermediate_region = `Intermediate Region Name`,
    country_name = `Country or Area`,
    country = `ISO-alpha3 Code`,
  )
countrycodes$LDC_01 = ifelse(countrycodes$`Least Developed Countries (LDC)` == 'x', 1, 0)
countrycodes$LLDC_01 = ifelse(countrycodes$`Land Locked Developing Countries (LLDC)` == 'x', 1, 0)
countrycodes$SIDS_01 = ifelse(countrycodes$`Small Island Developing States (SIDS)` == 'x', 1, 0)
countrycodes = countrycodes %>% select(-c(`Least Developed Countries (LDC)`, `Land Locked Developing Countries (LLDC)`, `Small Island Developing States (SIDS)`))

#joining them all together
undf$id <- 1:nrow(undf)
undf = left_join(undf, countrycodes, by = "country")
undf = left_join(undf, vdem, by = c("country", "year"))

##########HAND CODING############

#to select hand coding documents:
#export
#set.seed(92073)
#handcoding <- undf[sample(1:nrow(undf), 100),]
#write.csv(handcoding, "UNGDhandcoding.csv", row.names=F)

#import handcoding
handcoded <- read_csv("UNGDhandcoding_final.csv")
handcoded_tojoin = handcoded %>% select("doc_id", "country", "session", "year", "critical_01_1", "critical_01_2")
#looking at handcoding
undf <- left_join(undf, handcoded_tojoin, by=c("doc_id", "country", "session", "year"))
table(undf$critical_01_1, undf$critical_01_2)

undf$critical <- ifelse(undf$critical_01_1==1 | undf$critical_01_2==1, 1, 0)
table(undf$critical)

#intercoder reliability:
#Confusion Matrix
table(undf$critical_01_1, undf$critical_01_2)
#Krippendorff's alpha
kripp.alpha(t(undf[,c("critical_01_1", "critical_01_2")]))

#################PROCESSING THE CORPUS#######################
#pre-processing: bag of words (from lectures)
corpus_undf <- corpus(undf, text_field = "text")
toks <- tokens(corpus_undf, remove_punct = TRUE, remove_numbers=TRUE)
toks <- tokens_wordstem(toks)
toks <- tokens_select(toks,  stopwords("en"), selection = "remove")
dfm <- dfm(toks)
dfm

dfm_trimmed <- dfm_trim(dfm, min_docfreq = 0.05, docfreq_type = "prop")
dfm_trimmed
#8,093 documents, 2,481 features (76.32% sparse)
textplot_wordcloud(dfm_trimmed, col="black")


#############Building model##################
#from discussion 6:
unlabeled <- which(is.na(undf$critical))
labeled <- which(!is.na(undf$critical))

#creating training vs. validation
set.seed(32123)
training <- sample(labeled, round(length(labeled)*.75))
validation <- labeled[!labeled%in%training]

#Create separate dfm's for each
dfmat_train <- dfm_subset(dfm, docvars(corpus_undf, "id")%in%training)
dfmat_val <- dfm_subset(dfm, docvars(corpus_undf, "id")%in%validation)

############LASSO###############
library(glmnet)

#Adding inverse frequency weighting
w_pos <- sum(docvars(dfmat_train, "critical")==0)/nrow(dfmat_train)
w_neg <- sum(docvars(dfmat_train, "critical")==1)/nrow(dfmat_train)
weights <- ifelse(docvars(dfmat_train, "critical") == 1, w_pos, w_neg)

lasso.1 <- glmnet(dfmat_train, docvars(dfmat_train, "critical"),
                  family="binomial", alpha=1)

#These are all of the different lambdas glmnet used
lasso.1$lambda

sort(lasso.1$beta[,40], decreasing=T)[1:40]
sort(lasso.1$beta[,40], decreasing=F)[1:40]

#out of sample
predict.test <- predict(lasso.1, dfmat_val, type="response")
table(docvars(dfmat_val, "critical"),as.numeric(predict.test[,40]))
table(docvars(dfmat_val, "critical"),as.numeric(predict.test[,1]))

#Fit again
#Cross-validation with Lasso
cv <- cv.glmnet(dfmat_train, docvars(dfmat_train, "critical"),
                family="binomial", alpha=1, type.measure = "class", weights=weights
                )
plot(log(cv$lambda), cv$cvm, xlab="Log Lambda", ylab="Misclassification error")

predict.val <- predict(cv, dfmat_val, type="class", lambda=cv$lambda.min)

tab_val <- table(docvars(dfmat_val, "critical"), predict.val)
tab_val

#precision
diag(tab_val)/colSums(tab_val)
#recall
diag(tab_val)/rowSums(tab_val)

#Confusion matrix
conf_matrix <- confusionMatrix(factor(predict.val), 
                               factor(docvars(dfmat_val, "critical")),
                               mode="prec_recall", positive="1")
print(conf_matrix)


#Applying this model to the whole dataset:
undf$predict.critical <- as.numeric(as.character(predict(cv, dfm, type="class", lambda=cv$lambda.min)))

#Prediction of critical for the whole corpus
prop.table(table(undf$predict.critical))
#handcoded proportion:
prop.table(table(undf$critical))


#What are some speeches that are predicted to be critical?
set.seed(87394)
sample(undf$doc_id[undf$predict.critical==1],10)

##########ANALYSIS##########

#Which regions have the highest proportion of critical speeches?
regions = undf %>% 
  filter(!is.na(region_name)) %>% 
  select(country, session, year, region_name, subregion, country_name, LDC_01, LLDC_01, SIDS_01, predict.critical) %>% 
  group_by(region_name) %>% 
  summarise(prop_critical = sum(predict.critical)/ n() )
regions %>% arrange(desc(prop_critical))

#Which subregions?
subregions = undf %>% 
  select(country, session, year, region_name, subregion, country_name, LDC_01, LLDC_01, SIDS_01, predict.critical) %>% 
  group_by(subregion) %>% 
  summarise(prop_critical = sum(predict.critical)/ n() )
subregions %>% arrange(desc(prop_critical
                            ))

#putting region and subregion together
undf %>% 
  filter(!is.na(region_name) & !is.na(subregion)) %>% 
  select(country, session, year, region_name, subregion, country_name, LDC_01, LLDC_01, SIDS_01, predict.critical) %>% 
  group_by(region_name, subregion) %>% 
  summarise(prop_critical = sum(predict.critical)/ n() ) %>% 
  arrange(desc(prop_critical))
  
#Do least developed countries criticize more? are they predicted to?
undf %>% 
  select(country, session, year, region_name, subregion, country_name, LDC_01, LLDC_01, SIDS_01, predict.critical) %>% 
  group_by(LDC_01) %>% 
  summarise(prop_critical = sum(predict.critical) / n())

#what about Land- Locked developing countries?
undf %>% 
  select(country, session, year, region_name, subregion, country_name, LDC_01, LLDC_01, SIDS_01, predict.critical) %>% 
  group_by(LLDC_01) %>% 
  summarise(prop_critical = sum(predict.critical) / n())

#What about Small Island Developing Countries?
undf %>% 
  select(country, session, year, region_name, subregion, country_name, LDC_01, LLDC_01, SIDS_01, predict.critical) %>% 
  group_by(SIDS_01) %>% 
  summarise(prop_critical = sum(predict.critical) / n())

#top countries critical of the UN, by count
undf %>% 
  group_by(country, country_name) %>% 
  filter(predict.critical == 1) %>% 
  summarise(count = n()) %>% 
  arrange(desc(count)) %>% 
  head(10)

#top countries critical of the UN, by proportion
undf %>% 
  group_by(country, country_name) %>% 
  filter(predict.critical == 1) %>% 
  summarise(count = n(),
            prop_critical = sum(predict.critical)/ count) %>% 
  arrange(desc(count)) %>% 
  head(10)

#least critical countries, by count
undf %>% 
  group_by(country) %>% 
  filter(predict.critical == 0) %>% 
  summarise(count = n()) %>% 
  arrange(desc(count)) %>% 
  head(10)

#least critical countries, by proportion
undf %>% 
  group_by(country, country_name) %>% 
  filter(predict.critical == 0) %>% 
  summarise(count = n(),
            prop_critical = sum(predict.critical)/ count) %>% 
  arrange(desc(count)) %>% 
  head(10)

#group of seven
undf %>% 
  filter(country%in%c("CAN", "FRA", "DEU", "ITA", "JPN", "GBR", "USA")) %>% 
  group_by(country, country_name) %>% 
  summarise(count = n(),
            prop_critical = sum(predict.critical)/ count) %>% 
  arrange(desc(prop_critical))


########PLOTS########
#Democracy vs. predict.critical
undf %>% 
  ggplot(aes(x = factor(predict.critical), y = v2x_polyarchy, fill = factor(predict.critical))) +
  geom_boxplot()+
  labs(title = "Mean democracy score, by critical score",
       x = "Predict.critial",
       y = "Democracy score")+
  theme_minimal()

#predicted critical speeches over time
undf %>% 
  group_by(year) %>% 
  summarise(prop_crit_yr = sum(predict.critical) / n()) %>% 
  ggplot(aes(x = year, y = prop_crit_yr)) +
  geom_point() +
  geom_line(linetype = "dotted")+
  theme_minimal()

#regional criticism
#as a boxplot
undf %>% 
  filter(!is.na(country) & !is.na(region_name)) %>% 
  select(country, session, year, region_name, subregion, country_name, predict.critical) %>% 
  group_by(country_name) %>% 
  summarise(prop_critical = sum(predict.critical)/ n(),
            region_name = region_name) %>% 
  ggplot(aes(x = reorder(region_name, prop_critical, decreasing = T), y = prop_critical, fill = region_name)) +
  geom_boxplot(position = position_dodge())+
  labs(x = "Region",
       y = "Proportion of critical speeches",
       title = "Region vs. criticism, by proportion") +
  geom_hline(yintercept = .193, linetype = "dotted")

#subregions, colored by region
#as a boxplot
undf %>% 
  filter(!is.na(country) & !is.na(region_name)) %>% 
  select(country, session, year, region_name, subregion, country_name, predict.critical) %>% 
  group_by(country_name) %>% 
  summarise(prop_critical = sum(predict.critical)/ n(),
            subregion = subregion,
            region_name = region_name) %>% 
  ggplot(aes(x = reorder(subregion, prop_critical, decreasing = T), y = prop_critical, fill = region_name)) +
  geom_boxplot(position = position_dodge())+
  labs(x = "Subregion",
       y = "Proportion of critical speeches",
       title = "Subregion vs. criticism, by proportion") +
  theme(axis.text.x = element_text(angle = 60, vjust = 1, hjust=1))

#ten most critical countries
undf %>% 
  filter(!is.na(country)) %>% 
  select(country, session, year, region_name, subregion, country_name, predict.critical) %>% 
  group_by(country_name, country) %>% 
  summarise(prop_critical = sum(predict.critical)/ n()) %>% 
  arrange(desc(prop_critical)) %>% 
  head(10)
