# VARIABILI ---------------------------------------------------------------

# Per tutte le dicotomiche 0 = False
# 1) PassengerId - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a 
#                  group the passenger is travelling with and pp is their number within the group. People 
#                  in a group are often family members, but not always.
# 2) HomePlanet - The planet the passenger departed from, typically their planet of permanent residence.
# 3) CryoSleep - Indicates whether the passenger elected to be put into suspended animation for the duration 
#                of the voyage. Passengers in cryosleep are confined to their cabins.
# 4) Cabin - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can 
#            be either P for Port or S for Starboard.
# 5) Destination - The planet the passenger will be debarking to.
# 6) Age - The age of the passenger.
# 7) VIP - Whether the passenger has paid for special VIP service during the voyage.
# 8/12) RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each 
#                                                           of the Spaceship Titanic's many luxury amenities.
# 13) Name - The first and last names of the passenger.
# 14) Transported - Whether the passenger was transported to another dimension. 

#a me piacerebbe prevedere correttamente i non trasportati in modo da poterli salvare ovvero i FALSE

# CARICAMENTO LIBRERIE ----------------------------------------------------

library(patchwork)
library(reshape2)
library(ROCR)
library(MASS)
library(tidyverse)
library(ggplot2)
library(dplyr)
library(plyr)
library(VIM)
library(mice)
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)
library(Boruta)
library(car)
library(factorMerger)
library(parsnip)
library(caretEnsemble)
library(caTools)
library(funModeling)
library(DALEX)
library(breakDown)

set.seed(1)

# IMPORTAZIONE DATI E PRIME ANALISI ---------------------------------------

train = read.csv("train.csv", header = TRUE, sep = ",", dec = ".",
                stringsAsFactors=TRUE, na.strings=c("NA","NaN", " ", "NULL", ""))

train$Transported = as.factor(train$Transported)
train$CryoSleep = as.factor(train$CryoSleep)
train$VIP = as.factor(train$VIP)
levels(train$Destination) = make.names(levels(train$Destination))

head(train)

train$PassengerId = as.character(train$PassengerId)
train$Cabin = as.character(train$Cabin)
train$Name = as.character(train$Name)

summary(train)
table(train$Transported) / nrow(train)
table(train$Transported, train$CryoSleep) / nrow(train)
table(train$Transported, train$HomePlanet) / nrow(train)
table(train$Transported, train$VIP) / nrow(train)
table(train$Transported, train$Destination) / nrow(train)

# Grafici

plot_gg = function(column){
  ggplot(data = train, mapping = aes(x = {{column}}, fill = Transported)) +
    geom_bar(position = 'dodge') +
    scale_fill_manual('Legenda', values = c("purple4", "pink2"))
}

plot_gg(HomePlanet) + 
  ggtitle("Numero di trasportati per HomePlanet") 
# Dei provenienti dalla terra si vede che sono di più quelli non trasportati, dall'europa il contrario,
# da Marte pi? o meno simile.

plot_gg(CryoSleep) + 
  ggtitle("Numero di trasportati per CryoSleep") 
# Dei pazienti in CryoSleep sono molti di pi? quelli trasportati mentre per quelli non in CryoSleep vieceversa,
# coerentemente con il fatto che i pazienti a riposo non hanno potuto lasciare le loro cabine.

plot_gg(Destination) + 
  ggtitle("Numero di trasportati per Destination") 
# Dei passeggeri diretti a TRAPPIST-1e, cio? la maggior parte dei passeggeri, sono di pi? quelli che non sono
# stati trasportati, per 55 Cancri e viceversa mentre per PSO J318.5-22 uguale.

plot_gg(VIP) + 
  ggtitle("Numero di trasportati per VIP") 
# I passeggeri VIP sono molto minori dei passeggeri normali ci? nonostante si nota che tra di essi sono
# di pi? quelli che non vengono trasportati a differenza dei non VIP dove avviene l'opposto e dell'intera
# popolazione dove ? ugualmente distribuito.




# MISSING -----------------------------------------------------------------

train = train %>% 
  mutate(CabinDeck = str_sub(Cabin, 1, 1),
         CabinNum = as.numeric(str_sub(Cabin, 3, -3)),
         CabinSide = str_sub(Cabin, -1, -1),
         idGroup = substring(PassengerId, 1, 4),
         idNum = substring(PassengerId, 6, 7)
  )
# studiamo CabinDeck
plot_gg(CabinDeck) + 
  ggtitle("Numero di trasportati per CabinDeck") 

# studiamo CabinSide
plot_gg(CabinSide) + 
  ggtitle("Numero di trasportati per CabinSide") 

#Ricaviamo il numero di componenti per ogni gruppo
size = count(train, "idGroup")
names(size) = c("idGroup", "GroupSize")
train = left_join(train, size, by = "idGroup")
train$GroupSize = as.factor(train$GroupSize)

# studiamo GroupSize
plot_gg(GroupSize) + 
  ggtitle("Numero di trasportati per Groupsize") 

train$CabinDeck = as.factor(train$CabinDeck)
train$CabinSide = as.factor(train$CabinSide)


train2 = train
train2$PassengerId = NULL
train2$Cabin = NULL
train2$Name = NULL
train2$idGroup = NULL
train2$idNum = NULL

missing = as.data.frame(sapply(train2, function(x)(sum(is.na(x)))))
names(missing) = "MISSING"
missing$PERC = missing$MISSING / nrow(train2)
missing = missing[(missing$PERC != 0),]

missingness = aggr(train2[, rownames(missing)],col=c('pink2','purple4'),numbers=TRUE,sortVars=TRUE,
                   labels=names(df),cex.axis=.7,gap=2)

covdata = train2[,-11]
tempData = mice(covdata, m=1, maxit=20, meth='cart', seed=500)
train_imputed = complete(tempData,1)  

train_imputed = cbind(train_imputed, train2$Transported)  
colnames(train_imputed)[15] = "Transported"

missingness = aggr(train_imputed,col=c('navyblue','yellow'),numbers=TRUE,sortVars=TRUE,
                   labels=names(df),cex.axis=.7,gap=2)

names(train_imputed)[15] = c("Transported")

write.table(train_imputed, file = "dataset_imputed.csv", 
            sep = ";", row.names = FALSE)

Trainindex = createDataPartition(y = train$Transported, p = .75, list = FALSE)

train = train_imputed[Trainindex,]
validation = train_imputed[-Trainindex,]

# FEATURE ENGEENERING --------

# Ricaviamo ponte, lato, e numero cabina e scomponiamo ID


#raggruppo le spese
train$FoodExpenses = train$FoodCourt + train$ShoppingMall
train$OtherExpendes = train$RoomService + train$Spa + train$VRDeck
validation$FoodExpenses = validation$FoodCourt + validation$ShoppingMall
validation$OtherExpendes = validation$RoomService + validation$Spa + validation$VRDeck

train$FoodCourt = NULL
train$ShoppingMall = NULL
train$RoomService = NULL
train$Spa = NULL
train$VRDeck = NULL
validation$FoodCourt = NULL
validation$ShoppingMall = NULL
validation$RoomService = NULL
validation$Spa = NULL
validation$VRDeck = NULL


  
# MODEL SELCTION ----------------------------------------------------------

cvCtrl = trainControl(method = "cv", number=10, search="grid", classProbs = TRUE)
rpartTuneCvA = train(Transported ~ ., data = train, method = "rpart",
                      tuneLength = 10,
                      trControl = cvCtrl)

rpartTuneCvA
getTrainPerf(rpartTuneCvA)

plot(varImp(object=rpartTuneCvA),main="train tuned - Variable Importance")
plot(rpartTuneCvA)

vi_t = as.data.frame(rpartTuneCvA$finalModel$variable.importance)
viname_t = row.names(vi_t)
head(viname_t)

#Random Forest
rfTune = train(Transported ~ ., data = train, method = "rf",
                tuneLength = 10,
                trControl = cvCtrl)

rfTune
getTrainPerf(rfTune)

plot(varImp(object=rfTune),main="train tuned - Variable Importance")
plot(rfTune)

vi_rf = data.frame(varImp(rfTune)[1])
vi_rf$var = row.names(vi_rf)
head(vi_rf)
viname_rf = vi_rf[,2]

#Boruta
boruta.train = Boruta(Transported ~., data = train_imputed, doTrace = 1)
plot(boruta.train, xlab = "features", xaxt = "n", ylab="MDI")

print(boruta.train)

boruta.metrics = attStats(boruta.train)
head(boruta.metrics)
table(boruta.metrics$decision)

vi_bo = subset(boruta.metrics, decision == "Confirmed")
head(vi_bo)  
viname_bo = rownames(vi_bo)

viname_t
viname_rf
viname_bo
selected = c("Transported", "HomePlanet", "CryoSleep", "Destination", "Age", "CabinDeck", 
             "CabinNum", "CabinSide", "GroupSize", "FoodExpenses", "OtherExpendes")

#Dataest selected
train_selected = train[,selected]
validation_selected = validation[,selected]

# LOGISTICO ---------------------------------------------------------------

cvCtrl = trainControl(method = "boot", number=10, search="grid", classProbs = TRUE)
glm = train(Transported ~ ., data = train_selected,  method = "glm",   trControl = cvCtrl,
              preProcess=c("corr", "nzv"), metric = "Sens")
glm
confusionMatrix(glm)

glmpred = predict(glm, validation_selected)
glmpred_p = predict(glm, validation_selected, type = c("prob"))

confusionMatrix(glmpred, validation_selected$Transported)

# KNN ---------------------------------------------------------------

train_knn = train_selected
train_knn$HomePlanet = as.integer(train_knn$HomePlanet)
train_knn$CabinDeck = as.integer(train_knn$CabinDeck)
train_knn$CabinSide = as.integer(train_knn$CabinSide)
train_knn$CryoSleep = as.integer(train_knn$CryoSleep)
train_knn$Destination = as.integer(train_knn$Destination)

cvCtrl = trainControl(method = "boot", number=10, searc="grid", 
                      summaryFunction = twoClassSummary, 
                      classProbs = TRUE)
knn = train(Transported ~., data=train_selected,
            method = "knn", tuneLength = 10,
            preProcess = c("center", "scale", "corr", "nzv"),
            trControl = cvCtrl,
            metric = "Sens")

knn
confusionMatrix(knn)

KNNPred_p = predict(knn, validation_selected, type = c("prob"))
KNNPred = predict(knn, validation_selected)

confusionMatrix(KNNPred, validation_selected$Transported)

# SDA ---------------------------------------------------------------------

train_sda = train_knn

cvCtrl = trainControl(method = "boot", number=10, searc="grid", 
                      summaryFunction = twoClassSummary, 
                      classProbs = TRUE)
sda = train(Transported ~., data=train_selected,
            method = "sda", tuneLength = 10,
            preProcess = c("corr", "nzv"),
            metric="Sens",
            trControl = cvCtrl)
sda
sda$results$Sens
sda$bestTune
confusionMatrix(sda)

sdaPred_p = predict(sda, validation_selected, type = c("prob"))
sdaPred = predict(sda, validation_selected)

confusionMatrix(sdaPred, validation_selected$Transported)

# LASSO -------------------------------------------------------------------

cvCtrl = trainControl(method = "boot", number=10, searc="grid", 
                      summaryFunction = twoClassSummary, 
                      classProbs = TRUE)
lasso = train(Transported ~., data=train,
            method = "glmnet", tuneLength = 10,
            preProcess = c("corr", "nzv"),   
            metric="Sens",
            trControl = cvCtrl)
lasso$results$Spec
lasso$bestTune
confusionMatrix(lasso)

lassoPred_p = predict(lasso, validation, type = c("prob"))
lassoPred = predict(lasso, validation)

confusionMatrix(lassoPred, validation$Transported)

# PLS ---------------------------------------------------------------------

cvCtrl = trainControl(method = "boot", number=10, searc="grid", 
                      summaryFunction = twoClassSummary, 
                      classProbs = TRUE)
pls = train(Transported ~., data=train,
              method = "pls", tuneLength = 10,
              preProcess = c("center"),
              metric="Sens",
              trControl = cvCtrl)

pls
confusionMatrix(pls)

plsPred_p = predict(pls, validation, type = c("prob"))
plsPred = predict(pls, validation)

confusionMatrix(plsPred, validation$Transported)

# NAIVE -------------------------------------------------------------------

cvCtrl = trainControl(method = "boot", number=10, searc="grid", 
                      summaryFunction = twoClassSummary, 
                      classProbs = TRUE)
naive = train(Transported ~., data=train_selected,
            method = "naive_bayes", tuneLength = 10,
            preProcess = c("corr", "nzv"), 
            metric="Sens",
            trControl = cvCtrl)

naive
confusionMatrix(naive)

naivePred_p = predict(naive, validation_selected, type = c("prob"))
naivePred = predict(naive, validation_selected)

confusionMatrix(naivePred, validation_selected$Transported)

# TREE --------------------------------------------------------------------

tree_rpart = rpart(Transported ~ ., data = train, 
                   method = "class",
                   cp = 0, 
                   minsplit = 1)
tree_rpart$cptable
rpart.plot(tree_rpart, type = 4, extra = 1)

tree_pruned = prune(tree_rpart, cp=  
                      tree_rpart$cptable[which.min(tree_rpart$cptable[,"xerror"]),"CP"])
rpart.plot(tree_pruned, type = 4, extra = 1)

treePred_pruned_p = predict(tree_pruned, validation, type = c("prob"))
treePred_pruned = predict(tree_pruned, validation, type = c("class"))

confusionMatrix(treePred_pruned, validation$Transported, positive="True")

# BAGGING -----------------------------------------------------------------

cvCtrl = trainControl(method = "boot", number=10, searc="grid", 
                      summaryFunction = twoClassSummary, 
                      classProbs = TRUE)

bagging = train(Transported ~., data=train,
           method = "treebag", ntree = 250,
           trControl = cvCtrl)

bagging
confusionMatrix(bagging)

baggingPred_p = predict(bagging, validation, type = c("prob"))
baggingPred = predict(bagging, validation)

confusionMatrix(baggingPred, validation_selected$Transported)

# GRADIENT BOOSTING -------------------------------------------------------

cvCtrl = trainControl(method = "boot", number=10, searc="grid", 
                      summaryFunction = twoClassSummary, 
                      classProbs = TRUE)

gbm_tune = expand.grid(
  n.trees = 500,
  interaction.depth = 4,
  shrinkage = 0.1,
  n.minobsinnode = 10
)

gb = train(Transported ~., data=train,
              method = "gbm", tuneLength = 10,
              metric="Sens", tuneGrid = gbm_tune,
              trControl = cvCtrl)

gb
confusionMatrix(gb)

gbPred_p = predict(gb, validation, type = c("prob"))
gbPred = predict(gb, validation)

confusionMatrix(gbPred, validation$Transported)

# RANDOM FOREST -----------------------------------------------------------

cvCtrl = trainControl(method = "boot", number=10, searc="grid", 
                      summaryFunction = twoClassSummary, 
                      classProbs = TRUE)

rf = train(Transported ~., data=train,
           method = "rf", tuneLength = 10,
           metric="Sens",
           trControl = cvCtrl,
           verbose = FALSE)

rf
confusionMatrix(rf)

rfPred_p = predict(rf, validation, type = c("prob"))
rfPred = predict(rf, validation)

confusionMatrix(rfPred, validation$Transported)

# EMPLEMENTAZIONE NN ------------------------------------------------------

cvCtrl = trainControl(method = "boot", number=10, searc="grid", 
                      summaryFunction = twoClassSummary, 
                      classProbs = TRUE)

#nn 
nn = train(Transported ~., data=train_selected,
                     method = "nnet",
                     preProcess = c("scale", "corr", "nzv"), 
                     tuneLength = 5, metric="Sens", trControl=cvCtrl, trace = TRUE,
                     maxit = 100)

plot(nn)
print(nn)
getTrainPerf(nn)

confusionMatrix(nn)

nnPred_p = predict(nn, validation_selected, type = c("prob"))
nnPred = predict(nn, validation)

confusionMatrix(nnPred, validation_selected$Transported)

#nn tuned
cvCtrl = trainControl(method = "boot", number=10, searc="grid", 
                      summaryFunction = twoClassSummary, 
                      classProbs = TRUE)

tunegrid = expand.grid(size=c(1:5), decay = c(0.001, 0.01, 0.05 , .1, .3))

nn_tuned = train(Transported ~., data=train_selected,
                         method = "nnet",
                         preProcess =  c("scale", "corr", "nzv"), 
                         tuneLength = 10, metric= "Sens", trControl=cvCtrl, tuneGrid=tunegrid,
                         trace = TRUE,
                         maxit = 300)

plot(nn_tuned)
print(nn_tuned)
getTrainPerf(nn_tuned)

confusionMatrix(nn_tuned)

nn_tunedPred_p = predict(nn_tuned, validation_selected, type = c("prob"))
nn_tunedPred = predict(nn_tuned, validation_selected)

confusionMatrix(nn_tunedPred, validation_selected$Transported)

# STACKING ----------------------------------------------------------------

cvCtrl = trainControl(method = "boot", number=10, searc="grid", 
                      summaryFunction = twoClassSummary, 
                      classProbs = TRUE)

model_list = caretList(
  Transported ~., data = train,
  trControl = cvCtrl,
  methodList = c("glm", "knn", "naive_bayes", "rf", "nnet")
)

glm_ensemble = caretStack(
  model_list,
  method="glm",
  metric="Sens",
  trControl = cvCtrl
)

model_preds = lapply(model_list, predict, newdata = validation, type="prob")
model_preds2 = model_preds
model_preds$ensemble = predict(glm_ensemble, newdata = validation, type="prob")
model_preds2$ensemble = predict(glm_ensemble, newdata = validation)
CF = coef(glm_ensemble$ens_model$finalModel)[-1]
colAUC(model_preds$ensemble, validation$Transported)
confusionMatrix(model_preds2$ensemble, validation$Transported)

gbm_ensemble = caretStack(
  model_list,
  method="gbm",
  metric="Sens",
  trControl = cvCtrl
)

model_preds3 = model_preds
model_preds4 = model_preds
model_preds3$ensemble = predict(gbm_ensemble, newdata=validation, type="prob")
model_preds4$ensemble = predict(gbm_ensemble, newdata=validation)
colAUC(model_preds3$ensemble, validation$Transported)
confusionMatrix(model_preds4$ensemble, validation$Transported)

# ROC ---------------------------------------------------------------------

#logit
y = validation$Transported
y = ifelse(y == "False", 1, 0)
glmpredR = prediction(glmpred_p[,1], y)
roc_log = performance(glmpredR, measure = "tpr", x.measure = "fpr")
plot(roc_log)
abline(a=0, b= 1)

#knn
knnpredR = prediction(KNNPred_p[,1], y)
roc_knn = performance(knnpredR, measure = "tpr", x.measure = "fpr")
plot(roc_knn)
abline(a=0, b= 1)

#sda
sdaPredR = prediction(sdaPred_p[,1], y)
roc_sda = performance(sdaPredR, measure = "tpr", x.measure = "fpr")
plot(roc_sda)
abline(a=0, b= 1)

#lasso
lassoPredR = prediction(lassoPred_p[,1], y)
roc_lasso = performance(lassoPredR, measure = "tpr", x.measure = "fpr")
plot(roc_lasso)
abline(a=0, b= 1)

#pls
plsPredR = prediction(plsPred_p[,1], y)
roc_pls = performance(plsPredR, measure = "tpr", x.measure = "fpr")
plot(roc_pls)
abline(a=0, b= 1)

#naive
naivePredR = prediction(naivePred_p[,1], y)
roc_naive = performance(naivePredR, measure = "tpr", x.measure = "fpr")
plot(roc_naive)
abline(a=0, b= 1)

#tree
treePredR = prediction(treePred_pruned_p[,1], y)
roc_tree = performance(treePredR, measure = "tpr", x.measure = "fpr")
plot(roc_tree)
abline(a=0, b= 1)

#gb
gbPredR = prediction(gbPred_p[,1], y)
roc_gb = performance(gbPredR, measure = "tpr", x.measure = "fpr")
plot(roc_gb)
abline(a=0, b= 1)

#rf
rfPredR = prediction(rfPred_p[,1], y)
roc_rf = performance(rfPredR, measure = "tpr", x.measure = "fpr")
plot(roc_rf)
abline(a=0, b= 1)

#nn 
nnPredR = prediction(nnPred_p[,1], y)
roc_nn = performance(nnPredR, measure = "tpr", x.measure = "fpr")
plot(roc_nn)
abline(a=0, b= 1)

#nn tuned
nn_tunedPredR = prediction(nn_tunedPred_p[,1], y)
roc_nn_tuned = performance(nn_tunedPredR, measure = "tpr", x.measure = "fpr")
plot(roc_nn_tuned)
abline(a=0, b= 1)

#glm stack
glm_sPredR = prediction(model_preds$ensemble, y)
roc_glm_s = performance(glm_sPredR, measure = "tpr", x.measure = "fpr")
plot(roc_glm_s)
abline(a=0, b= 1)

#gbm stack
gbm_sPredR = prediction(model_preds3$ensemble, y)
roc_gbm_s = performance(gbm_sPredR, measure = "tpr", x.measure = "fpr")
plot(roc_gbm_s)
abline(a=0, b= 1)

#bagging
baggingPredR = prediction(baggingPred_p[,1], y)
roc_bagging = performance(baggingPredR, measure = "tpr", x.measure = "fpr")
plot(roc_bagging)
abline(a=0, b= 1)

plot(roc_log, col = "dodgerblue", lwd = 2) 
par(new = TRUE)
plot(roc_gb, col = "darkorange", lwd = 2) 
par(new = TRUE)
plot(roc_rf, col = "green", lwd = 2) 
par(new = TRUE)
plot(roc_nn, col = "purple", lwd = 2) 
par(new = TRUE)
plot(roc_knn, col = "yellow4", lwd = 2)
par(new = TRUE)
plot(roc_glm_s, col = "red", lwd = 2) 
par(new = TRUE)


legend("bottomright", legend=c("logit", "gb", "rf", "nn", "knn", "glmstack"),
       col=c("dodgerblue", "darkorange", "green", "purple", "yellow4", "red"),
       lty = 1, cex = 0.7, text.font=4, y.intersp=0.5, x.intersp=0.1, lwd = 3)

# LIFT --------------------------------------------------------------------

copy = train
copy$glm = predict(glm, copy, type = c("prob"))[,1]
gain_lift(data = copy, score='glm', target='Transported')

copy = train
copy$knn = predict(knn, copy, type = c("prob"))[,1]
gain_lift(data = copy, score='knn', target='Transported')

copy = train
copy$lasso = predict(lasso, copy, type = c("prob"))[,1]
gain_lift(data = copy, score='lasso', target='Transported')

copy = train
copy$pls = predict(pls, copy, type = c("prob"))[,1]
gain_lift(data = copy, score='pls', target='Transported')

copy = train
copy$gb = predict(gb, copy, type = c("prob"))[,1]
gain_lift(data = copy, score='gb', target='Transported')

copy = train
copy$nn = predict(nn, copy, type = c("prob"))[,1]
gain_lift(data = copy, score='nn', target='Transported')

copy = train
copy$glm_s = predict(glm_ensemble, copy, type = c("prob"))
gain_lift(data = copy, score='glm_s', target='Transported')

copy = train
copy$rf = predict(rf, copy, type = c("prob"))[,1]
gain_lift(data = copy, score='rf', target='Transported')

#selezioniamo staking

