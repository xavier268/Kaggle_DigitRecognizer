# makePredictionsAndSave


if(!exists("model")) { source("MLPModel.R")}

preds <- predict(model, test, array.layout = "rowmajor")
dim(preds) # 10 x 28 000 - Careful ! - default for maxnet is to put samples in columns ...

preds.y <- max.col(t(preds)) - 1 # vector 28 000 answers
submission <- data.frame(1:length(preds.y),preds.y)
colnames(submission) <- c("ImageId","Label")

fname <- paste0("SubmissionTest", date(),".csv")
write.csv(submission, file = fname, quote = FALSE, row.names = FALSE )


  


