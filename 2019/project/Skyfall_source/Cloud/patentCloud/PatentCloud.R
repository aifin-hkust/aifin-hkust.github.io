rm(list=ls())
text17 <- scan("C:/Users/Crystal/Dropbox/AI-optionB/Cloud/patentCloud/Patent17.txt", what = character(0))
text18 <- scan("C:/Users/Crystal/Dropbox/AI-optionB/Cloud/patentCloud/Patent18.txt", what = character(0), sep = ",")
text19 <- scan("C:/Users/Crystal/Dropbox/AI-optionB/Cloud/patentCloud/Patent19.txt", what = character(0))
install.packages('jiebaR')
library(jiebaR)
# Extract the contents
fenci <- worker(stop_word = 'C:/Users/Crystal/Dropbox/AI-optionB/Cloud/patentCloud/stopwords/hit.txt')
t17 <- segment(text17, fenci)
t18 <- segment(text18, fenci)
t19 <- segment(text19, fenci)
# Calculate frequency of words
wordsT17 <- data.frame(table(t17), row.names = NULL)
wordsT17 <- wordsT17[order(wordsT17[,2], decreasing = T),]
wordsT17 <- wordsT17[!wordsT17[,2]<2, ]
print(wordsT19[1:10,])
wordsT18 <- data.frame(table(t18), row.names = NULL)
wordsT18 <- wordsT18[order(wordsT18[,2], decreasing = T),]
wordsT18 <- wordsT18[!wordsT18[,2]<2, ]
wordsT19 <- data.frame(table(t19), row.names = NULL)
wordsT19 <- wordsT19[order(wordsT19[,2], decreasing = T),]
wordsT19 <- wordsT19[!wordsT19[,2]<2, ]
# install.packages('devtools')
# devtools::install_github("lchiffon/wordcloud2")
library(wordcloud2)
figPath <- system.file("examples/batman.png", package = "wordcloud2")
# D:/Program Files/R/R-3.3.0/library/wordcloud2/examples/batman.png"
wordcloud2(wordsT17, figPath = figPath, size = 2)
wordcloud2(wordsT18, figPath = figPath, size = 2)
wordcloud2(wordsT19, figPath = figPath, size = 2)
