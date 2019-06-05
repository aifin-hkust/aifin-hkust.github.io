rm(list=ls())
text17 <- scan("Software17.txt", what = character(0), sep = ",")
text18 <- scan("Software18.txt", what = character(0), sep = ",")
text19 <- scan("Software19.txt", what = character(0), sep = ",")
# install.packages('jiebaR')
library(jiebaR)
# Extract the contents
fenci <- worker()
new_user_word(fenci, c('3D', '2D', 'ÒÆ¶¯¶Ë', 'ÈËÁ³½âËø'))
t17 <- segment(text17, fenci)
t18 <- segment(text18, fenci)
t19 <- segment(text19, fenci)
tAll <- c(t17, t18, t19)
# Calculate frequency of words
wordsTable <- data.frame(table(tAll), row.names = NULL)
wordsTable <- wordsTable[order(wordsTable[,2], decreasing = T),]
# install.packages('devtools')
# devtools::install_github("lchiffon/wordcloud2")
library(wordcloud2)
figPath <- system.file("examples/batman.png", package = "wordcloud2")
# D:/Program Files/R/R-3.3.0/library/wordcloud2/examples/batman.png"
wordcloud2(wordsTable, figPath = figPath, size = 3.5)
