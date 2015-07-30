#!/root/spark/bin/sparkR
##!/usr/bin/Rscript
### CV
# example : ~> Rscript ML_parallel.R local[10] cv_small.txt

# stop when there is an warning
ptm <- proc.time()

options(warn=2)

# load library
library(BayesTree, quietly=TRUE)
## install SparkR
#library(devtools)
#install_github("amplab-extras/SparkR-pkg", subdir="pkg")

library(SparkR)

# read in data
nir_wet_chem <- read.csv("wet_chem_withNIR.csv", stringsAsFactors = FALSE)

# read in command arguments
args <- commandArgs(trailingOnly=TRUE)
if(length(args) < 2){
   print("Usage : ML_parallel.R <master> <cv_file> <nslices=1>")
   print("Example : Rscript ML_parallel.R local[10] cv_small.txt")
   q("no")
}

# get list of cross validation parameters
cvpar = readLines(args[[2]])

#Initialize Spark context 
sc <- sparkR.init(args[[1]], "bartcv")
#sc <- sparkR.init("spark://ec2-54-158-78-86.compute-1.amazonaws.com:7077", "bartcv")

cv_bart <- function(indata) {
   print("In cv_bart")
   print(indata)
   params <- strsplit(indata, split=",")[[1]]
   # column 2 to 1040 are the nir
   nir_carbon_data <- cbind(depth = nir_wet_chem[, "Depth.x"], carbon = nir_wet_chem[, params[1]], nir_wet_chem[,2:1040])
   nir_carbon_data <- na.omit(nir_carbon_data)
   
   # top or sub soil
   nir_carbon_data_top <- nir_carbon_data[nir_carbon_data$depth == "Topsoil",-1]
   
   nfolder <- 5 # 5 folder CV
   
   # delete missing data
   Y <- nir_carbon_data_top[, 1]
   X <- nir_carbon_data_top[, -1]
   Y <- log(Y)
   
   X_temp <- cbind(NA, X)
   X_temp <- X_temp[, -dim(X_temp)[2]]
   X_der <- X-X_temp
   X_der <- X_der[,-1]
   
   # cv parameters
   ntree.cv = as.numeric(params[5])
   nfolder <- 5 #5 fold cv 
   sigdf.quant= c(as.numeric(params[2]), as.numeric(params[3]))
   k.cv = as.numeric(params[4])
   
   # set random seed
   set.seed(params[6])
   sizetest <- ceiling(length(Y)/nfolder)
   print(sizetest)
   print(length(Y))
   testindex <- sort(base:::sample(1:length(Y), size=sizetest))
   x.train <- X[-testindex, ]
   y.train <- Y[-testindex]
   
   x.test <- X[testindex, ]
   y.test <- Y[testindex]	
   
   # run bart	   
   library(BayesTree, quietly=TRUE)
   temp.bart <- bart(x.train, y.train, x.test, sigest = sd(y.train), sigdf = sigdf.quant[1], sigquant=sigdf.quant[2], k = k.cv, verbose=FALSE, usequant=TRUE, ntree = ntree.cv, ndpost=500, keepevery=10, nskip=50)

   prediction.error.sse <- sum((y.test - temp.bart$yhat.test.mean)^2)/sum((y.test-mean(y.test))^2)

   myres = paste(params[1],params[2],  params[3], params[4], params[5], params[6], prediction.error.sse)
   #return(myres)
}

nslices = 1
if(length(args) == 3) {nslices = args[[3]]}

rdd <- SparkR:::parallelize(sc, coll=cvpar, numSlices = nslices)
myerr <- SparkR:::flatMap(rdd,cv_bart)
output <- SparkR:::collect(myerr)
print(output)
proc.time() - ptm
