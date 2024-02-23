#!/bin/bash
javac -cp h2o-genmodel.jar:-J-Xms2g sla_binary.java
# Run the resulting executable with the following command, providing the feature vector via args:
# java -cp .:h2o-genmodel.jar main 0 2 1 3 0
