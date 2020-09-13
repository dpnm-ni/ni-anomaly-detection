#!/bin/bash
#javac -cp h2o-genmodel.jar:-J-Xms2g sla_binary.java

javac -cp models/sla_binary/h2o-genmodel_sla.jar:-J-Xms2g sla_binary.java #sla_binary compile
echo "sla_binary model compile complete"

javac -cp models/sla_rcl/h2o-genmodel_rcl.jar:-J-Xms2g sla_rcl.java #sla_rcl compile
echo "sla_rcl model compile complete"

javac -cp models/resource_overload/h2o-genmodel_overload.jar:-J-Xms2g resource_overload.java #sla_rcl compile
echo "overload model compile complete"

#java -cp .:models/resource_overload/h2o-genmodel_overload.jar resource_overload 0 38.551 0 0 487944.65 471069.57 2 39.197 0 0 542723.67 544092.75 0 37.231 0 0 563432.77 543408.56 0.5 69.234 0 0 510342.71 521637.12 1.996 63.799 0 0 506293.13 516719.66


# Run the resulting executable with the following command, providing the feature vector via args:
# java -cp .:h2o-genmodel.jar main 0 2 1 3 0
