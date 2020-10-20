

# set default path
setwd("C:/Users/jbhong/Documents/GitHub/ni-anomaly-detection-public")
getwd()


# install h2o package (if h2o is already installed, skip this)
#install.packages("h2o")

library(h2o)

# use localhost h2o platform
#h2o.init()

# connect the h2o platform in AI Node
h2o.connect(ip = "141.223.82.61", port = 54321)


library(tidyverse)

# read csv file
test_dataset_01 <- read_csv("data/test_dataset_01.csv")
test_dataset_02 <- read_csv("data/test_dataset_02.csv")
test_dataset_03 <- read_csv("data/test_dataset_03.csv")


# convert categorical metric
test_dataset_01 <- test_dataset_01 %>%
    mutate(
        SLA_label = factor(SLA_label)
    )

test_dataset_02 <- test_dataset_02 %>%
    mutate(
        SLA_label = factor(SLA_label)
    )

test_dataset_03 <- test_dataset_03 %>%
    mutate(
        SLA_label = factor(SLA_label)
    )


# upload to h2o platform
test_dataset_01.h2o <-  h2o.splitFrame(
    data = as.h2o(test_dataset_01),
    ratios = 0.5,
    destination_frames = c("test_dataset_01_part1", "test_dataset_01_part2"),
    seed=12220
)

test_dataset_02.h2o <-  h2o.splitFrame(
    data = as.h2o(test_dataset_02),
    ratios = 0.5,
    destination_frames = c("test_dataset_02_part1", "test_dataset_02_part2"),
    seed=12221
)

test_dataset_03.h2o <-  h2o.splitFrame(
    data = as.h2o(test_dataset_03),
    ratios = 0.5,
    destination_frames = c("test_dataset_03_part1", "test_dataset_03_part2"),
    seed=12222
)
