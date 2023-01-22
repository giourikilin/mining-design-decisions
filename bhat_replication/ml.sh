#!/bin/sh
if [ "$1" = "up" ]; then
    docker network create ml-network
    docker-compose -f docker-compose-spark.yml up
fi

if [ "$1" = "down" ]; then
    docker-compose -f docker-compose-spark.yml down
    docker-compose -f docker-compose-driver.yml down
    docker network rm ml-network
fi

if [ "$1" = "kill" ]; then
    docker-compose -f docker-compose-spark.yml kill
    docker-compose -f docker-compose-driver.yml kill
    docker network rm ml-network
fi

if [ "$1" = "submit" ]; then
    docker-compose -f docker-compose-driver.yml up --build
fi