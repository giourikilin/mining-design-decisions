@echo off

if "%1" == "up" (
    docker network create ml-network
    docker-compose -f docker-compose-spark.yml up
)

if "%1" == "down" (
    docker-compose -f docker-compose-spark.yml down
    docker-compose -f docker-compose-driver.yml down
    docker network rm ml-network
)

if "%1" == "kill" (
    docker-compose -f docker-compose-spark.yml kill
    docker-compose -f docker-compose-driver.yml kill
    docker network rm ml-network
)

if "%1" == "submit" (
    docker-compose -f docker-compose-driver.yml up --build
)