#!/bin/bash
#docker exec -it vectordb-lab-postgres-db psql -U postgres -c "CREATE DATABASE imdb;"
# Executing SQL Script. Setting up the Database and Table. 
docker exec -it vectordb-lab-postgres-db psql -U postgres -d postgres -f /sample-data/setup-postgres/mylibrary/create-mylibrary-db-schema.sql
