-- DELETE Database if it exists
DROP DATABASE mylibrary;
-- Create database if it doesn't exist
CREATE DATABASE mylibrary;

-- Switch to the new database
\c mylibrary;

-- Create schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS dev;

CREATE TABLE IF NOT EXISTS dev.books (
    ISBN VARCHAR PRIMARY KEY,
    Title VARCHAR,
    Edition VARCHAR,
    PublishedYear INTEGER,
    Publisher VARCHAR,
    Authors VARCHAR[],
    Tags VARCHAR[]
);
