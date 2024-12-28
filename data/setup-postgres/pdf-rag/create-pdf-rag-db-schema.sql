-- DELETE Database if it exists
DROP DATABASE pdf_rag;
-- Create database if it doesn't exist
CREATE DATABASE pdf_rag;

-- Switch to the new database
\c pdf_rag;
-- Create PGVector Extension
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA pg_catalog;
-- Create schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS dev;

CREATE TABLE IF NOT EXISTS dev.pdf_rag_data (
    source VARCHAR,
    page_content VARCHAR,
    page_number INTEGER,
    page_content_embeddings vector(768)
);
SELECT column_name, data_type, character_maximum_length FROM information_schema.columns WHERE table_schema = 'dev' AND table_name = 'pdf_rag_data';

\d dev.pdf_rag_data